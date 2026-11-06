"""Query-Adaptive 时序检索（E2，本项目差异化算法，VideoTree 启发）。

流程（开发说明书 §7.3）：
1. coarse_search：字幕（BM25）+ 语义（CLIP/Mock）召回候选窗口，RRF 融合 + IoU-NMS
   去重；两者都不可用时退化为均匀粗帧 VLM 检索。
2. 粗观察：对每个候选窗口内取 coarse_frames_per_window 帧，VLM 判定相关性，
   用观察到的相关率更新窗口分数（E1 语义召回 + VLM 的效果）。
3. estimate_uncertainty：基于证据一致性（ASR/视觉）、窗口宽度、观察密度的
   **启发式代理指标**（非校准概率）。
4. refine：对最不确定的窗口局部加密抽帧（coarse_fps），VLM 再判定，用相关帧的
   时间跨度收窄窗口边界（VideoTree 深度扩展的简化：二分缩窄 -> 局部补帧）。
5. stop：不确定度低于阈值 / 帧预算耗尽 / 达到 max_refinement_rounds。
每轮记录 before_window → sampled_frames → new_window → reason（refinement_rounds）。

所有 VLM 观察共享同一帧预算（budget_frames），预算耗尽即停并标记
budget_exhausted（说明书 §9.1 公平性）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import Config
from ..embeddings.base import EmbeddingAdapter
from ..ingest.frames import extract_local_frames, frames_within
from ..ingest.pipeline import VideoIndex
from ..logging_utils import TraceWriter
from ..models.interface import VLMAdapter, VLMError
from ..schemas import Evidence, SearchResult, TimeWindow, VideoFrame
from .fusion import nms_dedupe, rrf_fuse
from .semantic import semantic_search
from .subtitle import subtitle_search
from .uniform import uniform_search
from .window_growth import expand_to_shots

__all__ = ["adaptive_search", "estimate_uncertainty"]


@dataclass
class _WinState:
    window: TimeWindow
    asr_support: bool = False
    visual_support: bool = False
    observed: int = 0
    score: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    settled: bool = False  # 细化无变化后标记收敛，不再重复烧预算


def _grow_window(
    st: _WinState,
    index_frames: list[VideoFrame],
    duration_sec: float,
    vlm: VLMAdapter,
    query: str,
    config: Config,
    used: list[str],
    frame_budget: int,
    vlm_calls: int,
    max_vlm_calls: int | None,
    trace: TraceWriter | None,
    method: str,
) -> tuple[list[str], int, list[str], dict | None]:
    """证据驱动的窗口生长（VideoTree 宽度扩展启发，简化版）。

    针对实测"62% 有重叠未命中 = 窗口在 GT 内但过窄"（EXPERIMENTS.md §6）：
    对粗观察发现相关帧的窗口，把窗口边缘之外的相邻索引帧交给 VLM 判定，
    相关则扩展窗口（±window_pad_sec），首个非相关帧即停止，每侧 ≤ max_steps。

    返回 (新增已用帧, 新增 VLM 调用, 警告, rounds_log 条目|None)。
    """
    gcfg = config.retrieval.adaptive.grow
    w = st.window
    n_coarse_rel = len(st.evidence)  # 粗观察相关帧数（evidence 只收相关帧）
    if n_coarse_rel / max(st.observed, 1) < gcfg.min_relevant_rate:
        return [], 0, [], None
    frames_left = frame_budget - len(used)
    call_budget = max_vlm_calls - vlm_calls if max_vlm_calls is not None else None
    if frames_left <= 0 or (call_budget is not None and call_budget <= 0):
        return [], 0, ["帧/VLM 预算耗尽，跳过窗口生长"], None

    # 边缘相邻帧：左侧按时间倒序（最近的在前），右侧正序
    used_set = set(used)
    left = [f for f in index_frames if f.timestamp_sec < w.start_sec and f.frame_id not in used_set]
    right = [f for f in index_frames if f.timestamp_sec >= w.end_sec and f.frame_id not in used_set]
    left = sorted(left, key=lambda f: -f.timestamp_sec)[: gcfg.max_steps_per_side]
    right = sorted(right, key=lambda f: f.timestamp_sec)[: gcfg.max_steps_per_side]
    cand = left[::-1] + right  # 恢复全局时间序（供 _judge 分块）
    cand = cand[: max(0, frames_left)]
    if not cand:
        return [], 0, [], None

    rel, calls, errs = _judge(vlm, query, cand, trace, f"{method}:grow",
                              call_budget=call_budget)
    if not rel:
        return [f.frame_id for f in cand], calls, errs, None  # 相邻帧全不相关：窗口不变
    rel_ids = {f.frame_id for f in rel}

    # 连续性行走：从窗口边缘向外，包含连续相关帧，首个非相关帧停止
    pad = config.retrieval.uniform.window_pad_sec
    new_start, new_end = w.start_sec, w.end_sec
    for f in sorted(left, key=lambda f: f.timestamp_sec, reverse=True):  # 最近优先
        if f.frame_id not in rel_ids:
            break
        new_start = max(0.0, f.timestamp_sec - pad)
    for f in right:  # 已按时间正序 = 最近优先
        if f.frame_id not in rel_ids:
            break
        new_end = min(duration_sec, f.timestamp_sec + pad)
    new_start = min(new_start, w.start_sec)
    new_end = max(new_end, w.end_sec)
    if new_end - new_start > config.retrieval.max_window_sec:
        new_end = new_start + config.retrieval.max_window_sec

    if (new_start, new_end) == (w.start_sec, w.end_sec):
        return [f.frame_id for f in cand], calls, errs, None  # 无连续相关帧可扩展

    before = (round(w.start_sec, 3), round(w.end_sec, 3))
    st.window = TimeWindow(
        start_sec=round(new_start, 3), end_sec=round(new_end, 3),
        score=round((n_coarse_rel + len(rel)) / (st.observed + len(cand)), 4),
        evidence_ids=list(w.evidence_ids),
    )
    st.score = st.window.score
    st.observed += len(cand)
    st.evidence += [
        Evidence(
            evidence_id=f"frame_{f.frame_id}", type="frame",
            timestamp_sec=f.timestamp_sec, source_file=f.image_path,
            model_id=vlm.model_id, score=1.0,
        )
        for f in rel
    ]
    grow_log = {
        "round": -2,
        "before_window": [before[0], before[1]],
        "sampled_frames": [f.frame_id for f in cand],
        "after_window": [round(new_start, 3), round(new_end, 3)],
        "uncertainty": None,
        "reason": (f"生长扩展: [{before[0]:.2f},{before[1]:.2f}) -> "
                   f"[{new_start:.2f},{new_end:.2f})（{len(rel)}/{len(cand)} 帧相关）"),
    }
    if trace:
        trace.event("adaptive_grow",
                    before=[before[0], before[1]],
                    after=[new_start, new_end],
                    sampled=[f.frame_id for f in cand],
                    relevant=[f.frame_id for f in rel])
    return [f.frame_id for f in cand], calls, errs, grow_log


def estimate_uncertainty(
    st: _WinState,
    duration_sec: float,
    config: Config,
) -> float:
    """不确定度启发式代理（0~1，越大越不确定；**不是校准概率**，见 docs/DECISIONS.md）。

    - 证据一致性：ASR 与视觉都支持=0，单一=0.5，都无=1.0（权重 0.4）
    - 窗口宽度：width / uncertainty_width_reference_sec（权重 0.3）——
      宽窗口意味着时间边界粗，值得局部加密抽帧收窄
    - 观察密度：相对细化目标密度（coarse_fps）的覆盖率（权重 0.3）
    """
    cfg = config.retrieval
    w = st.window
    if st.asr_support and st.visual_support:
        agreement = 0.0
    elif st.asr_support or st.visual_support:
        agreement = 0.5
    else:
        agreement = 1.0
    width_n = min(1.0, w.duration_sec / max(cfg.adaptive.uncertainty_width_reference_sec, 1e-9))
    expect = max(1.0, w.duration_sec * cfg.adaptive.coarse_fps)
    coverage = min(1.0, st.observed / expect)
    u = 0.4 * agreement + 0.3 * width_n + 0.3 * (1.0 - coverage)
    return round(min(1.0, u), 4)


def _judge(
    vlm: VLMAdapter,
    query: str,
    frames: list[VideoFrame],
    trace: TraceWriter | None,
    method: str,
    call_budget: int | None = None,
) -> tuple[list[VideoFrame], int, list[str]]:
    """VLM 分块判定帧相关性；返回 (相关帧, 调用数, 错误)。

    call_budget：本次最多允许的 VLM 调用数（分块数），用于 Agent 总调用预算。
    """
    relevant: list[VideoFrame] = []
    calls = 0
    errors: list[str] = []
    max_chunk = getattr(vlm, "max_frames_per_call", 8)
    try:
        for i in range(0, len(frames), max_chunk):
            if call_budget is not None and calls >= call_budget:
                break
            chunk = frames[i : i + max_chunk]
            res = vlm.inspect_frames(query, chunk)
            calls += 1
            if trace:
                trace.event(
                    "vlm_call",
                    method=method,
                    model=vlm.model_id,
                    frames=[f.frame_id for f in chunk],
                    usage=getattr(vlm, "last_usage", {}),
                    relevant=[j.frame_id for j in res.judgments if j.relevant],
                )
            rel_ids = set(res.relevant_frame_ids)
            relevant.extend(f for f in chunk if f.frame_id in rel_ids)
    except VLMError as exc:
        errors.append(f"VLM 调用失败: {exc}")
    return relevant, calls, errors


def adaptive_search(
    video_id: str,
    query: str,
    index: VideoIndex,
    vlm: VLMAdapter,
    embedder: EmbeddingAdapter | None,
    config: Config,
    budget_frames: int | None = None,
    max_rounds: int | None = None,
    max_vlm_calls: int | None = None,
    frame_embeddings=None,
    method: str = "adaptive",
    trace: TraceWriter | None = None,
) -> SearchResult:
    t0 = time.perf_counter()
    rcfg = config.retrieval
    acfg = rcfg.adaptive
    budget = budget_frames or rcfg.frame_budget
    max_rounds = rcfg.max_refinement_rounds if max_rounds is None else max_rounds
    duration = index.asset.duration_sec
    warnings: list[str] = []
    rounds_log: list[dict] = []
    used: list[str] = []
    vlm_calls = 0
    evidence: list[Evidence] = []

    def _call_budget() -> int | None:
        """本次可用的 VLM 调用数（None = 不限制）。"""
        if max_vlm_calls is None:
            return None
        return max(0, max_vlm_calls - vlm_calls)

    # ---- 1. 粗召回（不耗视觉预算）----------------------------------------
    sub_res = subtitle_search(video_id, query, index.subtitles, duration, config) if index.subtitles else None
    sem_res = None
    if embedder is not None:
        sem_res = semantic_search(video_id, query, index.frames, duration, embedder, config,
                                  frame_embeddings=frame_embeddings, scenes=index.scenes)
        evidence.extend(sem_res.evidence)

    cand: list[TimeWindow] = []
    asr_keys: set[tuple[float, float]] = set()
    if sub_res and sub_res.windows:
        cand.extend(sub_res.windows)
        asr_keys.update((round(w.start_sec, 3), round(w.end_sec, 3)) for w in sub_res.windows)
        evidence.extend(sub_res.evidence)
    if sem_res and sem_res.windows:
        cand.extend(sem_res.windows)
    if len(cand) > 1 and sub_res and sem_res:
        cand = rrf_fuse(
            [("subtitle", sub_res.windows), ("semantic", sem_res.windows)],
            rrf_k=rcfg.fusion_config.rrf_k,
        )
    cand = nms_dedupe(cand, rcfg.fusion_config.nms_iou)[: rcfg.top_k_windows]

    if not cand:
        # 退路：无任何候选（无字幕且无嵌入）时退化为均匀粗帧检索（VideoMind 风格 fallback）
        warnings.append("无字幕且无语义候选，退化为 uniform 粗检索")
        fallback = uniform_search(
            video_id, query, index.frames, duration, vlm, config,
            budget_frames=min(budget, max(rcfg.frame_budget, 8)),
            trace=trace, scenes=index.scenes,
        )
        fallback.budget_exhausted = fallback.budget_exhausted or (len(fallback.used_frame_ids) >= budget)
        return fallback.model_copy(
            update={"method": method, "warnings": warnings + fallback.warnings}
        )

    states = [
        _WinState(
            window=w,
            asr_support=(round(w.start_sec, 3), round(w.end_sec, 3)) in asr_keys,
            score=w.score or 0.0,
        )
        for w in cand
    ]

    # ---- 2. 粗观察：每候选窗口内 coarse_frames_per_window 帧 --------------
    for st in states:
        if len(used) >= budget or (max_vlm_calls is not None and vlm_calls >= max_vlm_calls):
            warnings.append(f"帧预算 {budget} 或 VLM 调用预算在粗观察阶段耗尽")
            break
        wf = frames_within(index.frames, st.window.start_sec, st.window.end_sec)
        take = wf[: min(acfg.coarse_frames_per_window, budget - len(used))]
        if not take:
            continue
        rel, calls, errs = _judge(vlm, query, take, trace, f"{method}:coarse", _call_budget())
        vlm_calls += calls
        warnings.extend(errs)
        used.extend(f.frame_id for f in take)
        st.observed = len(take)
        if rel:
            st.visual_support = True
            st.score = len(rel) / len(take)
            st.evidence = [
                Evidence(
                    evidence_id=f"frame_{f.frame_id}", type="frame",
                    timestamp_sec=f.timestamp_sec, source_file=f.image_path,
                    model_id=vlm.model_id, score=1.0,
                )
                for f in rel
            ]
            evidence.extend(st.evidence)
            if acfg.grow.enabled:
                n_evidence_before = len(st.evidence)
                grow_used, grow_calls, grow_errs, grow_log = _grow_window(
                    st, index.frames, duration, vlm, query, config,
                    used, budget, vlm_calls, max_vlm_calls, trace, method,
                )
                vlm_calls += grow_calls
                warnings.extend(grow_errs)
                used.extend(grow_used)
                evidence.extend(st.evidence[n_evidence_before:])
                if grow_log:
                    rounds_log.append(grow_log)

    # 粗观察剪枝：纯视觉候选（无字幕支持）被 VLM 观察否定 -> 丢弃而非继续细化；
    # 有字幕支持的窗口保留（视觉观察与 ASR 证据冲突时以 ASR 候选存续，交由后续
    # 验证器/评测裁决——P3 Verifier 会处理该冲突）。
    pruned: list[tuple[_WinState, str]] = []
    for st in states:
        if st.observed > 0 and not st.visual_support and not st.asr_support:
            pruned.append((st, "粗观察未发现相关帧，丢弃候选"))
    for st, reason in pruned:
        states.remove(st)
        rounds_log.append(
            {
                "round": -1,
                "before_window": [st.window.start_sec, st.window.end_sec],
                "sampled_frames": [],
                "after_window": None,
                "uncertainty": None,
                "reason": reason,
            }
        )
        if trace:
            trace.event("adaptive_prune", window=[st.window.start_sec, st.window.end_sec], reason=reason)
    if not states:
        warnings.append("所有候选在粗观察阶段被否定（证据不足）")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            used_frame_ids=used, vlm_calls=vlm_calls,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            evidence=evidence, warnings=warnings, refinement_rounds=rounds_log,
        )

    # ---- 3. 细化循环 ------------------------------------------------------
    budget_exhausted = False
    for r in range(max_rounds):
        if len(used) >= budget:
            budget_exhausted = True
            warnings.append(f"帧预算 {budget} 耗尽，提前结束细化（第 {r} 轮）")
            break
        unsettled = [s for s in states if not s.settled]
        if not unsettled:
            break  # 全部收敛
        target = max(unsettled, key=lambda s: estimate_uncertainty(s, duration, config))
        u = estimate_uncertainty(target, duration, config)
        if u < acfg.uncertainty_threshold:
            break  # 所有窗口都已足够确定（max 不确定度低于阈值）
        before = target.window
        new_frames = extract_local_frames(
            index.asset.path,
            index.cache_dir / "refine",
            before.start_sec,
            before.end_sec,
            acfg.coarse_fps,
            max_side=config.video.max_frame_side_px,
            tag=f"r{r}_w{states.index(target)}",
        )
        take = new_frames[: max(0, budget - len(used))]
        if not take:
            budget_exhausted = True
            warnings.append(f"帧预算 {budget} 耗尽，无法细化窗口 [{before.start_sec:.1f},{before.end_sec:.1f})")
            break
        rel, calls, errs = _judge(vlm, query, take, trace, f"{method}:refine", _call_budget())
        vlm_calls += calls
        warnings.extend(errs)
        used.extend(f.frame_id for f in take)
        target.observed += len(take)

        after = before
        reason = "细化未发现相关帧，保留原窗口（视觉证据不足）"
        if rel:
            s = min(f.timestamp_sec for f in rel)
            e = max(f.timestamp_sec for f in rel)
            pad = rcfg.uniform.window_pad_sec
            raw = TimeWindow(
                start_sec=max(before.start_sec, s - pad),
                end_sec=min(before.end_sec, e + pad),
                score=len(rel) / len(take),
                evidence_ids=before.evidence_ids,
            )
            # 与粗观察一致的镜头粒度：收窄到相关帧最集中的镜头（∩ 原窗口），
            # 避免把 shot 对齐窗口又缩回"帧跨度+固定外扩"的过窄粒度（QV 第二轮诊断）
            shot = expand_to_shots(
                raw, index.scenes, rcfg.max_window_sec,
                cluster_ts=[f.timestamp_sec for f in rel],
            )
            narrowed = TimeWindow(
                start_sec=max(raw.start_sec, shot.start_sec),
                end_sec=min(raw.end_sec, shot.end_sec),
                score=raw.score,
                evidence_ids=raw.evidence_ids,
            )
            if narrowed.duration_sec >= rcfg.min_window_sec:
                after = narrowed
                target.window = after
                target.score = narrowed.score or 0.0
                target.visual_support = True
                reason = (
                    f"细化收窄: [{before.start_sec:.2f},{before.end_sec:.2f}) -> "
                    f"[{after.start_sec:.2f},{after.end_sec:.2f})"
                    f"（{len(rel)}/{len(take)} 帧相关）"
                )
            else:
                reason = f"细化后窗口过窄（< {rcfg.min_window_sec}s），保留原窗口"
        if after.start_sec == before.start_sec and after.end_sec == before.end_sec:
            target.settled = True  # 细化无变化：标记收敛，避免重复烧预算
        rounds_log.append(
            {
                "round": r,
                "before_window": [round(before.start_sec, 3), round(before.end_sec, 3)],
                "sampled_frames": [f.frame_id for f in take],
                "after_window": [round(after.start_sec, 3), round(after.end_sec, 3)],
                "uncertainty": u,
                "reason": reason,
            }
        )
        if trace:
            trace.event(
                "adaptive_refine", round=r, uncertainty=u,
                before=[before.start_sec, before.end_sec],
                after=[after.start_sec, after.end_sec],
                sampled=[f.frame_id for f in take], reason=reason,
            )

    windows = sorted(states, key=lambda s: s.score, reverse=True)
    return SearchResult(
        video_id=video_id,
        query=query,
        # 窗口分数与状态分数同步（粗观察/细化会更新 st.score）
        windows=[st.window.model_copy(update={"score": st.score})
                 for st in windows[: rcfg.top_k_windows]],
        method=method,
        used_frame_ids=used,
        vlm_calls=vlm_calls,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        evidence=evidence,
        warnings=warnings,
        budget_exhausted=budget_exhausted,
        refinement_rounds=rounds_log,
    )
