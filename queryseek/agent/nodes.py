"""Agent 角色节点（VideoMind 启发：Planner/Grounder/Verifier，说明书 §7.4）。

第一版 = 1 个 VLM + 3 个功能角色节点（不把每个函数都变成 LLM 调用）：
- planner           功能节点：查询类型分类 + 初始策略确认（不调 VLM）
- initial_retrieve  功能节点：调用既有检索策略（subtitle/uniform/semantic/adaptive）
- ground            功能节点：refuted → 全视频重采样（失败恢复）；
                    insufficient → 局部加密补帧收窄边界
- verify            VLM 节点：verify_window 判定 top 候选（confirmed/refuted/insufficient）
- finalize          功能节点：汇总已确认窗口
- validate_artifact 功能节点：程序规则独立校验（不依赖 VLM 自评，说明书 §7.4）
"""

from __future__ import annotations

import time
from typing import Any

try:  # langgraph 可选依赖：节点仅在其存在时使用；LangGraph 只向标注为
    # RunnableConfig 的参数注入 config
    from langchain_core.runnables import RunnableConfig
except ImportError:  # pragma: no cover
    RunnableConfig = Any  # type: ignore[assignment,misc]

from ..config import Config
from ..ingest.frames import extract_local_frames, frames_within
from ..ingest.pipeline import load_index
from ..logging_utils import TraceWriter
from ..models.factory import build_vlm
from ..models.interface import VLMError
from ..retrieval.adaptive import adaptive_search
from ..retrieval.semantic import semantic_search
from ..retrieval.subtitle import subtitle_search
from ..retrieval.uniform import uniform_search
from ..schemas import TimeWindow
from ..tools.validate import is_valid_window
from .budget import BudgetGuard
from .state import dict_to_window, dicts_to_windows, window_to_dict, windows_to_dicts

_VISUAL_CUES = {"画面", "看", "颜色", "图形", "图像", "镜头", "出现", "穿着", "屏幕", "显示", "桌上", "手里"}
_SPEECH_CUES = {"说", "讲", "提到", "介绍", "解释", "讨论", "讲述"}


def _runtime(state: dict, config: RunnableConfig) -> tuple[Config, Any, Any, Any]:
    """从 configurable 重建运行时对象；测试可注入 vlm/embedder 替身。"""
    from ..embeddings.factory import build_embedder

    cfg = Config.model_validate(config["configurable"]["cfg"])
    index = load_index(state["video_id"], cfg)
    mock = bool(config["configurable"].get("mock", False))
    vlm = config["configurable"].get("vlm") or build_vlm(cfg, use_mock=mock, hints=index.content_hints)
    embedder = config["configurable"].get("embedder", "unset")
    if embedder == "unset":
        embedder = build_embedder(cfg, use_mock=mock, hints=index.content_hints, frames=index.frames)
    return cfg, index, vlm, embedder


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def planner(state: dict, config: RunnableConfig) -> dict:
    """查询类型启发式分类 + 初始策略确认（纯功能节点，零 VLM 调用）。"""
    query = state["query"]
    toks = set(query.lower())
    visual = bool(toks & _VISUAL_CUES)
    speech = bool(toks & _SPEECH_CUES)
    query_type = "visual" if visual and not speech else ("speech" if speech and not visual else "mixed")
    strategy = state["strategy"]
    rationale = {
        "visual": "查询含视觉线索词，优先视觉检索；字幕仅作候选",
        "speech": "查询含语音线索词，字幕检索可能更省预算（A0 对照见 docs/EXPERIMENTS.md）",
        "mixed": "混合查询：候选由多模态召回，VLM 验证",
    }[query_type]
    return {
        "plan": {
            "query_type": query_type,
            "strategy": strategy,
            "rationale": rationale,
            "steps": ["initial_retrieve", "verify", "ground*", "finalize", "validate_artifact"],
        },
        "status": "running",
    }


# ---------------------------------------------------------------------------
# initial_retrieve
# ---------------------------------------------------------------------------

def initial_retrieve(state: dict, config: RunnableConfig) -> dict:
    cfg, index, vlm, embedder = _runtime(state, config)
    strategy = state["strategy"]
    duration = index.asset.duration_sec
    query = state["query"]

    if strategy == "subtitle":
        res = subtitle_search(state["video_id"], query, index.subtitles, duration, cfg)
    elif strategy == "uniform":
        res = uniform_search(
            state["video_id"], query, index.frames, duration, vlm, cfg,
            budget_frames=state["budget"]["max_frames"], trace=_trace(config),
            scenes=index.scenes,
        )
    elif strategy == "semantic":
        res = semantic_search(state["video_id"], query, index.frames, duration, embedder, cfg)
    elif strategy == "adaptive":
        res = adaptive_search(
            state["video_id"], query, index, vlm, embedder, cfg,
            budget_frames=state["budget"]["max_frames"],
            max_vlm_calls=int(state["budget"]["max_vlm_calls"]),
            frame_embeddings=config["configurable"].get("frame_embeddings"),
            trace=_trace(config),
        )
    else:
        return {"fatal_errors": [f"未知策略 {strategy!r}"]}

    return {
        "candidate_windows": windows_to_dicts(res.windows),
        "evidence": [e.model_dump() for e in res.evidence],
        "used_frame_ids": list(res.used_frame_ids),
        "vlm_calls": state.get("vlm_calls", 0) + res.vlm_calls,
        "warnings": list(state.get("warnings", [])) + res.warnings,
        "budget_exhausted": bool(res.budget_exhausted),
        "last_observation": {"node": "initial_retrieve", "method": res.method,
                             "n_windows": len(res.windows),
                             "elapsed_ms": res.elapsed_ms},
    }


def _trace(config: RunnableConfig) -> TraceWriter | None:
    p = config["configurable"].get("trace_path")
    if not p:
        return None
    from pathlib import Path

    return TraceWriter(Path(p))


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(state: dict, config: RunnableConfig) -> dict:
    """VLM 验证 top 候选窗口；每次 verify_window 记 1 次 VLM 调用。"""
    _cfg, _index, vlm, _embedder = _runtime(state, config)
    guard = BudgetGuard(state["budget"])
    candidates = dicts_to_windows(state.get("candidate_windows", []))
    query = state["query"]
    results: list[dict[str, Any]] = []
    kept: list[TimeWindow] = []
    errors: list[str] = list(state.get("errors", []))
    vlm_calls = int(state.get("vlm_calls", 0))

    for w in candidates[:5]:
        if not guard.can_vlm_call(vlm_calls):
            break
        try:
            v = vlm.verify_window(query, w, state.get("evidence", [])[:20])
        except VLMError as exc:
            errors.append(f"verify VLM 失败: {exc}")
            v = None
        vlm_calls += 1
        if v is None:
            results.append({**window_to_dict(w), "verdict": "insufficient",
                            "confidence": 0.0, "reason": "VLM 调用失败，视为证据不足"})
            kept.append(w)  # 失败不武断丢弃
            continue
        results.append({**window_to_dict(w), "verdict": v.verdict,
                        "confidence": v.confidence, "reason": v.reason})
        if v.verdict == "confirmed" and v.confidence >= 0.5:
            kept.append(w)
        elif v.verdict == "insufficient" and v.confidence > 0:
            kept.append(w)  # 可细化
        # refuted：丢弃（ground 走失败恢复路径）

    if not kept and candidates:
        kept = [max(candidates, key=lambda w: w.score or 0.0)]  # 全部被否定时保留最优待恢复
    last = results[0] if results else {}
    return {
        "candidate_windows": windows_to_dicts(kept),
        "verifier_results": list(state.get("verifier_results", [])) + results,
        "last_verifier": last,
        "vlm_calls": vlm_calls,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# ground
# ---------------------------------------------------------------------------

def ground(state: dict, config: RunnableConfig) -> dict:
    """细化/恢复：refuted → 全视频重采样；insufficient → 局部加密补帧。"""
    cfg, index, vlm, _embedder = _runtime(state, config)
    guard = BudgetGuard(state["budget"])
    iteration = int(state.get("iteration", 0))
    used = list(state.get("used_frame_ids", []))
    vlm_calls = int(state.get("vlm_calls", 0))
    observations = list(state.get("observations", []))
    warnings = list(state.get("warnings", []))
    candidates = dicts_to_windows(state.get("candidate_windows", []))
    last = state.get("last_verifier", {})
    if not isinstance(last, dict):
        last = {}  # 状态边界防御（DECISIONS #43）：非 dict 一律按"无上次判定"处理
    duration = index.asset.duration_sec

    if not guard.can_iterate(iteration) or not guard.can_vlm_call(vlm_calls):
        return {"budget_exhausted": True,
                "warnings": warnings + ["预算耗尽，无法继续 ground"]}

    # ---- 路径 A：refuted / 无候选 → 全视频重采样（失败恢复）------------------
    if last.get("verdict") == "refuted" or not candidates:
        regrounds = sum(1 for o in observations if o.get("action") == "reground_uniform")
        if regrounds >= 1:
            return {"warnings": warnings + ["已重采样一次仍失败，停止恢复"],
                    "budget_exhausted": True}
        n = min(guard.frames_left(len(used)), max(8, state["budget"]["max_frames"] // 2))
        if n <= 0:
            return {"budget_exhausted": True, "warnings": warnings + ["帧预算耗尽"]}
        res = uniform_search(
            state["video_id"], state["query"], index.frames, duration, vlm, cfg,
            budget_frames=n, trace=_trace(config), scenes=index.scenes,
        )
        observations.append({
            "action": "reground_uniform", "iteration": iteration,
            "sampled_frames": res.used_frame_ids, "n_new_windows": len(res.windows),
        })
        return {
            "candidate_windows": windows_to_dicts(res.windows),
            "used_frame_ids": list(dict.fromkeys(used + res.used_frame_ids)),
            "vlm_calls": vlm_calls + res.vlm_calls,
            "iteration": iteration + 1,
            "observations": observations,
            "warnings": warnings + res.warnings,
            "last_observation": observations[-1],
        }

    # ---- 路径 B：insufficient → 局部加密补帧收窄边界 -------------------------
    target = candidates[0]
    frames_left = guard.frames_left(len(used))
    if frames_left <= 0:
        return {"budget_exhausted": True, "warnings": warnings + ["帧预算耗尽"]}
    new_frames = extract_local_frames(
        index.asset.path, index.cache_dir / "agent_refine",
        target.start_sec, target.end_sec, cfg.retrieval.adaptive.coarse_fps,
        max_side=cfg.video.max_frame_side_px,
        tag=f"ag{iteration}",
    )[:frames_left]
    rel_frames: list[Any] = []
    if new_frames:
        try:
            res = vlm.inspect_frames(state["query"], new_frames)
        except VLMError as exc:
            return {"errors": list(state.get("errors", [])) + [f"ground VLM 失败: {exc}"],
                    "iteration": iteration + 1,
                    "last_observation": {"action": "refine", "error": str(exc)}}
        vlm_calls += 1
        rel_frames = [f for f in new_frames if f.frame_id in set(res.relevant_frame_ids)]
        if _trace(config):
            _trace(config).event("agent_ground_refine", iteration=iteration,
                                 window=[target.start_sec, target.end_sec],
                                 sampled=[f.frame_id for f in new_frames],
                                 relevant=[f.frame_id for f in rel_frames])

    after = target
    reason = "细化未发现相关帧，保留原窗口"
    if rel_frames:
        s = min(f.timestamp_sec for f in rel_frames)
        e = max(f.timestamp_sec for f in rel_frames)
        pad = cfg.retrieval.uniform.window_pad_sec
        narrowed = TimeWindow(
            start_sec=max(target.start_sec, s - pad),
            end_sec=min(target.end_sec, e + pad),
            score=len(rel_frames) / max(len(new_frames), 1),
            evidence_ids=list(target.evidence_ids),
        )
        if narrowed.duration_sec >= cfg.retrieval.min_window_sec:
            after = narrowed
            reason = f"细化收窄: [{target.start_sec:.2f},{target.end_sec:.2f}) -> [{after.start_sec:.2f},{after.end_sec:.2f})"
    observations.append({
        "action": "refine", "iteration": iteration,
        "before_window": [target.start_sec, target.end_sec],
        "after_window": [after.start_sec, after.end_sec],
        "sampled_frames": [f.frame_id for f in new_frames],
        "reason": reason,
    })
    return {
        "candidate_windows": [window_to_dict(after)] + windows_to_dicts(candidates[1:]),
        "used_frame_ids": list(dict.fromkeys(used + [f.frame_id for f in new_frames])),
        "vlm_calls": vlm_calls,
        "iteration": iteration + 1,
        "observations": observations,
        "last_observation": observations[-1],
    }


# ---------------------------------------------------------------------------
# finalize / validate
# ---------------------------------------------------------------------------

def finalize(state: dict, config: RunnableConfig) -> dict:
    """汇总：已确认窗口优先，否则保留最优候选并标注未验证。

    注意：LangGraph 只应用节点返回的更新 dict，原地修改 state 无效。
    """
    warnings = list(state.get("warnings", []))
    confirmed = [
        r for r in state.get("verifier_results", [])
        if isinstance(r, dict) and r.get("verdict") == "confirmed"
        and (r.get("confidence") or 0) >= 0.5
    ]
    if confirmed:
        selected = [dict_to_window(r) for r in confirmed]
    else:
        candidates = dicts_to_windows(state.get("candidate_windows", []))
        selected = candidates[:1]
        if selected:
            warnings.append("无 VLM 确认窗口，保留分数最高候选（未经验证）")
    status = "ok" if selected else "no_result"
    elapsed = time.time() - float(state.get("started_at", time.time()))
    return {
        "selected_windows": windows_to_dicts(selected),
        "status": status,
        "elapsed_sec": round(elapsed, 3),
        "warnings": warnings,
        "artifacts": {
            "n_candidates": len(state.get("candidate_windows", [])),
            "n_evidence": len(state.get("evidence", [])),
            "n_observations": len(state.get("observations", [])),
            "n_verifier_results": len(state.get("verifier_results", [])),
        },
    }


def fail_node(state: dict, config: RunnableConfig) -> dict:
    return {
        "status": "failed",
        "errors": list(state.get("errors", [])) + list(state.get("fatal_errors", [])),
        "elapsed_sec": round(time.time() - float(state.get("started_at", time.time())), 3),
    }


def validate_artifact(state: dict, config: RunnableConfig) -> dict:
    """程序规则独立校验（不依赖 VLM 自评，说明书 §7.4）。

    - 时间契约：0 <= start < end <= duration
    - 状态一致性：status ok 必有窗口；预算未超限
    """
    cfg = Config.model_validate(config["configurable"]["cfg"])
    index = load_index(state["video_id"], cfg)
    duration = index.asset.duration_sec
    errors: list[str] = list(state.get("errors", []))
    warnings: list[str] = list(state.get("warnings", []))

    for wd in state.get("selected_windows", []):
        try:
            w = dict_to_window(wd)
            if not is_valid_window(w, duration):
                errors.append(f"非法时间窗口: {wd}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"窗口反序列化失败: {exc}")

    if state.get("status") == "ok" and not state.get("selected_windows"):
        errors.append("status=ok 但无选中窗口（状态不一致）")
    b = state.get("budget", {})
    if len(state.get("used_frame_ids", [])) > b.get("max_frames", 0):
        errors.append("观察帧数超出预算")
    if int(state.get("vlm_calls", 0)) > b.get("max_vlm_calls", 0):
        errors.append("VLM 调用数超出预算")

    if errors:
        return {"errors": errors, "status": "failed", "warnings": warnings}
    return {"warnings": warnings, "artifacts": {**state.get("artifacts", {}), "validated": True}}
