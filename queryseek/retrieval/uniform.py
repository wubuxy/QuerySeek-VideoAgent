"""Uniform 检索 baseline（E0）：预算 K 帧均匀抽样 + VLM 判定 + 时间聚簇。

帧均匀取自离线索引；VLM 按 max_frames_per_call 分块调用；相关帧按
cluster_gap_sec 聚簇成候选窗口并外扩 window_pad_sec。E0 输出的是时序定位
结果（时间窗口），不是自由文本 QA（开发说明书 §7.2）。
"""

from __future__ import annotations

import time

from ..config import Config
from ..logging_utils import TraceWriter
from ..models.interface import VLMAdapter, VLMError
from ..schemas import Evidence, Scene, SearchResult, TimeWindow, VideoFrame
from ..tools.validate import normalize_windows
from .window_growth import expand_to_shots

__all__ = ["uniform_search", "select_budget_frames"]


def select_budget_frames(frames: list[VideoFrame], budget: int) -> list[VideoFrame]:
    """从帧索引中均匀选出最多 budget 帧（按时间排序后的等间隔下标）。"""
    if budget <= 0 or not frames:
        return []
    n = len(frames)
    if budget >= n:
        return list(frames)
    if budget == 1:
        return [frames[n // 2]]
    idx = [round(i * (n - 1) / (budget - 1)) for i in range(budget)]
    return [frames[i] for i in sorted(set(idx))]


def uniform_search(
    video_id: str,
    query: str,
    frames: list[VideoFrame],
    duration_sec: float,
    vlm: VLMAdapter,
    config: Config,
    budget_frames: int | None = None,
    method: str = "uniform",
    trace: TraceWriter | None = None,
    scenes: list[Scene] | None = None,
) -> SearchResult:
    t0 = time.perf_counter()
    rcfg = config.retrieval
    budget = budget_frames or rcfg.frame_budget
    warnings: list[str] = []
    vlm_calls = 0

    if not frames:
        warnings.append("帧索引为空，无法执行视觉检索")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0, warnings=warnings,
        )

    selected = select_budget_frames(frames, budget)
    if len(selected) < budget:
        warnings.append(f"帧预算 {budget} 超出索引帧数，实际使用 {len(selected)} 帧")

    # VLM 分块判定相关性
    relevant: list[VideoFrame] = []
    judgments = []
    max_chunk = getattr(vlm, "max_frames_per_call", 8)
    try:
        for i in range(0, len(selected), max_chunk):
            chunk = selected[i : i + max_chunk]
            call_t0 = time.perf_counter()
            res = vlm.inspect_frames(query, chunk)
            vlm_calls += 1
            if trace:
                trace.event(
                    "vlm_call",
                    method=method,
                    model=vlm.model_id,
                    frames=[f.frame_id for f in chunk],
                    latency_ms=round((time.perf_counter() - call_t0) * 1000.0, 1),
                    usage=getattr(vlm, "last_usage", {}),
                    relevant=[j.frame_id for j in res.judgments if j.relevant],
                )
            judgments.extend(res.judgments)
            rel_ids = set(res.relevant_frame_ids)
            for f in chunk:
                if f.frame_id in rel_ids:
                    relevant.append(f)
    except VLMError as exc:
        warnings.append(f"VLM 调用失败: {exc}")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            used_frame_ids=[f.frame_id for f in selected],
            vlm_calls=vlm_calls,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            warnings=warnings,
        )

    if not relevant:
        warnings.append("VLM 未找到与查询相关的帧（证据不足 / 查询内容不存在）")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            used_frame_ids=[f.frame_id for f in selected],
            vlm_calls=vlm_calls,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            warnings=warnings,
        )

    # 相关帧按时间聚簇（间隔 > cluster_gap_sec 则开新簇）
    relevant.sort(key=lambda f: f.timestamp_sec)
    clusters: list[list[VideoFrame]] = []
    for f in relevant:
        if clusters and f.timestamp_sec - clusters[-1][-1].timestamp_sec <= rcfg.uniform.cluster_gap_sec:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    pad = rcfg.uniform.window_pad_sec
    raw_windows = []
    for cluster in clusters:
        s = cluster[0].timestamp_sec - pad
        e = cluster[-1].timestamp_sec + pad
        w = TimeWindow(
            start_sec=max(0.0, s),
            end_sec=min(duration_sec, e),
            score=float(len(cluster)),
            evidence_ids=[f.frame_id for f in cluster],
        )
        # 镜头对齐扩张（真实时刻通常包含在镜头内；稀疏观察下固定外扩过窄）
        raw_windows.append(
            expand_to_shots(w, scenes or [], rcfg.max_window_sec,
                            cluster_ts=[f.timestamp_sec for f in cluster])
        )
    windows = normalize_windows(
        raw_windows,
        duration_sec=duration_sec,
        min_sec=rcfg.min_window_sec,
        max_sec=rcfg.max_window_sec,
    )
    windows = sorted(windows, key=lambda w: (w.score or 0.0), reverse=True)[: rcfg.top_k_windows]

    evidence = [
        Evidence(
            evidence_id=f"frame_{f.frame_id}",
            type="frame",
            timestamp_sec=f.timestamp_sec,
            content=next((j.reason for j in judgments if j.frame_id == f.frame_id), ""),
            source_file=f.image_path,
            model_id=vlm.model_id,
            score=1.0,
        )
        for f in relevant
    ]

    return SearchResult(
        video_id=video_id,
        query=query,
        windows=windows,
        method=method,
        used_frame_ids=[f.frame_id for f in selected],
        vlm_calls=vlm_calls,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        evidence=evidence,
        warnings=warnings,
    )
