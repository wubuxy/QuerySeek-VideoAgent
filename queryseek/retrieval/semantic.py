"""语义检索（E1 基础组件）：查询与帧的跨模态相似度 -> 时间邻接聚簇。

- OpenCLIP 路径：使用 ingest 缓存的 embeddings.npy（顺序与 frames.jsonl 一致），
  仅需编码查询文本。
- Mock 路径：帧嵌入现场计算（content hint 标签的特征哈希），不读像素。
- 相似度聚簇后经 normalize_windows 统一为合法时间窗口。
"""

from __future__ import annotations

import time

import numpy as np

from ..config import Config
from ..embeddings.base import EmbeddingAdapter
from ..schemas import Evidence, Scene, SearchResult, TimeWindow, VideoFrame
from ..tools.validate import normalize_windows
from .window_growth import expand_to_shots

__all__ = ["semantic_search"]


def semantic_search(
    video_id: str,
    query: str,
    frames: list[VideoFrame],
    duration_sec: float,
    embedder: EmbeddingAdapter,
    config: Config,
    frame_embeddings: np.ndarray | None = None,
    top_k: int | None = None,
    method: str = "semantic",
    scenes: list[Scene] | None = None,
) -> SearchResult:
    t0 = time.perf_counter()
    rcfg = config.retrieval
    scfg = rcfg.semantic
    top_k = top_k or rcfg.top_k_windows
    warnings: list[str] = []

    if not frames:
        warnings.append("帧索引为空，无法执行语义检索")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0, warnings=warnings,
        )

    if frame_embeddings is None:
        frame_embeddings = embedder.encode_images([f.image_path for f in frames])
    if frame_embeddings.shape[0] != len(frames):
        raise RuntimeError(
            f"帧嵌入数量 {frame_embeddings.shape[0]} 与帧索引 {len(frames)} 不一致，请重新 ingest"
        )
    qvec = embedder.encode_texts([query])[0]
    sims = np.asarray(frame_embeddings @ qvec, dtype=np.float64)

    order = np.argsort(-sims)
    selected: list[VideoFrame] = []
    sim_of: dict[str, float] = {}
    for i in order[: scfg.top_k_frames]:
        if sims[i] < scfg.min_sim:
            continue
        selected.append(frames[i])
        sim_of[frames[i].frame_id] = float(sims[i])
    if not selected:
        warnings.append(f"未找到语义相关的帧（最大相似度 {float(sims.max()):.3f} < {scfg.min_sim}，证据不足）")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0, warnings=warnings,
        )

    # 时间邻接聚簇（间隔 > cluster_gap_sec 开新簇）
    selected.sort(key=lambda f: f.timestamp_sec)
    clusters: list[list[VideoFrame]] = []
    for f in selected:
        if clusters and f.timestamp_sec - clusters[-1][-1].timestamp_sec <= scfg.cluster_gap_sec:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    pad = scfg.window_pad_sec
    raw_windows = []
    for cluster in clusters:
        s = cluster[0].timestamp_sec - pad
        e = cluster[-1].timestamp_sec + pad
        mean_sim = float(np.mean([sim_of[f.frame_id] for f in cluster]))
        w = TimeWindow(
            start_sec=max(0.0, s),
            end_sec=min(duration_sec, e),
            score=mean_sim,
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
    windows = sorted(windows, key=lambda w: (w.score or 0.0), reverse=True)[:top_k]

    evidence = [
        Evidence(
            evidence_id=f"frame_{f.frame_id}",
            type="frame",
            timestamp_sec=f.timestamp_sec,
            content=f"sim={sim_of[f.frame_id]:.3f}",
            source_file=f.image_path,
            model_id=embedder.model_id,
            score=sim_of[f.frame_id],
        )
        for f in selected
    ]

    return SearchResult(
        video_id=video_id,
        query=query,
        windows=windows,
        method=method,
        used_frame_ids=[f.frame_id for f in selected],
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        evidence=evidence,
        warnings=warnings,
    )
