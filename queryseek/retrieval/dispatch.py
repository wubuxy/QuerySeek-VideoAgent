"""CLI 和网页共用的检索入口；curve 为 E4，不创建 VLM 客户端。"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..embeddings.cache import check_backend, load_embeddings
from ..embeddings.factory import build_embedder
from ..ingest.pipeline import VideoIndex
from ..logging_utils import TraceWriter
from ..models.factory import build_vlm
from ..schemas import SearchResult
from .adaptive import adaptive_search
from .curve_spans import curve_search
from .semantic import semantic_search
from .subtitle import subtitle_search
from .uniform import uniform_search

AGENT_STRATEGIES = ("subtitle", "uniform", "semantic", "adaptive")
STRATEGIES = (*AGENT_STRATEGIES, "curve")


def embedder_for(index: VideoIndex, cfg: Config, use_mock: bool):
    if use_mock:
        return build_embedder(cfg, use_mock=True, hints=index.content_hints, frames=index.frames), None
    if cfg.embedding.backend in ("none", "disabled"):
        raise RuntimeError("视觉检索需要 --mock 或配置 embedding.backend=openclip")
    embedder = build_embedder(cfg, use_mock=False)
    if embedder is None:
        raise RuntimeError("OpenCLIP 不可用：安装 .[embedding] 并重新 ingest，或使用 --mock")
    loaded = load_embeddings(index.cache_dir)
    if loaded is None:
        return embedder, None
    mat, meta = loaded
    check_backend(meta, "openclip")
    if meta.get("model_id") != embedder.model_id or meta.get("frame_ids") != [f.frame_id for f in index.frames]:
        raise RuntimeError("嵌入模型或帧顺序与索引不一致，请按当前配置重新 ingest")
    return embedder, np.asarray(mat)


def search_index(
    cfg: Config,
    index: VideoIndex,
    video_id: str,
    query: str,
    strategy: str,
    budget_frames: int | None = None,
    mock: bool = False,
    trace: TraceWriter | None = None,
) -> SearchResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy!r}（可用: {' | '.join(STRATEGIES)}）")
    if strategy == "subtitle":
        return subtitle_search(video_id, query, index.subtitles, index.asset.duration_sec, cfg)
    if strategy == "uniform":
        vlm = build_vlm(cfg, use_mock=mock, hints=index.content_hints)
        return uniform_search(video_id, query, index.frames, index.asset.duration_sec, vlm, cfg,
                              budget_frames=budget_frames, trace=trace, scenes=index.scenes)
    embedder, mat = embedder_for(index, cfg, mock)
    if strategy == "curve":
        result = curve_search(video_id, query, index, embedder, cfg, frame_embeddings=mat, trace=trace)
        if budget_frames is not None:
            result.warnings.append("curve 使用全视频 CLIP 索引；--budget-frames 不限制索引帧数，VLM 调用为 0")
        return result
    if strategy == "semantic":
        return semantic_search(video_id, query, index.frames, index.asset.duration_sec,
                               embedder, cfg, frame_embeddings=mat)
    vlm = build_vlm(cfg, use_mock=mock, hints=index.content_hints)
    return adaptive_search(video_id, query, index, vlm, embedder, cfg,
                           budget_frames=budget_frames, trace=trace, frame_embeddings=mat)
