"""Embedding 工厂：openclip | mock | none。"""

from __future__ import annotations

from ..config import Config
from .base import EmbeddingAdapter, MockEmbedding
from .openclip_encoder import EmbeddingUnavailableError, OpenCLIPEmbedding


def build_embedder(
    config: Config,
    use_mock: bool = False,
    hints: list[dict] | None = None,
    frames=None,
) -> EmbeddingAdapter | None:
    """按配置构建嵌入后端；不可用/禁用时返回 None（调用方决定降级或报错）。"""
    backend = config.embedding.backend
    if use_mock or backend == "mock":
        return MockEmbedding(hints=hints, frames=frames)
    if backend in ("none", "disabled"):
        return None
    if backend == "openclip":
        if not OpenCLIPEmbedding.is_available():
            return None
        try:
            return OpenCLIPEmbedding(
                model_name=config.embedding.model_name,
                pretrained=config.embedding.pretrained,
                device=config.embedding.device,
                max_side=config.video.max_frame_side_px,
            )
        except EmbeddingUnavailableError:
            return None
    return None
