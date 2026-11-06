"""图文 Embedding 层：OpenCLIP（真实）+ Mock（确定性特征哈希）+ 缓存。"""

from .base import EmbeddingAdapter, MockEmbedding
from .cache import check_backend, load_embeddings, save_embeddings
from .factory import build_embedder
from .openclip_encoder import EmbeddingUnavailableError, OpenCLIPEmbedding

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingUnavailableError",
    "MockEmbedding",
    "OpenCLIPEmbedding",
    "build_embedder",
    "check_backend",
    "load_embeddings",
    "save_embeddings",
]
