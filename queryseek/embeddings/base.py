"""Embedding Adapter 契约与确定性 Mock（与 VLM Adapter 同设计哲学）。

- 真实后端：OpenCLIP（openclip_encoder.py）；确定性 Mock：字符 n-gram 特征哈希。
- Mock 的图像"语义"来自 content hints（帧时间戳 -> 标签），文本侧与图像侧同构，
  保证离线单测可真实检验"语义检索 -> 时间聚簇"的算法逻辑；Mock 不读像素。
- 嵌入后端在 ingest 与 search 之间必须一致（embedding_meta.json 校验）。
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from ..hints import hint_at
from ..tokenize import tokenize_query
from ..schemas import VideoFrame

__all__ = ["EmbeddingAdapter", "MockEmbedding"]


class EmbeddingAdapter(Protocol):
    """文本/图像编码契约：输出 float32 单位向量矩阵。"""

    model_id: str
    dim: int

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...
    def encode_images(self, paths: list[str]) -> np.ndarray: ...


def _feature_hash(tokens: list[str], dim: int) -> np.ndarray:
    """确定性特征哈希（hashlib 而非内置 hash——后者带进程盐，跨进程不可复现）。"""
    v = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class MockEmbedding:
    """确定性 Mock：文本=查询 token 哈希；图像=帧所属 content hint 标签哈希。

    无 hints 时图像侧为全零向量（与任何查询相似度 0）——Mock 不假装看见内容。
    """

    model_id = "mock-embedding-1"
    dim = 256

    def __init__(self, hints: list[dict] | None = None, frames: list[VideoFrame] | None = None):
        self.hints = hints or []
        self._time_of = {f.image_path: f.timestamp_sec for f in (frames or [])}

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([_feature_hash(tokenize_query(t), self.dim) for t in texts])

    def encode_images(self, paths: list[str]) -> np.ndarray:
        rows = []
        for p in paths:
            label = hint_at(self.hints, self._time_of.get(p, -1.0))
            rows.append(_feature_hash(tokenize_query(label), self.dim))
        return np.stack(rows)
