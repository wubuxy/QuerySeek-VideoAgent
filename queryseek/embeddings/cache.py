"""帧嵌入缓存：embeddings.npy + embedding_meta.json（与帧索引顺序一致）。

后端/模型任何变化都会改变 embedding_meta，检索时校验后端一致性，
防止"ingest 用 A 后端、search 用 B 后端"的静默错配。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..ingest.cache import read_json, write_json_atomic


def save_embeddings(cache_dir: Path, mat: np.ndarray, frame_ids: list[str], meta: dict) -> None:
    np.save(cache_dir / "embeddings.npy", mat.astype(np.float32))
    write_json_atomic(
        cache_dir / "embedding_meta.json",
        {
            **meta,
            "dim": int(mat.shape[1]),
            "n_frames": int(mat.shape[0]),
            "frame_ids": frame_ids,
            "normalized": True,
        },
    )


def load_embeddings(cache_dir: Path) -> tuple[np.ndarray, dict] | None:
    """返回 (mat, meta)；不存在返回 None。"""
    meta = read_json(cache_dir / "embedding_meta.json")
    path = cache_dir / "embeddings.npy"
    if not meta or not path.is_file():
        return None
    mat = np.load(path)
    if mat.shape[0] != len(meta["frame_ids"]) or mat.shape[1] != meta["dim"]:
        raise RuntimeError("embeddings.npy 与 embedding_meta.json 不一致，请重新 ingest")
    return mat, meta


def check_backend(meta: dict | None, expected: str) -> None:
    if meta and meta.get("backend") != expected:
        raise RuntimeError(
            f"索引嵌入后端为 {meta.get('backend')!r}，与当前检索后端 {expected!r} 不一致；"
            "请用相同后端重新 ingest"
        )
