"""缓存：video_sha256 + pipeline_version + 各步骤配置 -> 缓存键；原子写 JSON。

缓存键包含场景/抽帧/ASR/Embedding 配置（开发说明书 §7.1），配置变化不会命中旧缓存。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .. import PIPELINE_VERSION


def compute_cache_key(sha256: str, config_snapshot: dict[str, Any]) -> str:
    payload = {
        "video_sha256": sha256,
        "config": config_snapshot,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_dir_for(cache_root: Path, sha256: str) -> Path:
    # v1 artifacts stay untouched: source-PTS sampling must never overwrite them.
    return cache_root / f"pipeline_{PIPELINE_VERSION}" / sha256


def write_json_atomic(path: Path, data: Any) -> None:
    """原子写入：先写临时文件再替换，避免半写状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cache_hit(cache_root: Path, sha256: str, config_snapshot: dict[str, Any]) -> bool:
    """index_meta.json 存在且缓存键一致即命中。"""
    meta = read_json(cache_dir_for(cache_root, sha256) / "index_meta.json")
    if not meta:
        return False
    expected = compute_cache_key(sha256, config_snapshot)
    return meta.get("cache_key") == expected and meta.get("pipeline_version") == PIPELINE_VERSION
