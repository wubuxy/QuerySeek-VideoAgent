"""视频注册表：video_id -> 视频资产映射（data_root/registry.json）。

CLI 以 --video-id 引用已 ingest 的视频；video_id 白名单校验
（[A-Za-z0-9_-]），从根上拒绝路径穿越。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..schemas import VideoAsset
from .cache import read_json, write_json_atomic

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def sanitize_video_id(video_id: str) -> str:
    """video_id 白名单：字母/数字/_/./-；显式拒绝路径穿越（斜杠与 '..'）。"""
    if not _SAFE_ID.fullmatch(video_id) or ".." in video_id or "/" in video_id or "\\" in video_id:
        raise ValueError(
            f"video_id 含非法字符（仅允许字母/数字/_/./-，且不得含路径穿越）: {video_id!r}"
        )
    return video_id


def registry_path(data_root: Path) -> Path:
    return data_root / "registry.json"


def load_registry(data_root: Path) -> dict:
    raw = read_json(registry_path(data_root)) or {"videos": {}}
    return raw


def register_video(data_root: Path, asset: VideoAsset) -> None:
    reg = load_registry(data_root)
    reg["videos"][asset.video_id] = asset.model_dump()
    write_json_atomic(registry_path(data_root), reg)


def resolve_video(data_root: Path, video_id: str) -> VideoAsset:
    """按 video_id 解析已注册视频；未注册给出明确错误。"""
    video_id = sanitize_video_id(video_id)
    reg = load_registry(data_root)
    entry = reg.get("videos", {}).get(video_id)
    if not entry:
        raise KeyError(
            f"视频 {video_id!r} 未注册：请先运行 "
            f"`python -m queryseek.cli ingest --video <路径> --video-id {video_id}`"
        )
    return VideoAsset.model_validate(entry)
