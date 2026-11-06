"""离线预处理：ffprobe / 镜头 / 抽帧 / 字幕 / 缓存 / 编排。"""

from .pipeline import VideoIndex, ingest, load_index
from .registry import resolve_video, sanitize_video_id

__all__ = ["VideoIndex", "ingest", "load_index", "resolve_video", "sanitize_video_id"]
