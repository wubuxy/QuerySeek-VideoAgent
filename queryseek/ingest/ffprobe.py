"""基于 ffprobe CLI 的视频元数据探测（复用 FFmpeg，不自己实现解析器）。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from ..schemas import VideoAsset

_CHUNK = 1024 * 1024


class FFmpegNotFound(RuntimeError):
    pass


def find_ffmpeg() -> tuple[str | None, str | None]:
    """返回 (ffmpeg_path, ffprobe_path)，未安装为 None。"""
    return shutil.which("ffmpeg"), shutil.which("ffprobe")


def require_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FFmpegNotFound(
            "未找到 ffprobe。请安装 FFmpeg 并加入 PATH（https://ffmpeg.org/download.html）"
        )
    return ffprobe


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def probe(path: str | Path) -> dict:
    """ffprobe -show_format -show_streams 的 JSON 输出。"""
    ffprobe = require_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {proc.stderr.decode(errors='replace')[:500]}")
    return json.loads(proc.stdout.decode("utf-8"))


def _parse_fps(r_frame_rate: str) -> float:
    try:
        f = Fraction(r_frame_rate)
        return float(f)
    except (ValueError, ZeroDivisionError):
        return 0.0


def asset_from_probe(video_id: str, path: str | Path) -> VideoAsset:
    """探测元数据并构造 VideoAsset。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {path}")
    info = probe(path)
    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or 0.0)

    vstream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if vstream is None:
        raise RuntimeError(f"未找到视频流: {path}")
    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    fps = _parse_fps(vstream.get("r_frame_rate") or vstream.get("avg_frame_rate") or "0")
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))

    # 旋转元数据（displaymatrix side data）
    rotation = 0.0
    for side in vstream.get("side_data_list", []):
        if "rotation" in side:
            try:
                rotation = float(side["rotation"])
            except (TypeError, ValueError):
                rotation = 0.0

    return VideoAsset(
        video_id=video_id,
        path=str(path.resolve()),
        sha256=sha256_file(path),
        duration_sec=duration,
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
        rotation=rotation,
    )
