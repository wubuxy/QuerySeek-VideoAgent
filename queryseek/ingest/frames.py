"""关键帧抽取：FFmpeg 均匀抽帧 + 每个镜头取代表帧（scene_or_uniform）。

默认缩图到 max_frame_side_px（512），绝不逐帧分析长视频（开发说明书 §4）。
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from ..schemas import Scene, VideoFrame
from .ffprobe import FFmpegNotFound

_FRAME_JPG_QUALITY = 3
_SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*([-+\d.eE]+)")


def source_timestamps(stderr: bytes) -> list[tuple[int, float]]:
    """FFmpeg showinfo reports the PTS of selected source frames, in seconds."""
    result=[]
    for line in stderr.decode(errors="replace").splitlines():
        if "showinfo" not in line:
            continue
        match=_SHOWINFO.search(line)
        if match:
            number=int(match.group(1));timestamp=float(match.group(2))
            if not math.isfinite(timestamp):raise RuntimeError("Non-finite source PTS")
            result.append((number,timestamp))
    return result


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FFmpegNotFound(
            "未找到 ffmpeg。请安装 FFmpeg 并加入 PATH（https://ffmpeg.org/download.html）"
        )
    return ffmpeg


def _scale_filter(max_side: int) -> str:
    # 长边不超过 max_side，保持宽高比；-2 保证偶数尺寸（yuv420p 需要）。
    return f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'"


def extract_uniform_frames(
    video: str | Path,
    out_dir: Path,
    fps: float,
    duration_sec: float,
    max_side: int = 512,
) -> list[VideoFrame]:
    """Select first source frame at/after each grid target, recording actual PTS.

    Unlike the fps filter this does not assign synthetic grid timestamps to a
    different image. Irregular source timing is retained (including VFR).
    """
    if not math.isfinite(fps) or fps<=0 or not math.isfinite(duration_sec) or duration_sec<=0:
        raise ValueError("Invalid sampling rate/duration")
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "info",
        "-i", str(video),
        "-vf", f"select='gte(t,selected_n/{fps})',{_scale_filter(max_side)},showinfo",
        "-fps_mode", "vfr",
        "-q:v", str(_FRAME_JPG_QUALITY),
        str(out_dir / "u_%06d.jpg"),
    ]
    proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败: {proc.stderr.decode(errors='replace')[:500]}")

    frames = []
    stamps=source_timestamps(proc.stderr)
    if not stamps:raise RuntimeError("FFmpeg returned no source timestamps")
    for i,t in stamps:
        img = out_dir / f"u_{i + 1:06d}.jpg"
        if not img.is_file() or not 0<=t<duration_sec:
            continue
        frames.append(
            VideoFrame(
                frame_id=f"frame_{i + 1:06d}",
                timestamp_sec=t,
                image_path=str(img),
                source="uniform",
            )
        )
    return frames


def extract_scene_frames(
    video: str | Path,
    out_dir: Path,
    scenes: list[Scene],
    max_side: int = 512,
) -> list[VideoFrame]:
    """每个镜头取中间一帧作为代表帧（-ss 精确 seek）。"""
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx, scene in enumerate(scenes):
        t = round((scene.start_sec + scene.end_sec) / 2.0, 6)
        img = out_dir / f"s_{idx + 1:06d}.jpg"
        cmd = [
            ffmpeg, "-y", "-loglevel", "info",
            "-ss", f"{t:.6f}",
            "-i", str(video),
            "-frames:v", "1",
            "-vf", f"{_scale_filter(max_side)},showinfo",
            "-q:v", str(_FRAME_JPG_QUALITY),
            str(img),
        ]
        proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=120)
        if proc.returncode != 0 or not img.is_file():
            continue  # 单帧抽取失败（如尾部不足）不致命，跳过
        stamps=source_timestamps(proc.stderr)
        if not stamps:raise RuntimeError("FFmpeg scene extraction returned no source PTS")
        frames.append(
            VideoFrame(
                frame_id="",  # 最终统一编号
                timestamp_sec=t+stamps[0][1],
                image_path=str(img),
                shot_id=scene.scene_id,
                source="scene",
            )
        )
    return frames


def select_frames(
    video: str | Path,
    out_dir: Path,
    scenes: list[Scene],
    duration_sec: float,
    coarse_fps: float,
    mode: str = "scene_or_uniform",
    max_side: int = 512,
) -> list[VideoFrame]:
    """按配置抽取帧；scene_or_uniform = 均匀网格 + 无覆盖镜头的代表帧。"""
    uniform = extract_uniform_frames(video, out_dir, coarse_fps, duration_sec, max_side)
    selected = list(uniform)
    if mode == "scene_or_uniform":
        covered = {f.shot_id for f in selected if f.shot_id}
        # 先给 uniform 帧补 shot_id（就近镜头包含判定）
        for f in selected:
            f.shot_id = _scene_of(f.timestamp_sec, scenes)
        covered = {f.shot_id for f in selected}
        uncovered=[scene for scene in scenes if scene.scene_id not in covered]
        for sf in extract_scene_frames(video, out_dir, uncovered, max_side):
            if sf.shot_id not in covered:
                selected.append(sf)
    selected.sort(key=lambda f: f.timestamp_sec)
    # 统一 frame_id（按时间排序后稳定编号）
    for i, f in enumerate(selected):
        f.frame_id = f"frame_{i + 1:06d}"
    return selected


def _scene_of(t: float, scenes: list[Scene]) -> str | None:
    for s in scenes:
        if s.start_sec <= t < s.end_sec:
            return s.scene_id
    return scenes[-1].scene_id if scenes else None


def frames_within(frames: list[VideoFrame], start_sec: float, end_sec: float) -> list[VideoFrame]:
    """索引帧中落在 [start_sec, end_sec) 内的帧（按时间排序）。"""
    return [f for f in frames if start_sec <= f.timestamp_sec < end_sec]


def extract_local_frames(
    video: str | Path,
    out_dir: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
    max_side: int = 512,
    tag: str = "local",
) -> list[VideoFrame]:
    """局部窗口加密抽帧：记录源帧真实PTS，再加seek起始秒数。

    与离线统一抽帧分离，不写入 frames.jsonl；缓存目录下独立存放。
    """
    if (not all(math.isfinite(x) for x in (start_sec,end_sec,fps))
            or not 0<=start_sec<end_sec or fps<=0
            or not re.fullmatch(r"[A-Za-z0-9_-]+",tag)):
        raise ValueError("Invalid local sampling parameters")
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "info",
        "-ss", f"{start_sec:.6f}",
        "-i", str(video),
        "-t", f"{end_sec-start_sec:.6f}",
        "-vf", f"select='gte(t,selected_n/{fps})',{_scale_filter(max_side)},showinfo",
        "-fps_mode", "vfr",
        "-q:v", str(_FRAME_JPG_QUALITY),
        str(out_dir / f"{tag}_%04d.jpg"),
    ]
    proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 局部抽帧失败: {proc.stderr.decode(errors='replace')[:300]}")

    frames = []
    stamps=source_timestamps(proc.stderr)
    if not stamps:raise RuntimeError("FFmpeg local extraction returned no source PTS")
    for i,offset in stamps:
        t=start_sec+offset
        img = out_dir / f"{tag}_{i + 1:04d}.jpg"
        if not img.is_file() or not start_sec<=t<end_sec:
            continue
        frames.append(
            VideoFrame(
                frame_id=f"{tag}_{i:04d}",
                timestamp_sec=t,
                image_path=str(img),
                source="refine",
            )
        )
    return frames


def save_frames_index(frames: list[VideoFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr.model_dump(), ensure_ascii=False) + "\n")


def load_frames_index(path: Path) -> list[VideoFrame]:
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frames.append(VideoFrame.model_validate(json.loads(line)))
    return frames
