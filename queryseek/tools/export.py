"""FFmpeg 结构化剪辑与导出（开发说明书 §7.5）。

- 只接受结构化时间窗口，LLM 不生成 shell；subprocess 一律 shell=False + timeout。
- 输出限制在 outputs/<job_id>/ 内；job_id 白名单校验拒绝路径穿越。
- 片段统一重编码（libx264/aac）保证帧率/分辨率/音频流一致，再用 concat
  demuxer 拼接；-ss/-to 放在 -i 之后保证时间精度（见 docs/DECISIONS.md）。
- 导出后用 ffprobe 验证 MP4 可解码、时长在容忍范围内、含视频流。
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..logging_utils import TraceWriter
from ..schemas import (
    ExportManifest,
    ExportSegment,
    VideoAsset,
    TimeWindow,
)
from .validate import normalize_windows

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"job_{ts}_{uuid.uuid4().hex[:6]}"


def sanitize_job_id(job_id: str) -> str:
    if not _SAFE_ID.fullmatch(job_id) or ".." in job_id:
        raise ValueError(f"job_id 含非法字符（仅允许字母数字_.-，且不得含路径穿越）: {job_id!r}")
    return job_id


def _run(cmd: list[str], timeout: int, what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg {what} 失败: {proc.stderr.decode(errors='replace')[:500]}"
        )
    return proc


def _ffmpeg() -> str:
    import shutil

    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("未找到 ffmpeg，无法导出")
    return exe


def _ffprobe() -> str:
    import shutil

    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("未找到 ffprobe，无法验证导出")
    return exe


def _probe_duration(path: Path) -> tuple[float, bool]:
    cmd = [
        _ffprobe(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 验证失败: {proc.stderr.decode(errors='replace')[:300]}")
    info = json.loads(proc.stdout.decode("utf-8"))
    has_video = any(s.get("codec_type") == "video" for s in info.get("streams", []))
    duration = float(info.get("format", {}).get("duration") or 0.0)
    return duration, has_video


def _cut_segment(
    ffmpeg: str,
    src: Path,
    window: TimeWindow,
    dst: Path,
    cfg: Config,
    has_audio: bool,
) -> None:
    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-ss", f"{window.start_sec:.3f}",
        "-to", f"{window.end_sec:.3f}",
        "-map", "0:v:0",
        "-c:v", cfg.export.codec, "-preset", "veryfast", "-crf", str(cfg.export.crf),
        "-vf", f"fps={cfg.export.fps}",
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-map", "0:a:0?", "-c:a", cfg.export.audio_codec, "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd.append(str(dst))
    _run(cmd, timeout=600, what=f"裁剪 [{window.start_sec:.2f},{window.end_sec:.2f})")


def export_clip(
    asset: VideoAsset,
    windows: list[TimeWindow],
    output_dir: Path,
    config: Config,
    job_id: str | None = None,
    target_seconds: float | None = None,
    trace: TraceWriter | None = None,
) -> tuple[Path, ExportManifest]:
    """把窗口导出为统一编码的 MP4（多段拼接），并写 manifest.json。

    返回 (输出文件路径, manifest)。
    """
    job_id = sanitize_job_id(job_id or new_job_id())
    output_dir = output_dir / job_id
    seg_dir = output_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _ffmpeg()
    cfg = config.export
    duration = asset.duration_sec

    # 1. 规范化：裁剪、合并重叠、满足最短/最长约束
    norm = normalize_windows(
        windows,
        duration_sec=duration,
        min_sec=min(config.retrieval.min_window_sec, duration / 2.0),
        max_sec=config.retrieval.max_window_sec,
    )
    if not norm:
        raise RuntimeError("没有可导出的有效时间窗口")

    # 2. 按分数降序选择；有目标时长时贪心累加，不足则全量并警告
    ordered = sorted(norm, key=lambda w: (w.score or 0.0), reverse=True)
    warnings: list[str] = []
    if target_seconds is not None:
        chosen: list[TimeWindow] = []
        total = 0.0
        for w in ordered:
            chosen.append(w)
            total += w.duration_sec
            if total >= target_seconds:
                break
        if total < target_seconds:
            warnings.append(
                f"候选素材总时长 {total:.1f}s 不足目标 {target_seconds}s，导出全部候选"
            )
        ordered = chosen
    else:
        warnings.append("未指定 --target-seconds，导出全部候选窗口")

    # 3. 逐段重编码 + 拼接
    segments_out: list[ExportSegment] = []
    concat_list = output_dir / "concat.txt"
    lines = []
    for i, w in enumerate(ordered):
        seg_path = seg_dir / f"seg_{i + 1:03d}.mp4"
        t0 = None
        if trace:
            import time

            t0 = time.perf_counter()
            trace.event("export_segment_start", index=i, window=[w.start_sec, w.end_sec])
        _cut_segment(ffmpeg, Path(asset.path), w, seg_path, config, asset.has_audio)
        if trace:
            import time

            trace.event(
                "export_segment_done", index=i,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
        dur, has_video = _probe_duration(seg_path)
        if not has_video or dur <= 0:
            raise RuntimeError(f"片段 {seg_path} 验证失败（无视频流或时长为 0）")
        lines.append(f"file '{seg_path.resolve().as_posix()}'")
        exp_start = sum(s.export_end_sec - s.export_start_sec for s in segments_out)
        segments_out.append(
            ExportSegment(
                index=i,
                source_start_sec=round(w.start_sec, 3),
                source_end_sec=round(w.end_sec, 3),
                export_start_sec=round(exp_start, 3),
                export_end_sec=round(exp_start + dur, 3),
                score=w.score,
                evidence_ids=list(w.evidence_ids),
                segment_file=str(seg_path),
            )
        )
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_file = output_dir / "clip.mp4"
    _run(
        [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(out_file),
        ],
        timeout=600,
        what="拼接",
    )

    # 4. 验证输出
    out_dur, out_has_video = _probe_duration(out_file)
    expected = sum(s.export_end_sec - s.export_start_sec for s in segments_out)
    if not out_has_video:
        raise RuntimeError(f"导出文件无视频流: {out_file}")
    if abs(out_dur - expected) > cfg.target_duration_tolerance_sec:
        raise RuntimeError(
            f"导出时长 {out_dur:.2f}s 与期望 {expected:.2f}s 差异超过容忍值 {cfg.target_duration_tolerance_sec}s"
        )
    if target_seconds is not None and abs(out_dur - target_seconds) > cfg.target_duration_tolerance_sec:
        warnings.append(
            f"导出总时长 {out_dur:.2f}s 与目标 {target_seconds}s 差异超过容忍值（素材有限）"
        )

    manifest = ExportManifest(
        job_id=job_id,
        video_id=asset.video_id,
        source_path=asset.path,
        source_sha256=asset.sha256,
        query="",  # 由调用方填充（run 命令在写 manifest 前设置）
        strategy="",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        output_file=str(out_file),
        output_duration_sec=round(out_dur, 3),
        segments=segments_out,
        warnings=warnings,
    )
    return out_file, manifest
