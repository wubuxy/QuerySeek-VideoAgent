"""按计划时间戳用 ffmpeg 抽单帧，带 per-video manifest 缓存。

复用 queryseek.ingest.frames 的 ffmpeg 调用规范（subprocess shell=False +
timeout，AGENTS.md 规则 3）；时间戳清单即本步骤的配置，清单变化则补抽
（AGENTS.md 规则 7 的缓存键思路）。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from queryseek.ingest.frames import _require_ffmpeg

MANIFEST_NAME = "manifest.json"


def _frame_path(out_dir: Path, t: float) -> Path:
    return out_dir / f"t{int(round(t, 2) * 1000):08d}.jpg"


def extract_frames_at(
    video: Path,
    timestamps: list[float],
    out_dir: Path,
    max_side: int = 512,
) -> tuple[list[Path], list[float]]:
    """在指定时间戳抽单帧；返回 (成功帧路径, 失败时间戳)。

    输出文件 t{毫秒:08d}.jpg；manifest.json 记录已抽时间戳，重跑时清单一致
    的部分直接复用（帧文件缺失时按 manifest 结果复核，不存在则视为缺失）。
    """
    ffmpeg = _require_ffmpeg()
    ts = sorted({round(max(0.01, t), 2) for t in timestamps})
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    done: dict[str, float] = {}
    if manifest_path.is_file():
        try:
            done = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            done = {}

    failed: list[float] = []
    for t in ts:
        if f"{t:.2f}" in done:
            continue
        dst = _frame_path(out_dir, t)
        cmd = [ffmpeg, "-y", "-ss", f"{t:.2f}", "-i", str(video),
               "-frames:v", "1", "-vf", f"scale='min({max_side},iw)':-2",
               "-q:v", "3", str(dst)]
        try:
            proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            failed.append(t)
            continue
        if proc.returncode == 0 and dst.is_file():
            done[f"{t:.2f}"] = t
        else:
            failed.append(t)

    manifest_path.write_text(json.dumps(done, ensure_ascii=False, indent=0),
                             encoding="utf-8")
    ok = [p for t in ts if f"{t:.2f}" in done and (p := _frame_path(out_dir, t)).is_file()]
    return ok, failed
