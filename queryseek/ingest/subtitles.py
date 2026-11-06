"""字幕加载：优先用户 SRT（`srt` 包解析）；ASR 可选。

ASR 后端顺序（asr.backend=auto）：用户 SRT -> faster-whisper（若已安装）-> disabled。
P1 不安装 faster-whisper（重依赖），仅保留 lazy-import 路径；
无音轨/无字幕时系统必须继续支持视觉搜索，不崩溃。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import srt as srtlib

from ..schemas import SubtitleSegment

log = logging.getLogger(__name__)


def load_srt(path: str | Path, duration_sec: float | None = None) -> list[SubtitleSegment]:
    """解析 SRT -> SubtitleSegment 列表；排序、去重空文本、裁剪到视频时长。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SRT 文件不存在: {path}")
    subs = list(srtlib.parse(path.read_text(encoding="utf-8-sig")))
    segments: list[SubtitleSegment] = []
    for i, sub in enumerate(subs):
        text = " ".join(sub.content.split()).strip()
        if not text:
            continue
        s, e = sub.start.total_seconds(), sub.end.total_seconds()
        if duration_sec is not None:
            e = min(e, duration_sec)
        if e <= s:
            continue
        segments.append(
            SubtitleSegment(
                seg_id=f"seg_{i:04d}", start_sec=round(s, 3), end_sec=round(e, 3), text=text
            )
        )
    segments.sort(key=lambda x: x.start_sec)
    return segments


def try_local_asr(video: str | Path, model: str, compute_type: str) -> list[SubtitleSegment] | None:
    """faster-whisper CPU 转写（lazy import）。未安装/失败返回 None，不阻塞主流程。"""
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError:
        log.info("faster-whisper 未安装，跳过本地 ASR")
        return None
    try:
        m = WhisperModel(model, device="cpu", compute_type=compute_type)
        segments, _info = m.transcribe(str(video), vad_filter=True)
        out = []
        for i, seg in enumerate(segments):
            if not seg.text.strip():
                continue
            out.append(
                SubtitleSegment(
                    seg_id=f"asr_{i:04d}",
                    start_sec=round(seg.start, 3),
                    end_sec=round(seg.end, 3),
                    text=seg.text.strip(),
                    source="asr",
                )
            )
        return out
    except Exception as exc:  # noqa: BLE001 — ASR 失败只降级，不中断 ingest
        log.warning("本地 ASR 失败，回退为无字幕: %s", exc)
        return None


def save_transcript(segments: list[SubtitleSegment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg.model_dump(), ensure_ascii=False) + "\n")


def load_transcript(path: Path) -> list[SubtitleSegment]:
    segments = []
    if not path.is_file():
        return segments
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            segments.append(SubtitleSegment.model_validate(json.loads(line)))
    return segments
