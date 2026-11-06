"""生成合成测试样本（开发说明书 §8.1）：不依赖网络，只用 FFmpeg。

产出（默认 assets/demo/）：
- sample.mp4           60 秒、320x240、12fps、5 段纯色卡片 + 大写字幕 + 正弦音轨
- sample.srt           5 段关键词丰富的字幕
- sample.labels.json   每段内容提示（供 Mock VLM 做确定性关键词匹配）+ 标准查询/GT 窗口

用途：单测、CI、Mock Demo——保证 clone 后无密钥/无 GPU 也能验证全流程。
GT 窗口只用于测试与评测，不代表真实 VLM 准确率。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SEGMENTS = [
    # (color, text, start_sec, end_sec, mock label)
    ("red", "INTRO", 0, 12, "intro welcome 欢迎"),
    ("blue", "ARCHITECTURE", 12, 24, "architecture 系统架构 讲解 架构"),
    ("green", "RETRIEVAL", 24, 36, "retrieval 语义检索 检索 demo"),
    ("yellow", "AGENT", 36, 48, "agent 智能体 规划 planner"),
    ("purple", "EXPORT", 48, 60, "export 导出 剪辑 pipeline demonstrated"),
]

SRT = """1
00:00:00,000 --> 00:00:11,500
欢迎来到 QuerySeek 演示视频 intro welcome，本视频展示长视频理解与智能剪辑系统。

2
00:00:12,000 --> 00:00:23,500
现在讲解系统架构 architecture：ingest 预处理、retrieval 检索、agent 规划、export 剪辑。

3
00:00:24,000 --> 00:00:35,500
这里展示语义检索功能 retrieval，通过图文相似度定位相关片段。

4
00:00:36,000 --> 00:00:47,500
本节介绍智能体 agent 的规划 planner 与验证 verifier 机制。

5
00:00:48,000 --> 00:00:59,500
最后演示导出与剪辑 export pipeline demonstrated，生成最终 MP4 文件。
"""

LABELS = {
    "generator": "scripts/make_synthetic_fixture.py",
    "video": "sample.mp4",
    "note": "mock content hints + 标准查询/GT 窗口；仅供流程测试与评测，不代表真实 VLM 准确率",
    "segments": [
        {"start_sec": s, "end_sec": e, "label": label}
        for _c, _t, s, e, label in SEGMENTS
    ],
    "queries": [
        {"query": "寻找讲解系统架构的片段", "windows": [[12.0, 24.0]], "type": "mixed"},
        {"query": "寻找演示语义检索的片段", "windows": [[24.0, 36.0]], "type": "mixed"},
        {"query": "寻找介绍智能体规划的部分", "windows": [[36.0, 48.0]], "type": "mixed"},
        {"query": "Where is the export pipeline demonstrated?", "windows": [[48.0, 60.0]], "type": "mixed"},
        {"query": "寻找关于美食烹饪的片段", "windows": [], "type": "negative"},
    ],
}


def _find_font() -> str | None:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    return None


def _require_ffmpeg() -> str:
    import shutil

    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("未找到 ffmpeg：请先安装并加入 PATH（https://ffmpeg.org/download.html）")
    return exe


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=600)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg 失败: {proc.stderr.decode(errors='replace')[-800:]}")


def build_video(out_dir: Path, duration: float = 60.0) -> Path:
    """5 段纯色卡片 + 文本标签 concat + 正弦音轨。"""
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _find_font()
    seg_dur = duration / len(SEGMENTS)

    inputs: list[str] = []
    for color, _text, _s, _e, _label in SEGMENTS:
        inputs += ["-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=12:d={seg_dur}"]

    def drawtext(i: int) -> str:
        text = SEGMENTS[i][1]
        fontfile = font.replace(":", "\\:").replace("'", "\\'")
        return (
            f"[{i}:v]drawtext=fontfile='{fontfile}':text='{text}':"
            f"fontcolor=white:fontsize=28:x='(w-text_w)/2':y='(h-text_h)/2'[v{i}]"
        )

    fc = ";".join(
        drawtext(i) if font else f"[{i}:v]null[v{i}]" for i in range(len(SEGMENTS))
    )
    fc += ";" + "".join(f"[v{i}]" for i in range(len(SEGMENTS)))
    fc += f"concat=n={len(SEGMENTS)}:v=1:a=0[vout]"

    tmp_video = out_dir / ".sample_video_only.mp4"
    cmd = [ffmpeg, "-y"] + inputs + [
        "-filter_complex", fc, "-map", "[vout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-pix_fmt", "yuv420p", str(tmp_video),
    ]
    try:
        _run(cmd)
    except SystemExit as exc:
        if font:
            # drawtext 失败则回退为无文本版本（仍保留颜色分段与 SRT 语义）
            print(f"警告: drawtext 失败，回退无文本生成。原始错误: {exc}")
            fc = ";".join(f"[{i}:v]null[v{i}]" for i in range(len(SEGMENTS)))
            fc += ";" + "".join(f"[v{i}]" for i in range(len(SEGMENTS)))
            fc += f"concat=n={len(SEGMENTS)}:v=1:a=0[vout]"
            cmd = [ffmpeg, "-y"] + inputs + [
                "-filter_complex", fc, "-map", "[vout]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                "-pix_fmt", "yuv420p", str(tmp_video),
            ]
            _run(cmd)
        else:
            raise

    out = out_dir / "sample.mp4"
    cmd = [
        ffmpeg, "-y", "-i", str(tmp_video),
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
        "-shortest", str(out),
    ]
    _run(cmd)
    tmp_video.unlink(missing_ok=True)
    return out


def build_timing_video(out_dir: Path) -> Path:
    """8s lossless 12fps fixture; luminance 16+frame index identifies each source frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "timing_audit.mkv"
    _run([_require_ffmpeg(), "-y", "-f", "lavfi", "-i",
          "nullsrc=s=64x64:r=12:d=8,geq=lum='16+N':cb=128:cr=128",
          "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(target)])
    return target


def main() -> None:
    # Windows 控制台默认 GBK 会导致 UTF-8 输出乱码；显式强制 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    out_dir = Path("assets/demo")
    if "--out-dir" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    if "--timestamp-audit" in sys.argv:
        print(f"生成时间审计样本: {build_timing_video(out_dir)}")
        return

    video = build_video(out_dir)
    srt_path = out_dir / "sample.srt"
    srt_path.write_text(SRT, encoding="utf-8")
    labels_path = out_dir / "sample.labels.json"
    labels_path.write_text(json.dumps(LABELS, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"生成完成（全部可离线复现）:")
    print(f"  {video}")
    print(f"  {srt_path}")
    print(f"  {labels_path}")
    print(f"  5 段 × 12s，含正弦音轨；GT 查询 {len(LABELS['queries'])} 条（含 1 条负样本）")


if __name__ == "__main__":
    main()
