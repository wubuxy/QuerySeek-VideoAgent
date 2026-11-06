"""帧拼接图（montage）构建：把按时间顺序的多帧拼成单张图。

百炼视觉 SFT 单条记录只允许一张图（≤1024px，docs/FINE_TUNING.md），本模块
把 k 帧拼成网格并在每格左上角烧录序号标签，使模型能按序号指认具体帧
（训练记录文本里的 frame_id / 序号与画面格子一一对应）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# 百炼准入限制：单图宽高 ≤ 1024px（docs/FINE_TUNING.md）
MAX_SIDE = 1024
JPEG_QUALITY = 90
_LABEL_W, _LABEL_H = 34, 30


def _tile(image: np.ndarray, index: int, tile_side: int) -> np.ndarray:
    """缩放一帧到 tile_side 正方形画布（黑边填充保持宽高比）并烧录左上角序号。"""
    h, w = image.shape[:2]
    scale = tile_side / max(h, w)
    resized = cv2.resize(
        image, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((tile_side, tile_side, 3), dtype=np.uint8)
    y0 = (tile_side - resized.shape[0]) // 2
    x0 = (tile_side - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    # 白底黑字序号（1 起始，与训练记录文本的序号一致）
    cv2.rectangle(canvas, (2, 2), (2 + _LABEL_W, 2 + _LABEL_H), (255, 255, 255), -1)
    cv2.putText(canvas, str(index), (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def decode(path: Path) -> np.ndarray:
    """读取帧图片（np.fromfile 兼容含非 ASCII 的 Windows 路径）。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取帧: {path}")
    return img


def build_montage(frame_paths: list[Path], cols: int = 2,
                  quality: int = JPEG_QUALITY) -> bytes:
    """按给定顺序拼接帧为单张图，返回 JPEG 字节；总边长 ≤ MAX_SIDE。

    cols=2 时 4 帧拼 2×2（每格 512px → 1024×1024）；
    cols=4 时 8 帧拼 4×2（每格 256px → 1024×512）。
    """
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")
    rows = (len(frame_paths) + cols - 1) // cols
    tile_side = MAX_SIDE // max(cols, rows)
    grid: list[np.ndarray] = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            i = r * cols + c
            if i < len(frame_paths):
                row_tiles.append(_tile(decode(frame_paths[i]), i + 1, tile_side))
        if row_tiles:
            grid.append(np.hstack(row_tiles))
    montage = np.vstack(grid)
    ok, buf = cv2.imencode(".jpg", montage, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("montage JPEG 编码失败")
    return buf.tobytes()
