"""百炼上传准入校验（规则来源与日期见 docs/FINE_TUNING.md）。

校验项：data.jsonl 可解析 / image 为裸文件名且符合 ASCII 规则 / 图片存在、
≤1024px、≤10MB / 角色合法 / 分片总大小 ≤ 上限（按未压缩大小估计）。
上传前的最后一层防线——任何一项不过都会在上传后导致整包被拒。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

BARE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.(jpg|jpeg|png|bmp|tif|tiff|webp)$")
_ROLES = {"system", "user", "assistant"}


def validate_shard(
    shard_dir: str | Path,
    max_image_side: int = 1024,
    max_image_mb: float = 10.0,
    max_total_mb: float = 1900.0,
) -> dict:
    """校验单个 shard 目录；返回报告（ok / records / images / total_mb / errors）。"""
    shard_dir = Path(shard_dir)
    errors: list[str] = []
    n_records, n_images, total_bytes = 0, 0, 0

    data = shard_dir / "data.jsonl"
    if not data.is_file():
        return _report(shard_dir, 0, 0, 0.0, [f"缺少 data.jsonl: {data}"])
    total_bytes += data.stat().st_size

    for lineno, line in enumerate(data.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"data.jsonl 第 {lineno} 行不是合法 JSON: {exc}")
            continue
        n_records += 1
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or not msgs:
            errors.append(f"第 {lineno} 行缺少 messages 数组")
            continue
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in _ROLES:
                errors.append(f"第 {lineno} 行非法 role: {m.get('role') if isinstance(m, dict) else m!r}")
                continue
            content = m.get("content")
            if not isinstance(content, list):
                errors.append(f"第 {lineno} 行 role={m['role']} 的 content 不是数组")
                continue
            for item in content:
                if not isinstance(item, dict) or "image" not in item:
                    continue
                name = item["image"]
                if not isinstance(name, str) or not BARE_NAME_RE.match(name):
                    errors.append(f"第 {lineno} 行 image 不是裸文件名: {name!r}")
                    continue
                img = shard_dir / name
                if not img.is_file():
                    errors.append(f"第 {lineno} 行图片缺失: {name}")
                    continue
                n_images += 1
                size_mb = img.stat().st_size / 1e6
                total_bytes += img.stat().st_size
                if size_mb > max_image_mb:
                    errors.append(f"图片超 {max_image_mb}MB（{size_mb:.2f}MB）: {name}")
                arr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    errors.append(f"图片无法解码: {name}")
                elif max(arr.shape[:2]) > max_image_side:
                    errors.append(f"图片边长超 {max_image_side}px: {name}")

    total_mb = total_bytes / 1e6
    if total_mb > max_total_mb:
        errors.append(f"分片总大小超限: {total_mb:.1f}MB > {max_total_mb}MB")
    return _report(shard_dir, n_records, n_images, total_mb, errors)


def _report(shard_dir: Path, records: int, images: int,
            total_mb: float, errors: list[str]) -> dict:
    return {
        "shard": str(shard_dir),
        "records": records,
        "images": images,
        "total_mb": round(total_mb, 2),
        "errors": errors,
        "ok": not errors,
    }
