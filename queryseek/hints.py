"""Content hints 工具（合成样本的"内容提示"，供 Mock VLM / Mock Embedding 使用）。

hints 形如 [{"start_sec": f, "end_sec": f, "label": "关键词标签"}]，由
scripts/make_synthetic_fixture.py 生成、ingest 时复制进缓存目录。
Mock 组件只依据 hints 做确定性关键词匹配，不读像素、不假装视觉理解。
"""

from __future__ import annotations

from typing import Any

ContentHint = dict[str, Any]


def hint_at(hints: list[ContentHint], t: float) -> str:
    for h in hints:
        if h["start_sec"] <= t < h["end_sec"]:
            return str(h.get("label", "")).lower()
    return ""
