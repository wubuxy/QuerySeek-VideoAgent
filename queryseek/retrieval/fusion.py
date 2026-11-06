"""融合排序（开发说明书 §7.2）：RRF + IoU-NMS。

RRF 对"窗口"而非文档排序：每个方法的窗口列表视为其排序结果（方法内已按分数
降序），融合分数 = Σ 1/(k + rank_i)。不同方法给出的近重复窗口（IoU >= 阈值）
按 NMS 贪心去重。**不将未校准的异源相似度直接相加。**
"""

from __future__ import annotations

from ..schemas import TimeWindow
from ..tools.validate import iou as _iou

__all__ = ["rrf_fuse", "nms_dedupe"]


def _key(w: TimeWindow) -> tuple[float, float]:
    return (round(w.start_sec, 3), round(w.end_sec, 3))


def rrf_fuse(method_windows: list[tuple[str, list[TimeWindow]]], rrf_k: int = 60) -> list[TimeWindow]:
    """融合多方法窗口：score = Σ_m 1/(k + rank_m(w))，rank 从 1 起。"""
    scores: dict[tuple[float, float], float] = {}
    carriers: dict[tuple[float, float], TimeWindow] = {}
    for _method, ws in method_windows:
        for rank, w in enumerate(ws, start=1):
            key = _key(w)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in carriers:
                carriers[key] = w
    merged = []
    for key, score in scores.items():
        w = carriers[key]
        merged.append(w.model_copy(update={"score": round(score, 6)}))
    merged.sort(key=lambda w: (w.score or 0.0), reverse=True)
    return merged


def nms_dedupe(windows: list[TimeWindow], nms_iou: float = 0.5) -> list[TimeWindow]:
    """按分数降序贪心去重：与已保留窗口 IoU >= nms_iou 的候选被丢弃。"""
    ordered = sorted(windows, key=lambda w: (w.score or 0.0), reverse=True)
    kept: list[TimeWindow] = []
    for w in ordered:
        if any(_iou(w, k) >= nms_iou for k in kept):
            continue
        kept.append(w)
    return kept
