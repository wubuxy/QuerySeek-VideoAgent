"""时间窗口验证与规范化（开发说明书 §12.3、§7.5）。

统一处理候选窗口的越界、逆序、空区间、重叠合并与最短/最长长度约束；
导出前必须经过 normalize_windows。
"""

from __future__ import annotations

from ..schemas import TimeWindow

__all__ = [
    "iou",
    "is_valid_window",
    "merge_overlapping",
    "normalize_windows",
]


def iou(a: TimeWindow, b: TimeWindow) -> float:
    inter = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    union = (a.end_sec - a.start_sec) + (b.end_sec - b.start_sec) - inter
    if union <= 0:
        return 0.0
    return inter / union


def is_valid_window(w: TimeWindow, duration_sec: float) -> bool:
    return 0.0 <= w.start_sec < w.end_sec <= duration_sec


def merge_overlapping(windows: list[TimeWindow]) -> list[TimeWindow]:
    """合并重叠或相邻（间隙为 0）的窗口；分数取最大值，证据并集。"""
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w.start_sec)
    merged: list[TimeWindow] = [ordered[0]]
    for w in ordered[1:]:
        last = merged[-1]
        if w.start_sec <= last.end_sec:  # 重叠或接触
            merged[-1] = TimeWindow(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, w.end_sec),
                score=max(last.score or 0.0, w.score or 0.0),
                evidence_ids=sorted(set(last.evidence_ids) | set(w.evidence_ids)),
            )
        else:
            merged.append(w)
    return merged


def normalize_windows(
    windows: list[TimeWindow],
    duration_sec: float,
    min_sec: float = 3.0,
    max_sec: float = 120.0,
) -> list[TimeWindow]:
    """裁剪到 [0, duration]、丢弃无效、合并重叠、满足最短/最长长度约束。"""
    out: list[TimeWindow] = []
    for w in windows:
        s, e = max(0.0, w.start_sec), min(duration_sec, w.end_sec)
        if s >= e:
            continue
        out.append(TimeWindow(start_sec=s, end_sec=e, score=w.score, evidence_ids=list(w.evidence_ids)))
    out = merge_overlapping(out)
    result = []
    for w in out:
        s, e = w.start_sec, w.end_sec
        # 最短长度：向两侧对称外扩
        if e - s < min_sec:
            need = min_sec - (e - s)
            grow_l = min(need / 2.0, s)
            s -= grow_l
            e += need - grow_l
            if e > duration_sec:
                e = duration_sec
                s = max(0.0, e - min_sec)
        # 最长长度：截断
        if e - s > max_sec:
            e = s + max_sec
        result.append(TimeWindow(start_sec=s, end_sec=e, score=w.score, evidence_ids=list(w.evidence_ids)))
    return result
