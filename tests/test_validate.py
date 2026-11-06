"""窗口规范化与合并测试（开发说明书 §12.3：越界/逆序/空区间/重叠）。"""

from __future__ import annotations

import pytest

from queryseek.schemas import TimeWindow
from queryseek.tools.validate import (
    iou,
    is_valid_window,
    merge_overlapping,
    normalize_windows,
)


def test_iou():
    a = TimeWindow(start_sec=0.0, end_sec=10.0)
    assert iou(a, TimeWindow(start_sec=5.0, end_sec=15.0)) == pytest.approx(1.0 / 3.0)
    assert iou(a, TimeWindow(start_sec=5.0, end_sec=10.0)) == 0.5
    assert iou(a, TimeWindow(start_sec=10.0, end_sec=20.0)) == 0.0
    assert iou(a, a) == 1.0


def test_merge_overlapping():
    ws = [
        TimeWindow(start_sec=0.0, end_sec=5.0, score=0.9, evidence_ids=["a"]),
        TimeWindow(start_sec=4.0, end_sec=8.0, score=0.3, evidence_ids=["b"]),
        TimeWindow(start_sec=10.0, end_sec=12.0, score=0.7, evidence_ids=["c"]),
    ]
    merged = merge_overlapping(ws)
    assert len(merged) == 2
    assert (merged[0].start_sec, merged[0].end_sec) == (0.0, 8.0)
    assert merged[0].score == 0.9  # 取最大分数
    assert set(merged[0].evidence_ids) == {"a", "b"}


def test_normalize_clamps_and_drops():
    # 越界窗口绕过 pydantic 校验构造（模拟外部输入），normalize 必须兜底裁剪
    ws = [
        TimeWindow.model_construct(start_sec=-2.0, end_sec=3.0),  # 越界下界（clamp 到 0）
        TimeWindow.model_construct(start_sec=50.0, end_sec=70.0),  # 超出 duration=60（clamp 到 60）
    ]
    out = normalize_windows(ws, duration_sec=60.0)
    assert all(is_valid_window(w, 60.0) for w in out)
    assert len(out) == 2
    assert (out[0].start_sec, out[0].end_sec) == (0.0, 3.0)
    assert (out[1].start_sec, out[1].end_sec) == (50.0, 60.0)


def test_normalize_min_window_expands():
    w = TimeWindow(start_sec=10.0, end_sec=11.0)
    out = normalize_windows([w], duration_sec=60.0, min_sec=3.0)
    assert out[0].duration_sec >= 3.0 - 1e-9


def test_normalize_max_window_truncates():
    w = TimeWindow(start_sec=0.0, end_sec=50.0)
    out = normalize_windows([w], duration_sec=60.0, max_sec=10.0)
    assert out[0].duration_sec <= 10.0 + 1e-9
