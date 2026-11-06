"""核心数据结构测试：时间契约（开发说明书 §6.1）——浮点秒、[start, end)、0<=start<end<=duration。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from queryseek.schemas import SubtitleSegment, TimeWindow


class TestTimeWindow:
    def test_valid(self):
        w = TimeWindow(start_sec=1.0, end_sec=2.5, score=0.8, evidence_ids=["a"])
        assert w.duration_sec == 1.5

    def test_start_must_be_less_than_end(self):
        with pytest.raises(ValidationError):
            TimeWindow(start_sec=2.0, end_sec=2.0)
        with pytest.raises(ValidationError):
            TimeWindow(start_sec=3.0, end_sec=2.0)

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            TimeWindow(start_sec=-1.0, end_sec=2.0)

    def test_nan_rejected(self):
        with pytest.raises(ValidationError):
            TimeWindow(start_sec=float("nan"), end_sec=2.0)
        with pytest.raises(ValidationError):
            TimeWindow(start_sec=0.0, end_sec=float("inf"))

    def test_clamp_to(self):
        # clamp 到 [0, 10]
        w2 = TimeWindow(start_sec=8.0, end_sec=12.0).clamp_to(10.0)
        assert (w2.start_sec, w2.end_sec) == (8.0, 10.0)

    def test_clamp_empty_raises(self):
        w = TimeWindow(start_sec=11.0, end_sec=12.0)
        with pytest.raises(ValueError):
            w.clamp_to(10.0)

    def test_mutable_defaults_are_isolated(self):
        a = TimeWindow(start_sec=0.0, end_sec=1.0)
        b = TimeWindow(start_sec=1.0, end_sec=2.0)
        a.evidence_ids.append("x")
        assert b.evidence_ids == []


class TestSubtitleSegment:
    def test_valid(self):
        s = SubtitleSegment(start_sec=0.0, end_sec=2.0, text="hello")
        assert s.text == "hello"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            SubtitleSegment(start_sec=0.0, end_sec=2.0, text="   ")

    def test_invalid_interval_rejected(self):
        with pytest.raises(ValidationError):
            SubtitleSegment(start_sec=2.0, end_sec=2.0, text="x")
