"""融合测试：RRF + IoU-NMS（开发说明书 §7.2：不直接相加未校准分数）。"""

from __future__ import annotations

import pytest

from queryseek.retrieval.fusion import nms_dedupe, rrf_fuse
from queryseek.schemas import TimeWindow


def test_rrf_rank_effect():
    w_top = TimeWindow(start_sec=0.0, end_sec=10.0, score=0.9)
    w_mid = TimeWindow(start_sec=20.0, end_sec=30.0, score=0.8)
    fused = rrf_fuse([("m1", [w_top, w_mid]), ("m2", [])], rrf_k=60)
    assert fused[0].score == pytest.approx(1.0 / 61.0, abs=1e-6)
    assert fused[1].score == pytest.approx(1.0 / 62.0, abs=1e-6)


def test_rrf_consensus_beats_single():
    """两个方法都排第一的窗口，RRF 分数高于只在一个方法里出现的窗口。"""
    w_both = TimeWindow(start_sec=0.0, end_sec=10.0)
    w_only_a = TimeWindow(start_sec=20.0, end_sec=30.0)
    w_only_b = TimeWindow(start_sec=40.0, end_sec=50.0)
    fused = rrf_fuse(
        [("a", [w_both, w_only_a]), ("b", [w_both, w_only_b])], rrf_k=60
    )
    scores = {(w.start_sec, w.end_sec): w.score for w in fused}
    assert scores[(0.0, 10.0)] == pytest.approx(2.0 / 61.0, abs=1e-6)  # 共识窗口
    assert scores[(20.0, 30.0)] == pytest.approx(1.0 / 62.0, abs=1e-6)
    assert fused[0].start_sec == 0.0


def test_nms_dedupe():
    ws = [
        TimeWindow(start_sec=0.0, end_sec=10.0, score=0.9),
        TimeWindow(start_sec=9.0, end_sec=15.0, score=0.8),  # IoU 1/16 < 0.5 → 保留
        TimeWindow(start_sec=1.0, end_sec=9.0, score=0.7),  # IoU 0.8 ≥ 0.5 → 丢弃
    ]
    kept = nms_dedupe(ws, nms_iou=0.5)
    assert len(kept) == 2
    assert kept[0].score == 0.9


def test_nms_empty():
    assert nms_dedupe([]) == []
