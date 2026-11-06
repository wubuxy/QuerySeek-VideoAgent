import pytest

from queryseek.eval.runner import MethodResult, _method_metrics
from queryseek.schemas import TimeWindow


def test_rejections_still_count_query_cost_and_separate_negative_metrics():
    window = TimeWindow(start_sec=1, end_sec=5, score=1)
    results = [
        MethodResult(0, "present", "E4", 16, windows=[window], indexed_frames=100, elapsed_ms=20),
        MethodResult(1, "absent", "E4", 16, windows=[], status="no_result",
                     used_frames=8, vlm_calls=1, indexed_frames=100, elapsed_ms=40),
        MethodResult(2, "absent", "E4", 16, windows=[window], indexed_frames=100, elapsed_ms=60),
    ]
    metrics = _method_metrics(results, {0: [window], 1: [], 2: []}, 3)
    assert metrics["negative_query_count"] == 2 and metrics["positive_query_count"] == 1
    assert metrics["negative_rejection_rate"] == .5 and metrics["negative_false_positive_rate"] == .5
    assert metrics["positive_no_result_rate"] == 0
    assert metrics["avg_latency_ms"] == 40 and metrics["avg_vlm_calls"] == pytest.approx(1/3)
    assert metrics["avg_observed_frames"] == pytest.approx(8/3)
    assert metrics["avg_indexed_frames"] == 100


def test_no_negative_queries_are_not_reported_as_verified_rejection():
    window = TimeWindow(start_sec=1, end_sec=5, score=1)
    results = [MethodResult(0, "event", "E4", 16, windows=[], status="no_result")]
    metrics = _method_metrics(results, {0: [window]}, 1)
    assert metrics["negative_rejection_rate"] is None
    assert metrics["negative_false_positive_rate"] is None
    assert metrics["positive_no_result_rate"] == 1
