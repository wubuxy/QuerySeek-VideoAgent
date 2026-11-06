import pytest
pytest.importorskip('torch')

from scripts.analyze_localizer_pilot import proposal_diagnostic


def test_oracle_diagnostic_counts_empty_queries_and_distinguishes_top1_from_top5():
    records = [{'entry': {'qid': i, 'relevant_windows': [[2., 6.]]}} for i in range(3)]
    predictions = {0: [(0., 1., .9), (2., 6., .8)], 1: [(2., 6., .5)]}
    result = proposal_diagnostic(predictions, records)
    assert result == {'queries': 3, 'top1_iou05_hits': 1, 'top5_iou05_oracle_hits': 2}
