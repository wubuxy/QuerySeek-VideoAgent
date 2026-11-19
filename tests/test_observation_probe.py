"""The isolated matrix freezes its inputs and never needs a live model."""
import json

import pytest

from queryseek.config import Config
from scripts.observation_probe import run_matrix


def test_mock_matrix_freezes_grid_sources_and_artifacts(qs_env, demo):
    folder, rows = run_matrix(demo['video'], demo['srt'], '系统架构', mock=True,
                               methods=['uniform'], budgets=[8], frames_per_call=[4, 8],
                               output_tokens=[128, 512], max_calls=4, config=Config.load())
    assert len(rows) == 4 and all(r['status'] == 'completed' for r in rows)
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['max_output_tokens_per_call'] == [128, 512]
    assert protocol['model_id'] == 'mock-vlm-1' and protocol['accuracy'] == 'TODO / 未测'
    assert len(protocol['source_sha256']['queryseek/models/metered.py']) == 64
    assert protocol['subtitle_sha256'] and protocol['mock_hint_sha256']
    for row in rows:
        assert row['metrics']['unique_source_frames'] == 8
        assert row['metrics']['model_calls'] == 8 // row['frames_per_call']
        costs = json.loads((folder.parents[1] / row['run_id'] / 'costs.json').read_text())
        assert costs['limits']['max_output_tokens_per_call'] == row['max_output_tokens_per_call']


def test_empty_or_invalid_grid_stops_before_ingest(qs_env):
    with pytest.raises(ValueError, match='output-token'):
        run_matrix('not-read.mp4', None, 'query', mock=True, methods=['uniform'],
                   budgets=[8], frames_per_call=[4], output_tokens=[0], max_calls=4)
