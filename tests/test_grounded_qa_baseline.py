import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_grounded_qa_baseline as baseline


def request(video='unused'):
    return {'qid': '1', 'video_id': 'synthetic', 'question': 'What appears?',
            'choices': ['blue', 'red'], 'duration_sec': 60., 'video_path': str(video)}


def test_runner_rejects_labels_before_loading_media():
    with pytest.raises(ValueError, match='whitelisted'):
        baseline.check_requests([{**request(), 'right_answer': 'A'}])
    with pytest.raises(ValueError, match='duplicate'):
        baseline.check_requests([request(), request()])


def test_model_manifest_checks_revision_coverage_and_content(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}', encoding='utf-8')
    manifest = {'revision': 'fixed', 'files': {'config.json': hashlib.sha256(b'{}').hexdigest()}}
    (model / 'QUERYSEEK_SNAPSHOT.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert len(baseline.verify_model_snapshot(model, 'fixed')) == 64
    with pytest.raises(ValueError, match='revision'):
        baseline.verify_model_snapshot(model, 'other')
    (model / 'extra.json').write_text('x', encoding='utf-8')
    with pytest.raises(ValueError, match='cover all'):
        baseline.verify_model_snapshot(model, 'fixed')


def test_mock_baseline_real_media_cache_and_missing_denominator(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, 'ROOT', tmp_path)
    requests = tmp_path / 'requests.json'
    rows = [request(demo['video']), {**request(tmp_path / 'absent.mp4'), 'qid': '2'}]
    requests.write_text(json.dumps(rows), encoding='utf-8')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'frame_budgets': [16, 32], 'model_revision': 'unused'}), encoding='utf-8')
    first = baseline.run(requests, tmp_path / 'outputs/first', config, mock=True)
    assert first['scheduled'] == 2 and first['completed'] == first['failed'] == 1
    assert first['vlm_calls'] == 0 and first['accuracy'] is None
    record = json.loads((tmp_path / 'outputs/first/predictions.jsonl').read_text('utf-8').splitlines()[0])
    assert len(record['source_times_sec']) == 16
    assert record['preprocessing']['new_frames_extracted'] == 16
    assert all(0 <= t < 60.1 for t in record['source_times_sec'])
    second = baseline.run(requests, tmp_path / 'outputs/second', config, mock=True)
    cached = json.loads((tmp_path / 'outputs/second/predictions.jsonl').read_text('utf-8').splitlines()[0])
    assert second['completed'] == 1 and cached['preprocessing']['cache_hit']
    assert cached['preprocessing']['new_frames_extracted'] == 0
    assert record['source_times_sec'] == cached['source_times_sec']
    with pytest.raises(FileExistsError):
        baseline.run(requests, tmp_path / 'outputs/first', config, mock=True)
    with pytest.raises(ValueError, match='outputs/'):
        baseline.run(requests, tmp_path / 'outside', config, mock=True)
