import json
import os
from pathlib import Path

import pytest

from scripts import grounded_qa_input_study as study


def test_clue_union_sampling_preserves_seconds_and_avoids_gaps():
    assert study.clue_times([[10., 12.], [20., 22.]], 30., 4) == [10.5, 11.5, 20.5, 21.5]
    with pytest.raises(ValueError):
        study.clue_times([[10., 12.]], 11., 4)


def test_native_timebase_keeps_irregular_source_pts_and_pair_averages():
    times = [2.013, 3.12345, 103.333333, 1000.012345]
    meta = study.source_pts_metadata(times, 1200.)
    recovered = [i/meta['fps'] for i in meta['frames_indices']]
    assert recovered == pytest.approx(times, abs=.00000051)
    assert study.expected_video_timestamps(recovered) == pytest.approx([2.568225, 551.672839])
    for bad in [[0., 1., 1., 2.], [0., 1., 2.], [0., 1., 2., 1200.]]:
        with pytest.raises(ValueError):
            study.source_pts_metadata(bad, 1200.)


def test_prompts_exclude_annotation_and_scope_fields():
    request = {'question': 'What?', 'choices': ['x', 'y'], 'duration_sec': 100.,
               'right_answer': 'SECRET_ANSWER', 'gt_intervals_sec': [[70., 80.]]}
    text, schema = study.prompt_and_schema(request, {'output': 'answer'})
    assert 'SECRET_ANSWER' not in text and '70' not in text and 'duration' not in text
    assert schema['required'] == ['answer']
    text, schema = study.prompt_and_schema(request, {'output': 'grounded'})
    assert 'duration_sec' in text and 'SECRET_ANSWER' not in text
    assert schema['required'] == ['answer', 'intervals']


def test_prepare_mock_matrix_rejects_labels_in_inference_and_preserves_cache(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study.baseline, 'ROOT', tmp_path)
    requests = tmp_path/'requests.json'
    requests.write_text(json.dumps([{'qid': '1', 'video_id': 'sample', 'question': 'Color?',
        'choices': ['blue', 'red'], 'duration_sec': 60., 'video_path': str(demo['video'])}]))
    labels = tmp_path/'labels.json'
    labels.write_text(json.dumps([{'qid': '1', 'right_answer': 'A', 'gt_intervals_sec': [[10., 12.]]}]))
    conditions = [
        {'name': 'text', 'input': 'text', 'scope': 'prior_diagnostic', 'frames': 0, 'output': 'answer'},
        {'name': 'oracle', 'input': 'images', 'scope': 'oracle_diagnostic', 'frames': 4, 'output': 'answer'},
        {'name': 'oracle_video', 'input': 'video', 'scope': 'oracle_diagnostic', 'frames': 4, 'output': 'answer'},
        {'name': 'global', 'input': 'video', 'scope': 'global_baseline', 'frames': 4, 'output': 'grounded'}]
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'requests_sha256': study.sha256_file(requests),
        'labels_sha256': study.sha256_file(labels), 'conditions': conditions, 'max_generations': 4}))
    prepared_dir = tmp_path/'outputs/prepared'
    study.prepare(requests, labels, config, prepared_dir)
    prepared = prepared_dir/'prepared.json'
    rows = json.loads(prepared.read_text('utf-8'))
    assert rows[1]['frames'] == rows[2]['frames']
    assert rows[2]['preprocessing']['new_frames_extracted'] == 0
    assert all(10 <= f['timestamp_sec'] < 12 for f in rows[1]['frames'])
    result = study.run(prepared, config, tmp_path/'outputs/run', mock=True)
    assert result['scheduled'] == result['completed'] == 4 and result['backend_calls'] == 0
    with pytest.raises(FileExistsError):
        study.run(prepared, config, tmp_path/'outputs/run', mock=True)
    rows[0]['request']['right_answer'] = 'A'
    with pytest.raises(ValueError, match='whitelisted'):
        study.validate_prepared(rows, json.loads(config.read_text()))


@pytest.mark.skipif(not os.environ.get('QUERYSEEK_QWEN_SNAPSHOT'), reason='Requires locally provisioned Qwen processor only')
def test_real_native_processor_pixels_timestamps_and_no_resampling(demo, tmp_path, monkeypatch):
    from transformers import AutoProcessor
    monkeypatch.setattr(study.baseline, 'ROOT', tmp_path)
    processor = AutoProcessor.from_pretrained(os.environ['QUERYSEEK_QWEN_SNAPSHOT'], local_files_only=True)
    request = {'qid': '1', 'video_id': 'sample', 'question': 'Color?', 'choices': ['blue', 'red'],
               'duration_sec': 60., 'video_path': str(demo['video'])}
    areas = []
    for count in [4, 8]:
        _, frames, _ = study.baseline.sampled_frames(request, count)
        inputs, audit = study.native_inputs(processor, 'What appears?', frames, 65536, 60.)
        assert inputs.video_grid_thw[0, 0] == count//2
        assert audit['grouped_source_times_sec'] == pytest.approx(study.expected_video_timestamps([f.timestamp_sec for f in frames]), abs=.000001)
        areas.append(audit['actual_pixels_per_frame'])
    assert areas[0] == areas[1]
