from copy import deepcopy
import math

import pytest

from queryseek.eval.grounded_qa import (
    checked_intervals, evaluate_predictions, interval_set_iou,
    load_samples, score_prediction, uniform_times,
)


def row(qid=1):
    return {"qid": qid, "video_id": "synthetic", "question": "Which card appears?",
            "choices": ["blue", "red"], "right_answer": "A", "answer": "blue",
            "duration": 60., "gt_timestamps": [12., 24.],
            "gt_all_timestamps": [[12., 24.], [48., 60.]]}


def test_model_input_has_no_label_or_answer_fields():
    original = row()
    sample = load_samples([original])[0]
    prompt = sample.model_input()
    assert set(prompt) == {"qid", "video_id", "question", "choices", "duration_sec"}
    prompt["choices"].append("changed")
    assert len(sample.choices) == len(original["choices"]) == 2


def test_interval_union_preserves_multiple_clues_and_does_not_mutate():
    truth = [[0., 10.], [20., 30.]]
    prediction = [[5., 12.], [0., 8.], [20., 25.]]
    before = deepcopy(prediction)
    assert interval_set_iou(prediction, truth, 60.) == pytest.approx(15/22)
    assert prediction == before
    sample = load_samples([row()])[0]
    assert len(sample.gt_intervals) == 2
    # Taking max IoU over individual GT segments would incorrectly yield 1.
    assert interval_set_iou([[12., 24.]], sample.gt_intervals, 60.) == .5


@pytest.mark.parametrize("pair", [[0., math.nan], [0., math.inf], [-1., 2.], [1., 1.], [1., 61.], [False, 1.], ["0", 1.]])
def test_invalid_times_are_rejected(pair):
    with pytest.raises(ValueError):
        checked_intervals([pair], 60.)


def test_missing_and_failed_predictions_stay_in_denominator():
    samples = load_samples([row(1), row(2), row(3)])
    scored = evaluate_predictions(samples, [
        {"qid": 1, "status": "ok", "answer": "A", "intervals": [[12., 24.], [48., 60.]]},
        {"qid": 2, "status": "error"},
    ])
    assert scored["metrics"]["n"] == 3
    assert scored["metrics"]["missing_outputs"] == 1
    assert scored["metrics"]["acc_at_iou"] == pytest.approx(1/3)
    assert [r["status"] for r in scored["records"]] == ["ok", "failed", "missing"]


def test_joint_metric_requires_both_answer_and_evidence():
    sample = load_samples([row()])[0]
    correct = {"status": "ok", "answer": "A", "intervals": [[12., 24.]]}
    assert score_prediction(sample, correct)["joint_at_05"] == 1
    assert score_prediction(sample, {**correct, "answer": "B"})["acc_at_iou"] == 0
    assert score_prediction(sample, {**correct, "intervals": [[24., 36.]]})["acc_at_iou"] == 0
    assert score_prediction(sample, {**correct, "intervals": [[12., 61.]]})["status"] == "invalid"
    assert score_prediction(sample, {**correct, "intervals": []})["status"] == "invalid"


def test_unknown_or_duplicate_predictions_rejected():
    samples = load_samples([row()])
    with pytest.raises(ValueError, match="prediction"):
        evaluate_predictions(samples, [{"qid": 999}])
    with pytest.raises(ValueError, match="prediction"):
        evaluate_predictions(samples, [{"qid": 1}, {"qid": 1}])


def test_invalid_sample_identity_or_labels_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        load_samples([row(), row()])
    with pytest.raises(ValueError, match="identifier"):
        load_samples([{**row(), "video_id": "../secret"}])
    with pytest.raises(ValueError, match="Answer"):
        load_samples([{**row(), "right_answer": "A or B"}])


def test_uniform_requests_use_half_open_interval_without_labels():
    times = uniform_times(60., 16)
    assert len(times) == len(set(times)) == 16
    assert times[0] == 1.875 and times[-1] == 58.125
    with pytest.raises(ValueError):
        uniform_times(60., 0)


def test_prepare_checks_hashes_protects_output_and_separates_labels(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import prepare_grounded_qa as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    annotation = tmp_path / 'annotations.json'
    annotation.write_text(json.dumps([row()]), encoding='utf-8')
    sha = hashlib.sha256(annotation.read_bytes()).hexdigest()
    selection = {'annotation_sha256': sha, 'samples': [
        {'qid': 1, 'video_id': 'synthetic', 'duration': 60.}]}
    (tmp_path / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    config = {'annotation': 'annotations.json', 'annotation_sha256': sha,
              'selection': 'selection.json', 'video_root': 'media', 'frame_budgets': [16]}
    config_path = tmp_path / 'protocol.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    # Supply a source file for source-provenance hashing, not a video fixture.
    source = tmp_path / 'queryseek/eval/grounded_qa.py'
    source.parent.mkdir(parents=True)
    source.write_text('fixture source', encoding='utf-8')
    output = tmp_path / 'outputs/pilot'
    result = module.prepare(config_path, output)
    assert result['pilot_sources'] == 1 and result['vlm_calls'] == 0
    request = json.loads((output / 'requests.json').read_text('utf-8'))[0]
    assert not {'right_answer', 'gt_intervals_sec', 'gt_timestamps', 'answer'} & set(request)
    with pytest.raises(FileExistsError):
        module.prepare(config_path, output)
    with pytest.raises(ValueError, match='outputs/'):
        module.prepare(config_path, tmp_path / 'outside')
    annotation.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        module.prepare(config_path, tmp_path / 'outputs/other')
