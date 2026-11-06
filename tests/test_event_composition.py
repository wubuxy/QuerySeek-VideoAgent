import json

import numpy as np
import pytest

from queryseek.retrieval.event_composition import ordered_event_margin, ordered_event_score, sequence_time_controls
from queryseek.schemas import TimeWindow
from scripts.event_composition_probe import phase_diagnosis, run_probe, summarize, validate_chain
from scripts.review_event_chains import cached_review, envelope


def test_temporal_composition_recovers_order_that_mean_pooling_erases():
    scores = np.array([[1., 0.]] * 4 + [[0., 1.]] * 4)
    assert np.mean(scores) == np.mean(scores[::-1])
    assert ordered_event_score(scores)['split_position'] == 4
    assert ordered_event_margin(scores)['margin'] > 0
    assert ordered_event_margin(scores[::-1])['margin'] < 0
    assert ordered_event_margin(scores[:, ::-1])['margin'] < 0


def test_equal_event_support_and_ties_do_not_fabricate_direction():
    scores = np.full((8, 2), .3)
    assert ordered_event_margin(scores)['margin'] == pytest.approx(0.)
    result = ordered_event_score(np.array([[0., 1.]] * 2 + [[1., 0.]] * 6))
    assert 2 <= result['split_position'] <= 6
    for bad in (np.ones((8, 3)), np.ones((3, 2)), np.full((8, 2), np.nan)):
        with pytest.raises(ValueError):
            ordered_event_score(bad)
    with pytest.raises(ValueError):
        ordered_event_score(scores, True)


def test_source_time_reconstruction_and_reverse_are_not_extra_gold_examples():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(16, 2))
    controls = sequence_time_controls(scores, scores, [i + .1 for i in range(16)])
    original = controls['original']['sequence_axis']['margin']
    assert controls['reverse']['sequence_axis']['margin'] == pytest.approx(-original)
    for c in controls.values():
        assert c['source_pts_axis']['margin'] == pytest.approx(original)
        assert c['unordered_score'] == pytest.approx(controls['original']['unordered_score'])
    assert controls['shuffle_12']['display_order_label'] == 'no binary label'
    assert controls['reverse']['source_time_label'] == 'original A before B'


def test_source_times_and_context_are_validated():
    scores = np.ones((8, 2))
    for times in ([0.] * 8, list(range(8))[::-1], [np.nan] * 8, list(range(7))):
        with pytest.raises(ValueError):
            sequence_time_controls(scores, scores, times)
    with pytest.raises(ValueError):
        sequence_time_controls(scores, np.zeros((7, 2)), list(range(8)))


def test_oracle_annotations_stay_out_of_score_search_and_phase_denominators():
    chain = {'window': [0., 8.], 'event_a_window': [0., 2.], 'event_b_window': [6., 8.],
             'event_a': 'A', 'event_b': 'B', 'context_a': 'object', 'context_b': 'object'}
    assert validate_chain(chain, 8.) == (0., 8.)
    scores = np.array([[1., 0.]] * 4 + [[0., 1.]] * 4)
    assert ordered_event_score(scores)['split_position'] == 4  # not either annotated phase boundary
    phases = phase_diagnosis(scores, list(range(8)), chain)
    assert phases['a_frames'] == phases['b_frames'] == 2
    assert phases['a_vs_b_in_a_phase'] == phases['b_vs_a_in_b_phase'] == 1.
    with pytest.raises(ValueError):
        validate_chain({**chain, 'event_b_window': [1., 3.]}, 8.)
    with pytest.raises(RuntimeError):
        phase_diagnosis(scores, list(range(8)), {**chain, 'event_b_window': [7., 8.]})


def test_failed_cases_keep_intended_denominator_and_review_exclusions():
    manifest = {'chains': [{'qid': 1}, {'qid': 2}], 'excluded': [{'qid': 3}]}
    result = summarize([], manifest, [{'qid': 1, 'error_type': 'Failed'}], True)
    assert result['reviewed'] == 3 and result['eligible'] == 2 and result['completed'] == 0
    assert result['coverage'] == 0. and result['composed_positive'] == 0


def test_review_envelope_and_cache_keep_source_pts_and_configuration(tmp_path, monkeypatch, has_ffmpeg):
    if not has_ffmpeg:
        pytest.skip('FFmpeg unavailable')
    import scripts.review_event_chains as review
    import scripts.sampling_probe as sampling
    from scripts.make_synthetic_fixture import build_timing_video
    from queryseek.ingest.ffprobe import asset_from_probe
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    monkeypatch.setattr(sampling, 'ROOT', tmp_path)
    asset = asset_from_probe('timing', build_timing_video(tmp_path/'assets'))
    windows = [TimeWindow(start_sec=2., end_sec=4.), TimeWindow(start_sec=5., end_sec=6.)]
    assert envelope(windows, 8., 1.) == (1., 7.)
    with pytest.raises(ValueError):
        envelope(windows, 8., float('nan'))
    frames, hit = cached_review(asset, (1., 7.), 1., 128)
    assert not hit and len(frames) == 6 and all(1. <= f.timestamp_sec < 7. for f in frames)
    frames2, hit2 = cached_review(asset, (1., 7.), 1., 128)
    assert hit2 and frames2 == frames
    denser, hit3 = cached_review(asset, (1., 7.), 2., 128)
    assert not hit3 and len(denser) == 12


def test_generated_mock_chain_e2e_reuses_embeddings_and_preserves_failures(demo, tmp_path, monkeypatch):
    import scripts.event_composition_probe as probe
    import scripts.sampling_probe as sampling
    from queryseek.config import Config
    from queryseek.eval.runner import EvalSample
    monkeypatch.setattr(sampling, 'ROOT', tmp_path)
    monkeypatch.setattr(probe, 'snapshot', lambda *args: {})
    sample = EvalSample(0, str(demo['video']), 'sample', None, 'test', [])
    chain = {'qid': 0, 'window': [0., 24.], 'event_a_window': [0., 12.], 'event_b_window': [12., 24.],
             'event_a': 'intro welcome', 'event_b': 'architecture', 'context_a': 'intro', 'context_b': 'architecture'}
    manifest = {'chains': [chain], 'frames_per_chain': 16, 'min_frames_per_event': 2,
                'shuffle_seeds': [12, 23], 'excluded': []}
    cfg = Config.load(apply_env=False).resolved()
    hints = json.loads(demo['labels'].read_text(encoding='utf-8'))['segments']
    first = run_probe(manifest, cfg, tmp_path/'outputs/first', True, [sample], hints)
    second = run_probe(manifest, cfg, tmp_path/'outputs/second', True, [sample], hints)
    assert first['coverage'] == second['coverage'] == 1. and first['mock']
    assert first['new_encoded_frames'] == 16 and second['new_encoded_frames'] == 0
    row = json.loads((tmp_path/'outputs/first/records.jsonl').read_text())
    assert len(set(row['actual_times_sec'])) == 16
    assert (tmp_path/'outputs/first/per_query.csv').is_file()
    with pytest.raises(ValueError):
        run_probe(manifest, cfg, tmp_path/'outside', True, [sample], hints)
    def fail(*args, **kwargs):
        raise RuntimeError('Synthetic failure')
    monkeypatch.setattr(probe, 'extra_features', fail)
    result = run_probe(manifest, cfg, tmp_path/'outputs/failed', True, [sample], hints)
    assert result['coverage'] == 0. and result['eligible'] == 1 and len(result['errors']) == 1
