import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from queryseek.retrieval.localizer import (
    LocalizationHead, condition_features, decode_spans, localization_loss, supervision,
)
from scripts.localizer_pilot import check_video, features, fit, prediction, read
from scripts.sampling_probe import ROOT


def config():
    return read(ROOT/'configs/localizer_pilot_v1.json')


def test_supervision_half_open_overlapping_shortest_window_and_real_duration():
    labels, offsets = supervision([0., 2., 3., 4., 6.], 8., [[1., 6.], [2., 4.]])
    assert labels.tolist() == [0, 1, 1, 1, 0]
    assert offsets[1].tolist() == [0, .25]
    assert offsets[2].tolist() == [.125, .125]
    assert offsets[3].tolist() == [.375, .25]
    with pytest.raises(ValueError): supervision([0., 1.], 2., [[1., 3.]])
    with pytest.raises(ValueError): supervision([0., 1.], 2., [[.2, .3]])
    for timeline in [[0., 0.], [1., 0.], [0., 2.], [float('nan')]]:
        with pytest.raises(ValueError): condition_features(np.ones((len(timeline), 2)), [1., 0.], timeline, 2.)


def test_query_condition_has_interaction_and_zeroing_removes_all_query_channels():
    v = np.eye(2, dtype=np.float32); t = [.1, 1.1]
    a = condition_features(v, [1, 0], t, 2)
    b = condition_features(v, [0, 1], t, 2)
    z = condition_features(v, [1, 0], t, 2, drop_query=True)
    assert not torch.equal(a, b)
    assert torch.count_nonzero(z[:, 2:7]) == 0
    assert z[:, -1].tolist() == [0., 1.]


def test_frame_head_is_permutation_equivariant_and_temporal_head_uses_neighbors():
    torch.manual_seed(2); x = torch.randn(7, 9)
    frame = LocalizationHead(2, 8, 'frame_mlp')
    temporal = LocalizationHead(2, 8, 'temporal_conv')
    perm = torch.tensor([0, 3, 6, 1, 4, 2, 5])
    assert torch.allclose(frame(x[perm])[0], frame(x)[0][perm], atol=1e-6)
    assert not torch.allclose(temporal(x[perm])[0], temporal(x)[0][perm], atol=1e-6)
    changed = x.clone(); changed[3] += 4
    assert torch.equal(frame(changed)[0][2], frame(x)[0][2])
    assert not torch.equal(temporal(changed)[0][2], temporal(x)[0][2])


@pytest.mark.parametrize('architecture', ['frame_mlp', 'temporal_conv'])
def test_head_receives_finite_boundary_and_relevance_gradients(architecture):
    head = LocalizationHead(2, 8, architecture)
    x = condition_features(np.eye(2), [1, 0], [0., 1.], 2.)
    y, d = supervision([0., 1.], 2., [[0., 1.]])
    loss, _ = localization_loss(*head(x), y, d, config()['loss_weights'])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.predict.weight.grad.abs().sum() > 0


def test_decode_clamps_half_open_seconds_preserves_score_order_and_rejects_nan():
    windows = decode_spans([0., 2., 4.], 5., [-1., 2., 0.], [[.4, .4], [.4, .4], [.4, .4]], 2, .7)
    assert len(windows) == 2 and windows[0].score > windows[1].score
    assert all(0 <= w.start_sec < w.end_sec <= 5. for w in windows)
    with pytest.raises(ValueError): decode_spans([0.], 5., [0.], [[float('nan'), 1.]])
    with pytest.raises(ValueError): decode_spans([0.], 5., [0.], [[-1., 1.]])


def test_real_fixture_duration_contract_rejects_out_of_bounds_gt(demo):
    cfg = config(); entry = {'vid': 'synthetic', 'duration': 30., 'relevant_windows': [[1., 2.]]}
    from queryseek.ingest.ffprobe import asset_from_probe
    duration = asset_from_probe('synthetic', demo['video']).duration_sec
    entry['duration'] = duration
    info = check_video(entry, demo['video'], cfg)
    assert info['duration_sec'] > 0 and len(info['video_sha256']) == 64
    entry['relevant_windows'] = [[1., duration+1.]]
    with pytest.raises(ValueError, match='Ground truth'): check_video(entry, demo['video'], cfg)


def test_fixed_epoch_fit_learns_and_checkpoint_reloads_exactly(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(1); times = np.arange(12.)
    rr = []
    for i in range(3):
        images = rng.normal(0, .01, (12, 2)).astype(np.float32)
        images[3:8, 0] += 1
        rr.append({'images': images, 'query_embedding': np.array([1, 0], np.float32),
                   'times': times, 'duration_sec': 12., 'split': 'train' if i < 2 else 'validation',
                   'entry': {'qid': i, 'relevant_windows': [[3., 8.]]}})
    cfg = {**config(), 'hidden': 8, 'epochs': 15, 'learning_rate': .01}
    result = fit(rr, cfg, 'temporal_conv', tmp_path)
    assert result['train_loss_decreased'] and result['checkpoint_reload_exact']
    assert result['all_gradients_finite'] and result['epochs'] == 15
    assert len(json.loads((tmp_path/'temporal_conv_history.json').read_text())) == 15


def test_inference_has_no_dependency_on_gt():
    model = LocalizationHead(2, 8).eval()
    record = {'images': np.eye(2, dtype=np.float32), 'query_embedding': np.array([1, 0], np.float32),
              'times': [0., 1.], 'duration_sec': 2.}
    first, logits = prediction(model, record, config())
    second, other = prediction(model, {**record, 'entry': {'relevant_windows': [[100, 200]]}}, config())
    assert first == second and torch.equal(logits, other)


def test_feature_cache_reuses_visual_work_but_changes_with_sampling_and_rejects_corruption(demo, tmp_path, monkeypatch):
    import scripts.localizer_pilot as lp
    from queryseek.ingest.ffprobe import asset_from_probe
    task_root = tmp_path/'repo'; (task_root/'queryseek/ingest').mkdir(parents=True)
    (task_root/'queryseek/ingest/frames.py').write_text('synthetic extractor signature')
    monkeypatch.setattr(lp, 'ROOT', task_root)
    monkeypatch.setattr(lp, 'checked_output', lambda p: p)
    class FakeEncoder:
        model_id = 'mock_feature_cache'
        _preprocess = 'synthetic'
        image_calls = 0
        def encode_images(self, paths):
            self.image_calls += 1
            return np.tile(np.array([[1, 0]], np.float32), (len(paths), 1))
        def encode_texts(self, texts): return np.array([[1, 0]], np.float32)
    encoder = FakeEncoder(); asset = asset_from_probe('fixture', demo['video'])
    record = {'video': asset.path, 'video_sha256': asset.sha256, 'duration_sec': asset.duration_sec,
              'entry': {'query': 'synthetic', 'relevant_windows': [[0., 2.]]}}
    cfg = {**config(), 'max_side': 128, 'fps': .25}
    a = features(record, cfg, encoder, 'mockweights'); b = features(record, cfg, encoder, 'mockweights')
    assert not a['feature_cache_hit'] and b['feature_cache_hit'] and encoder.image_calls == 1
    assert np.array_equal(a['times'], b['times'])
    c = features(record, {**cfg, 'fps': .5}, encoder, 'mockweights')
    assert c['cache_key'] != a['cache_key'] and encoder.image_calls == 2
    cached = next((task_root/'outputs/localizer_text_cache').glob('*.npy'))
    cached.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='text cache'): features(record, cfg, encoder, 'mockweights')


def test_frozen_pilot_rejects_same_source_across_splits_and_evaluation_overlap(tmp_path, monkeypatch):
    import scripts.localizer_pilot as lp
    monkeypatch.setattr(lp, 'ROOT', tmp_path)
    monkeypatch.setattr(lp, 'checked_output', lambda p: p)
    root = tmp_path/'outputs/data'; root.mkdir(parents=True)
    rr = [{'qid': i, 'vid': f'source{i}_0.0_10.0', 'query': 'A visible event', 'duration': 10.,
           'relevant_windows': [[1., 4.]]} for i in range(32)]
    for name, data in [('pilot_train.jsonl', rr[:24]), ('pilot_validation.jsonl', rr[24:])]:
        (root/name).write_text('\n'.join(json.dumps(x) for x in data), encoding='utf-8')
    (root/'excluded_sources.json').write_text('[]')
    assert len(lp.frozen_entries({'data_run': 'outputs/data'})) == 32
    (root/'excluded_sources.json').write_text('["source3"]')
    with pytest.raises(ValueError, match='isolation'): lp.frozen_entries({'data_run': 'outputs/data'})
    (root/'excluded_sources.json').write_text('[]')
    rr[24]['vid'] = rr[0]['vid']
    (root/'pilot_validation.jsonl').write_text('\n'.join(json.dumps(x) for x in rr[24:]))
    with pytest.raises(ValueError, match='isolation'): lp.frozen_entries({'data_run': 'outputs/data'})
