import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.state_fact_bridge import bridge_summary, caption_layout, opaque_ids, run_bridge, validate_facts


def test_query_free_layout_hides_original_order_and_keeps_source_times_with_images():
    row = {'video_sha256': 'a'*64, 'actual_times_sec': [float(i) for i in range(16)],
           'chain': {'event_a': 'secret action query', 'event_b': 'another query'}}
    ids, order, text = caption_layout(row, 'shuffle_no_time', 12)
    assert 'secret action' not in text and 'another query' not in text and 'source_time_sec' not in text
    metadata = json.loads(text.rsplit('\n', 1)[-1])
    assert [x['frame_id'] for x in metadata] == [ids[i] for i in order]
    assert order != list(range(16)) and len(set(ids)) == 16
    _, order2, timed = caption_layout(row, 'shuffle_source_time', 12)
    timed_meta = json.loads(timed.rsplit('\n', 1)[-1])
    assert order2 == order
    assert [x['source_time_sec'] for x in timed_meta] == [row['actual_times_sec'][i] for i in order]


def test_caption_mapping_requires_complete_unique_ids_and_restores_source_identity():
    ids = ['opaque_a', 'opaque_b']
    data = {'observations': [{'frame_id': 'opaque_b', 'fact': 'Open box.'},
                             {'frame_id': 'opaque_a', 'fact': 'Closed box.'}]}
    assert validate_facts(data, ids) == ['Closed box.', 'Open box.']
    for wrong in ({'observations': data['observations'][:1]},
                  {'observations': [data['observations'][0]]*2},
                  {'observations': [{'frame_id': 'unknown', 'fact': 'X'}, data['observations'][1]]},
                  {'observations': [{'frame_id': 'opaque_b', 'fact': ''}, data['observations'][1]]}):
        with pytest.raises(ValueError):
            validate_facts(wrong, ids)


def test_partial_groups_do_not_shrink_failure_denominator():
    intended = [(48, g) for g in ('original_no_time', 'original_source_time', 'shuffle_no_time', 'shuffle_source_time')]
    report = bridge_summary([], intended, [{'qid': 48}], 1, 16, {'total_tokens': 0}, False)
    assert report['planned_groups'] == 4 and report['completed'] == 0 and report['coverage'] == 0.
    assert report['actual_vlm_calls'] == 1 and all(g['planned'] == 1 for g in report['groups'].values())


def test_call_bound_is_checked_before_io_and_no_silent_retry():
    manifest = {'qids': [1,2,3], 'groups': ['original_no_time', 'original_source_time', 'shuffle_no_time', 'shuffle_source_time'],
                'max_calls': 8, 'max_retries': 0}
    with pytest.raises(ValueError, match='eight-call'):
        run_bridge(manifest, [], None, Path('outside'), mock=True)


def test_synthetic_bridge_e2e_caption_cache_and_schema_failures(demo, tmp_path, monkeypatch):
    import scripts.state_fact_bridge as bridge
    import scripts.sampling_probe as sampling
    from queryseek.config import Config
    from queryseek.embeddings.base import MockEmbedding
    from queryseek.ingest.ffprobe import asset_from_probe
    from queryseek.ingest.pipeline import VideoIndex
    monkeypatch.setattr(sampling, 'ROOT', tmp_path)
    monkeypatch.setattr(bridge, 'ROOT', tmp_path)
    monkeypatch.setattr(bridge, 'snapshot', lambda *args: {})
    cfg = Config.load(apply_env=False).resolved()
    asset = asset_from_probe('sample', demo['video'])
    hints = json.loads(demo['labels'].read_text(encoding='utf-8'))['segments']
    times = [(i+.5)*24/16 for i in range(16)]
    frames, _, _, _, _ = sampling.extra_features(VideoIndex(asset, tmp_path, content_hints=hints),
                                                 times, MockEmbedding(), cfg, True)
    row = {'qid': 0, 'video_sha256': asset.sha256, 'actual_times_sec': [f.timestamp_sec for f in frames],
           'requested_times_sec': times, 'chain': {'event_a': 'intro welcome', 'event_b': 'architecture',
                                                  'event_a_window': [0.,12.], 'event_b_window': [12.,24.]}}
    manifest = {'qids': [0], 'groups': ['original_no_time', 'original_source_time', 'shuffle_no_time', 'shuffle_source_time'],
                'max_calls': 8, 'max_retries': 0, 'images_per_call': 16, 'shuffle_seed': 12,
                'max_output_tokens': 2200, 'enable_thinking': False}
    first = run_bridge(manifest, [row], cfg, tmp_path/'outputs/first', True, {0: frames}, hints)
    second = run_bridge(manifest, [row], cfg, tmp_path/'outputs/second', True, {0: frames}, hints)
    assert first['coverage'] == second['coverage'] == 1. and first['actual_vlm_calls'] == 0
    assert first['groups'] == second['groups']
    records = [json.loads(line) for line in (tmp_path/'outputs/second/records.jsonl').read_text().splitlines()]
    assert len(records) == 4 and all(r['cache_hit'] for r in records)
    with pytest.raises(ValueError):
        run_bridge(manifest, [row], cfg, tmp_path/'outside', True, {0: frames}, hints)
    def fail(*args, **kwargs):
        raise ValueError('Unsupported frame IDs')
    monkeypatch.setattr(bridge, 'validate_facts', fail)
    bad = run_bridge(manifest, [row], cfg, tmp_path/'outputs/failed', True, {0: frames}, hints)
    assert bad['coverage'] == 0. and bad['planned_groups'] == 4 and len(bad['errors']) == 4
