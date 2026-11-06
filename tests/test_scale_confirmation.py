import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts import scale_confirmation as study


def entry(q, source):
    return {'qid': q, 'vid': f'{source}_0.0_20.0'}


def test_holdout_keeps_every_query_and_old_prefix():
    old = [entry(1, 'old'), entry(2, 'old')]
    new = [entry(3, 'new'), entry(4, 'new')]
    assert study.holdout_partition(old+new, old, [entry(5, 'train')], ['excluded']) == (old, new)


@pytest.mark.parametrize('kind', ['source', 'query', 'excluded', 'duplicate', 'changed_old'])
def test_holdout_rejects_contamination(kind):
    old = [entry(1, 'old')]; val = old+[entry(2, 'new')]; train = [entry(3, 'train')]; excluded = []
    if kind == 'source': train = [entry(3, 'new')]
    if kind == 'query': train = [entry(2, 'train')]
    if kind == 'excluded': excluded = ['new']
    if kind == 'duplicate': val.append(entry(2, 'another'))
    if kind == 'changed_old': old = [entry(10, 'old')]
    with pytest.raises(ValueError): study.holdout_partition(val, old, train, excluded)


def test_unavailable_queries_remain_empty_denominator():
    rows = [entry(1, 'a'), entry(2, 'b')]
    assert study.complete_denominator({1: [(0., 1., .5)]}, rows) == {1: [(0., 1., .5)], 2: []}
    with pytest.raises(ValueError): study.complete_denominator({3: []}, rows)


def test_validation_tables_do_not_access_labels():
    r = {'entry': entry(1, 'a'), 'split': 'validation', 'images': np.eye(4),
         'query_embedding': np.array([0., .3, .8, .1]), 'times': np.array([0., 1., 2., 3.]), 'duration_sec': 4.}
    cfg = study.data.read(study.ROOT/'configs/multiscale_ranker_v1.json')
    _, first = study.build_tables([r], cfg)
    r['entry']['relevant_windows'] = 'must never be accessed'
    _, second = study.build_tables([r], cfg)
    assert not any('_y_' in key for key in first)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_generated_fixture_two_fit_and_replay(tmp_path, demo):
    # Only use the repository's standard fixture generator; no downloaded video.
    records = study.data.study_mock_examples(demo['video'], {})
    cfg = study.data.read(study.ROOT/'configs/multiscale_ranker_v1.json')
    p = {'ranker_config': cfg, 'model': study.data.read(study.ROOT/'configs/candidate_ranker_v1.json')['model']}
    out = tmp_path/'run'; out.mkdir()
    with threadpool_limits(limits=1):
        items, arrays = study.build_tables(records, cfg)
        first = study.predict(out, p, items, arrays, parent_models={})
        second = study.predict(out, p, items, arrays, replay=True, parent_models={})
    assert first == second
    assert set(first) == {'e4', 'full_512', 'structure_512'}
    assert len(list((out/'models').glob('*.joblib'))) == 2
    entries = [r['entry'] for r in records if r['split']=='validation']
    study.persist_predictions(out, first, entries, False)
    assert study.persist_predictions(out, first, entries, True) == first
