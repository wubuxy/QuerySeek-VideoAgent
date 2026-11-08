import json
from copy import deepcopy
from types import SimpleNamespace
import pytest

from queryseek.retrieval import grounded_reader as fix
from scripts import probe_grounded_reader_fix as p


def request():
    return {'question': 'Which object is shown after the event?', 'choices': ['first object', 'second object']}


def test_visual_read_retains_choices_and_marks_saturated_note_as_unverified():
    prompt, schema = fix.visual_remap_prompt(request(), {'fact': 'x'*320, 'evidence': [1]}, 4)
    assert all(x in prompt for x in request()['choices'])
    assert 'ORIGINAL IMAGES' in prompt and '"may_be_incomplete": true' in prompt
    assert set(schema['properties']) == {'answer', 'evidence'}
    blind, _ = fix.cited_prompt(request(), 2)
    assert 'x'*320 not in blind and 'UNVERIFIED_NOTE' not in blind


def test_citation_check_binds_local_indices_to_actual_views():
    proposal = {'answer': 'A', 'evidence': [5, 1]}
    subset, mapping = fix.cited_views(list(range(8)), proposal)
    assert subset == [5, 1] and mapping == [5, 1]
    final = fix.bind_citation_result(proposal, {'answer': 'A', 'evidence': [1]}, mapping)
    assert final['evidence'] == [1] and final['citation_status'] == 'confirmed'
    assert fix.bind_citation_result(proposal, {'answer': 'B', 'evidence': [0]}, mapping)['answer'] == 'abstain'
    assert fix.bind_citation_result(proposal, {'answer': 'abstain', 'evidence': []}, mapping)['evidence'] == []
    with pytest.raises(ValueError):
        fix.bind_citation_result(proposal, {'answer': 'A', 'evidence': [0]}, [1, 5])


@pytest.mark.parametrize('refs', [[2], [True], [-1], [0, 0], []])
def test_illegal_or_empty_citation_never_confirms_answer(refs):
    with pytest.raises(ValueError):
        fix.bind_citation_result({'answer': 'A', 'evidence': [5, 1]}, {'answer': 'A', 'evidence': refs}, [5, 1])


def inputs(demo, tmp_path):
    out = tmp_path/'outputs'; out.mkdir(exist_ok=True)
    frames = p.s.study.extract_scene_frames(demo['video'], out/'images',
        [p.s.Scene(scene_id=str(i), start_sec=t-.001, end_sec=t+.001) for i, t in enumerate([6., 18.])], max_side=320)
    row = {'request': {**request(), 'qid': 'x', 'video_id': 'x', 'video_path': str(demo['video']), 'duration_sec': 60.},
           'frames': [f.model_dump() for f in frames], 'frame_sha256': [p.s.sha(f.image_path) for f in frames]}
    observation = {'fact': 'A fallible description.', 'evidence': [0]}
    fact = {'step': 'fact', 'raw': json.dumps(observation), 'model_call': {'status': 'mock', 'generation_attempted': False},
            'image_sha256': row['frame_sha256'], 'source_times_sec': [f.timestamp_sec for f in frames], 'reused': False}
    prior = {'group': 'development', 'qid': 'x', 'input': 'matched', 'donor_qid': 'x', 'method': 'fact_map',
             'observation': observation, 'calls': [fact], 'status': 'ok', 'answer': 'A', 'evidence': [0]}
    return out, row, prior


class Spy:
    def __init__(self, replies): self.replies, self.seen = iter(replies), []
    def generate(self, prompt, frames, schema, pixels, constrained):
        self.seen.append((prompt, list(frames)))
        value = next(self.replies)
        return value, {'status': 'complete', 'generation_attempted': True, 'images': len(frames)}


def test_repair_keeps_images_then_checks_only_cited_images_without_answer(demo, tmp_path):
    out, row, prior = inputs(demo, tmp_path)
    backend = Spy(['{"answer":"B","evidence":[1]}', '{"answer":"B","evidence":[0]}'])
    engine = p.Engine({'pixels': 262144, 'max_generations': 2}, backend, out, False)
    visual, final = p.repair(row, row, prior, engine)
    assert len(backend.seen[0][1]) == 2 and len(backend.seen[1][1]) == 1
    assert backend.seen[1][1][0].image_path == backend.seen[0][1][1].image_path
    assert 'fallible description' not in backend.seen[1][0]
    assert visual['answer'] == final['answer'] == 'B' and final['evidence'] == [1]
    assert final['evidence_source_times_sec'] == [row['frames'][1]['timestamp_sec']]
    assert len(engine.calls) == 2 and len(final['calls']) == 3 and final['calls'][0]['reused']
    with pytest.raises(RuntimeError): engine.call('', {}, [], 'compact', {})


@pytest.mark.parametrize('first,second,status', [
    ('{"answer":"abstain","evidence":[]}', None, 'ok'),
    ('{"answer":"A","evidence":[0]}', '{"answer":"B","evidence":[0]}', 'ok'),
    ('{"answer":"A","evidence":[0]}', 'invalid', 'error'),
    ('invalid', None, 'error'),
])
def test_failed_citations_do_not_fall_back_to_unverified_answer(demo, tmp_path, first, second, status):
    out, row, prior = inputs(demo, tmp_path); backend = Spy([first, second])
    engine = p.Engine({'pixels': 262144, 'max_generations': 2}, backend, out, False)
    visual, final = p.repair(row, row, prior, engine)
    assert final['status'] == status and final['answer'] == 'abstain' and not final['evidence']
    assert len(engine.calls) == (1 if second is None else 2)
    assert len(final['calls']) == len(engine.calls)+1


def test_changed_observation_source_stops_before_generation(demo, tmp_path):
    out, row, prior = inputs(demo, tmp_path); prior['calls'][0]['image_sha256'] = ['bad']
    engine = p.Engine({'max_generations': 2}, None, out, True)
    a, b = p.repair(row, row, prior, engine)
    assert a['status'] == b['status'] == 'error' and not engine.calls


def test_real_parser_accepts_repaired_and_abstained_outputs():
    lm = pytest.importorskip('lmformatenforcer')
    _, schema = fix.cited_prompt(request(), 2)
    for raw in ['{"answer":"B","evidence":[1]}', '{"answer":"abstain","evidence":[]}']:
        parser = lm.JsonSchemaParser(schema)
        for ch in raw:
            assert ch in parser.get_allowed_characters(); parser = parser.add_character(ch)
        assert parser.can_end()


def test_full_mock_regression_uses_all_sources_and_reuses_history(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(p.s.study.baseline, 'ROOT', tmp_path)
    out, row, prior = inputs(demo, tmp_path)
    groups = {g: [] for g in ['development', 'new_sources']}; history = []
    for group in groups:
        for i in range(2):
            r = deepcopy(row); q = group+str(i); r['request'].update(qid=q, video_id=q); groups[group].append(r)
        for kind in ['matched', 'mismatched']:
            for i, r in enumerate(groups[group]):
                donor = groups[group][1-i] if kind == 'mismatched' else r
                for method in ['minimal', 'compact', 'fact_map']:
                    old = deepcopy(prior); old.update(group=group, input=kind, method=method, qid=r['request']['qid'], donor_qid=donor['request']['qid'])
                    history.append(old)
    pc = out/'parent_config.json'; p.s.write(pc, {'mock': True})
    pr = p.parent.begin(out/'parent_run', pc, True)
    p.s.write(pr/'prepared.json', groups)
    for r in history: p.s.append(pr/'records.jsonl', r)
    p.s.finish(pr, ['prepared.json', 'records.jsonl'], {'mock': True})
    labels = {}
    for group, rows in groups.items():
        labels[group] = out/(group+'_labels.json')
        p.s.write(labels[group], [{'qid': r['request']['qid'], 'right_answer': 'A'} for r in rows])
    config = out/'config.json'; p.s.write(config, {'parent_config_sha256': p.s.sha(pc), 'sample_count': 2,
        'frames': 2, 'pixels': 262144, 'max_generations': 16, 'dev_labels_sha256': p.s.sha(labels['development']),
        'fresh_labels_sha256': p.s.sha(labels['new_sources'])})
    a = SimpleNamespace(config=config, parent_config=pc, parent_run=pr, output=out/'run', mock=True)
    assert p.run(a)['records'] == 16
    a.runs=a.output; a.output=out/'eval'; a.dev_labels=labels['development']; a.labels=labels['new_sources']
    assert p.evaluate(a)['new_calls'] == 16
    e = p.s.read(a.output/'evaluation.json')
    assert all(v['n'] == 2 for g in e['groups'].values() for k in g.values() for v in k.values())
    assert e['new_physical_cost']['generation_attempted'] == 0
    with pytest.raises(ValueError): p.verify(a.runs, config, False)
