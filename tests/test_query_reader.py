import json
import numpy as np
import pytest

from queryseek.agent import tree_observations as cache
from queryseek.agent.temporal_tree import NodeObservation
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.query_reader import (METHODS, bind_answer, candidates_for, letter_token_ids,
    planner_prompt, query_scores, reader_prompt, token_chunks, validate_plan)
from queryseek.schemas import TimeWindow, VideoFrame
from scripts import probe_query_reader as probe


class Tokenizer:
    context_length = 77
    def encode(self, text, **kwargs):
        return text.split()
    def decode(self, ids):
        return ' '.join(ids)


def test_chunks_preserve_all_content_and_reject_silent_overflow():
    text = ' '.join(f'w{i}' for i in range(125))
    chunks = token_chunks(text, Tokenizer())
    assert list(map(lambda s:len(s.split()), chunks)) == [60,60,5]
    assert ' '.join(chunks) == text
    assert token_chunks('short question', Tokenizer()) == ['short question']
    with pytest.raises(ValueError):token_chunks('x '*181, Tokenizer())


def test_planner_sees_question_only_and_validates_limits():
    prompt, schema = planner_prompt('What happens after the door opens?')
    assert json.loads(prompt.split('\n')[-1]) == {'question':'What happens after the door opens?'}
    plan = {'queries':['person opens door'], 'temporal_requirement':'before_after','answer_focus':'next action'}
    assert validate_plan(plan) == plan
    for bad in [{**plan,'queries':['x'*121]}, {**plan,'queries':['same','same']},
                {**plan,'right_answer':'A'}, {**plan,'temporal_requirement':'guessed'}]:
        with pytest.raises(ValueError):validate_plan(bad)


def test_max_pool_is_or_not_order_or_conjunction():
    frames = [VideoFrame(frame_id=str(i),timestamp_sec=float(i),image_path='unused',source='uniform') for i in range(2)]
    obs = NodeObservation(frames,np.eye(2,dtype=np.float32))
    assert query_scores(obs,np.eye(2)).tolist() == [1.,1.]
    with pytest.raises(ValueError):query_scores(obs,[])


def candidate_fixture():
    frames = [VideoFrame(frame_id='a',timestamp_sec=10.,image_path='unused',source='uniform')]
    windows = [TimeWindow(start_sec=0.,end_sec=10.),TimeWindow(start_sec=10.,end_sec=20.)]
    return candidates_for(windows,frames,30.)


def test_candidate_id_uses_only_displayed_frames_half_open_and_keeps_original_index():
    candidates = candidate_fixture()
    assert candidates == [{'candidate_id':'c1','interval_sec':[10.,20.],'displayed_source_times_sec':[10.]}]
    answer = bind_answer({'answer':'B','candidate_id':'c1'},candidates,['one','two'],30.)
    assert answer['intervals'] == [[10.,20.]] and not answer['abstained']
    assert bind_answer({'answer':'abstain','candidate_id':'none'},[],['one','two'],30.)['abstained']


@pytest.mark.parametrize('prediction',[
    {'answer':'A','candidate_id':'c0'}, {'answer':'abstain','candidate_id':'c1'},
    {'answer':'B','candidate_id':'none'}, {'answer':'A','candidate_id':'c1','intervals':[[0,30]]}])
def test_bad_binding_does_not_silently_answer(prediction):
    with pytest.raises(ValueError):bind_answer(prediction,candidate_fixture(),['one','two'],30.)


def test_reader_interface_and_baseline_prompt_parity():
    request = {'question':'What color?','choices':['red','blue']}
    original, schema = probe.study.prompt_and_schema(request,{'output':'answer'})
    assert reader_prompt(request,'oracle_list_json')[:2] == (original,schema)
    prompt, schema, letter = reader_prompt(request,'oracle_labeled_letter')
    assert 'A. red\nB. blue' in prompt and letter
    prompt, schema, letter = reader_prompt(request,'raw',candidate_fixture())
    assert schema['properties']['candidate_id']['enum'] == ['c1','none'] and not letter
    assert 'sparse' in prompt and 'abstain' in schema['properties']['answer']['enum']


def test_letter_decoder_rejects_multi_token_or_colliding_choices():
    assert letter_token_ids(Tokenizer(),['A','B']) == ['A','B']
    with pytest.raises(ValueError):letter_token_ids(Tokenizer(),['A B','C'])
    with pytest.raises(ValueError):letter_token_ids(Tokenizer(),['A','A'])


def test_abstention_is_valid_but_not_success_and_failure_stays_in_denominator():
    label = {'right_answer':'A','gt_intervals_sec':[[10.,20.]]}
    assert probe.score_bound({'status':'ok','abstained':True},label,30.) == {
        'valid':1,'accepted':0,'correct':0,'iou':0.,'joint_at_05':0}
    result = probe.score_bound({'status':'ok','abstained':False,'answer':'A','intervals':[[10.,20.]]},label,30.)
    assert result['correct'] == result['joint_at_05'] == 1
    assert probe.score_bound({'status':'error'},label,30.)['valid'] == 0
    with pytest.raises(ValueError):probe.matrix([{'qid':'x','condition':'a'}],['x','y'],'condition',['a'])


def test_mock_full_flow_hash_binding_and_no_oracle_label_leakage(demo,tmp_path,monkeypatch):
    monkeypatch.setattr(cache,'ROOT',tmp_path)
    monkeypatch.setattr(probe.study,'ROOT',tmp_path)
    monkeypatch.setattr(probe.study.baseline,'ROOT',tmp_path)
    request = {'qid':'x','video_id':'sample','video_path':str(demo['video']),
               'duration_sec':60.,'question':'Find the architecture section.','choices':['one','two']}
    requests, oracle, config = (tmp_path/n for n in ['requests.json','oracle.json','config.json'])
    cache.write_json(requests,[request])
    frames,cost = probe.study.oracle_frames(request,[(12.,24.)],4)
    cache.write_json(oracle,[{'request':request,'condition':'oracle_video16',
        'frames':[f.model_dump() for f in frames], 'frame_sha256':[sha256_file(f.image_path) for f in frames],
        'preprocessing':cost}])
    cfg = probe.read(probe.ROOT/'configs/query_reader_v1.json')
    cfg.update(sample_count=1,max_generations=7,review_frames=4,
        requests_sha256=sha256_file(requests),oracle_prepared_sha256=sha256_file(oracle))
    cfg['sampling'].update(max_frames=24,max_side=96)
    cache.write_json(config,cfg)
    plan,prepared,run = (tmp_path/'outputs'/n for n in ['plan','prepared','run'])
    result = probe.plan_oracle(requests,oracle,config,plan,mock=True)
    assert result['plans_ok']==1 and result['oracle_ok']==3 and result['backend_calls']==0
    assert 'gt_intervals' not in (plan/'plans.jsonl').read_text()
    assert probe.prepare(requests,plan,None,config,prepared,mock=True)['completed']==3
    assert probe.answer(requests,prepared,config,run,mock=True)['completed']==3
    assert {r['condition'] for r in probe.lines(run/'predictions.jsonl')} == set(METHODS)
    with pytest.raises(ValueError,match='provenance/mock'):probe.verify_stage(run,config,False)
    with pytest.raises(FileExistsError):probe.answer(requests,prepared,config,run,mock=True)
    with (prepared/'prepared.json').open('a') as handle:handle.write(' ')
    with pytest.raises(ValueError,match='artifact changed'):probe.verify_stage(prepared,config,True)


def test_request_labels_rejected_before_inference(tmp_path):
    path = tmp_path/'requests.json'
    cache.write_json(path,[{'qid':'x','video_id':'s','video_path':'unused','duration_sec':30.,
        'question':'q','choices':['a','b'],'right_answer':'A'}])
    with pytest.raises(ValueError,match='whitelisted'):
        probe.requests_for(path,{'requests_sha256':sha256_file(path),'sample_count':1,'max_generations':7})
