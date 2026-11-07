import json
from collections import Counter
import pytest
from queryseek.agent import tree_observations as cache
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.evidence_packets import (propose_packets,packet_frames_valid,joint_view,
    verification_prompt,validate_verdict,aggregate,next_action)
from scripts import probe_evidence_packets as probe


def packet(cid='c0',a=0.,b=10.):
    return {'candidate_id':cid,'interval_sec':[a,b],'origin':'retrieval','seed_time_sec':None,
        'frames':[{'frame_id':cid+str(i),'timestamp_sec':a+(i+.5)*(b-a)/4,
                   'image_path':cid+str(i)+'.jpg','source':'scene'} for i in range(4)],
        'frame_sha256':['digest']*4}


def verdict(state='insufficient',answer='abstain',refs=None,missing='visual_detail'):
    return {'state':state,'answer':answer,'evidence_frames':[] if refs is None else refs,
            'missing_reason':missing,'observed_fact':'A visible object.'}


def checked(p,state='insufficient',answer='abstain',refs=None,missing='visual_detail'):
    return {'candidate_id':p['candidate_id'],'status':'ok',
            **validate_verdict(verdict(state,answer,refs,missing),p,['a','b'],100.)}


def test_uncovered_seed_is_a_search_hypothesis_with_clamped_nonzero_interval():
    r={'status':'ok','windows':[{'start_sec':10.,'end_sec':20.}],
       'observed_source_times_sec':[0.,15.,99.],'scores':[.8,.7,.9],
       'review_source_times_sec':[0.,15.,99.]}
    packets,remaining=propose_packets(r,100.,2,64.)
    assert [p['interval_sec'] for p in packets]==[[10.,20.],[36.,100.],[0.,64.]]
    assert remaining==[] and packets[1]['seed_time_sec']==99.
    assert 'frames' not in packets[1]
    packets,remaining=propose_packets(r,100.,1,64.)
    assert remaining==[0.]
    with pytest.raises(ValueError):propose_packets({**r,'review_source_times_sec':[3.]},100.)


def test_joint_order_changes_preserve_image_multiset_and_every_frame_is_referencable():
    packets=[packet(),packet('c1',20.,30.)]
    f,c=joint_view(packets,100.);rf,rc=joint_view(packets,100.,True)
    assert Counter(x['image_path'] for x in f)==Counter(x['image_path'] for x in rf)
    assert [x['candidate_id'] for x in rc]==['c1','c0']
    for frames,candidates in [(f,c),(rf,rc)]:
        assert sorted(i for x in candidates for i in x['frame_indices'])==list(range(8))
        assert all(x['interval_sec'][0]<=frames[i]['timestamp_sec']<x['interval_sec'][1]
                   for x in candidates for i in x['frame_indices'])


@pytest.mark.parametrize('bad',[
    verdict('supported','A',[0],'none'),verdict('supported','A',[0,0],'none'),
    verdict('supported','A',[0,9],'none'),verdict('contradicted','abstain',[],'none'),
    verdict('insufficient','A',[],'visual_detail'),verdict('insufficient','abstain',[],'none')])
def test_invalid_or_unsupported_claims_cannot_be_accepted(bad):
    with pytest.raises(ValueError):validate_verdict(bad,packet(),['a','b'],100.)


def test_half_open_frame_contract_rejects_frame_at_end():
    p=packet();p['frames'][-1]['timestamp_sec']=10.
    with pytest.raises(ValueError):packet_frames_valid(p,100.)


def test_controller_observes_unknown_but_not_contradiction_and_obeys_budget():
    ps=[packet(),packet('c1',20.,30.)]
    vs=[checked(ps[0],'contradicted','abstain',[0],'none'),checked(ps[1])]
    assert next_action(ps,vs,1)=={'action':'observe_more','candidate_id':'c1'}
    assert next_action(ps,vs,0)['reason']=='observation_budget_exhausted'
    vs[1]=checked(ps[1],missing='global_scope')
    assert next_action(ps,vs,1)['reason']=='no_locally_resolvable_candidate'


def test_agreement_uses_frozen_rank_and_conflict_abstains():
    ps=[packet(),packet('c1',20.,30.)]
    vs=[checked(p,'supported','A',[0,1],'none') for p in ps]
    assert aggregate(ps,list(reversed(vs)))['candidate_id']=='c0'
    vs[1]=checked(ps[1],'supported','B',[0,1],'none')
    assert aggregate(ps,vs)['reason']=='conflicting_supported_answers'
    assert next_action(ps,vs,1)['action']=='abstain'
    with pytest.raises(ValueError):aggregate(ps,vs[:1])


def test_prompt_preserves_question_but_excludes_evaluator_fields():
    prompt,schema=verification_prompt({'question':'First car after the crossing?',
        'choices':['red','blue'],'duration_sec':100.,'right_answer':'SECRET','gt_intervals_sec':[[2,3]]},packet())
    assert 'First car after the crossing?' in prompt and 'global_scope' in prompt
    assert 'SECRET' not in prompt and 'gt_intervals' not in prompt
    assert schema['properties']['evidence_frames']['items']['enum']==[0,1,2,3]


def test_full_mock_observation_refinement_and_immutable_input_identity(demo,tmp_path,monkeypatch):
    monkeypatch.setattr(cache,'ROOT',tmp_path)
    request={'qid':'x','video_id':'sample','video_path':str(demo['video']),'duration_sec':60.,
             'question':'Find architecture','choices':['a','b']}
    requests,retrieval,config=(tmp_path/n for n in ['requests.json','retrieval.jsonl','config.json'])
    cache.write_json(requests,[request])
    probe.append(retrieval,{'qid':'x','method':'dense_uniform','status':'ok',
        'windows':[{'start_sec':10.,'end_sec':20.}],
        'observed_source_times_sec':[5.,15.,40.,55.],'scores':[.1,.8,.9,.2],
        'review_source_times_sec':[5.,15.,40.,55.]})
    cfg=probe.read(probe.ROOT/'configs/evidence_packets_v1.json')
    cfg.update(sample_count=1,max_generations=11,requests_sha256=sha256_file(requests),
               retrieval_sha256=sha256_file(retrieval),max_side=96,frames_per_candidate=4,extra_frames=8)
    cache.write_json(config,cfg)
    prepared,run=tmp_path/'outputs/prepared',tmp_path/'outputs/run'
    result=probe.prepare(requests,retrieval,config,prepared,True)
    assert result['queries']==1 and result['candidates']==2 and result['new_frames_extracted']==8
    result=probe.run(prepared,config,run,mock=True)
    assert result['refinement_attempts']==1 and result['backend_calls']==0
    record=probe.lines(run/'records.jsonl')[0]
    assert record['action']['action']=='observe_more' and record['final_action']['action']=='abstain'
    assert len(record['refinement']['packet']['frames'])==12
    assert record['packet_observe']['abstained']
    with pytest.raises(ValueError,match='provenance/mock'):probe.verify(run,config,False)
    with pytest.raises(FileExistsError):probe.run(prepared,config,run,mock=True)
    rows=probe.read(prepared/'prepared.json');rows[0]['packets'][0]['frames'][0]['timestamp_sec']+=1
    cache.write_json(prepared/'prepared.json',rows)
    with pytest.raises(ValueError,match='Artifact'):probe.verify(prepared,config,True)
