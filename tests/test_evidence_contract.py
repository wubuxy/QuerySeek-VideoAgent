import json
from itertools import permutations
import pytest
from queryseek.retrieval.evidence_contract import strict_prompt,decode_verdict


def inputs():
    request={'question':'Which car passes first?','choices':['red','blue'],'duration_sec':100.}
    packet={'candidate_id':'c0','interval_sec':[10.,20.],
        'frames':[{'timestamp_sec':11.+i} for i in range(4)],'frame_sha256':['x']*4}
    return request,packet


@pytest.mark.parametrize('decision,evidence,state',[
    ('supported:B','0,3','supported'),('contradicted','2','contradicted'),
    ('insufficient:global_scope','','insufficient')])
def test_atomic_decisions_map_to_valid_canonical_state(decision,evidence,state):
    req,p=inputs();v=decode_verdict({'decision':decision,'evidence':evidence,'observed_fact':'Visible car.'},p,req['choices'],100.)
    assert v['state']==state
    if state=='supported':assert v['evidence_source_times_sec']==[11.,14.]


@pytest.mark.parametrize('decision,evidence',[
    ('supported:A','0'),('supported:A','1,1'),('supported:A','3,1'),('supported:A','0,9'),
    ('contradicted',''),('insufficient:none',''),('insufficient:visual_detail','0')])
def test_illegal_pairs_are_rejected(decision,evidence):
    req,p=inputs()
    with pytest.raises(ValueError):decode_verdict({'decision':decision,'evidence':evidence,'observed_fact':'fact'},p,req['choices'],100.)


def test_real_json_parser_enforces_cross_field_constraints_in_every_field_order():
    lm=pytest.importorskip('lmformatenforcer')
    req,p=inputs();_,schema=strict_prompt(req,p)
    cases=[('supported:A','0,1',True),('supported:A','0',False),('supported:A','1,1',False),
           ('contradicted','0',True),('contradicted','',False),('insufficient:global_scope','',True),
           ('insufficient:global_scope','0,1',False)]
    for decision,evidence,expected in cases:
        value={'decision':decision,'evidence':evidence,'observed_fact':'fact'}
        for order in permutations(value):
            parser=lm.JsonSchemaParser(schema);accepted=True
            for char in json.dumps({k:value[k] for k in order}):
                if char not in parser.get_allowed_characters():accepted=False;break
                parser=parser.add_character(char)
            assert (accepted and parser.can_end())==expected,(decision,evidence,order)


def test_strict_pipeline_reuses_controls_and_executes_bounded_observation(demo,tmp_path,monkeypatch):
    from queryseek.agent import tree_observations as cache
    from scripts import probe_evidence_contract as strict
    p=strict.p;monkeypatch.setattr(cache,'ROOT',tmp_path)
    request={'qid':'x','video_id':'sample','video_path':str(demo['video']),'duration_sec':60.,
             'question':'Find architecture','choices':['a','b']}
    requests,retrieval,base,config=(tmp_path/n for n in ['requests.json','retrieval.jsonl','base.json','strict.json'])
    p.write_json(requests,[request]);p.append(retrieval,{'qid':'x','method':'dense_uniform','status':'ok',
        'windows':[{'start_sec':10.,'end_sec':20.}],
        'observed_source_times_sec':[5.,15.,40.,55.],'scores':[.1,.8,.9,.2],
        'review_source_times_sec':[5.,15.,40.,55.]})
    cfg=p.read(p.ROOT/'configs/evidence_packets_v1.json')
    cfg.update(sample_count=1,max_generations=11,requests_sha256=p.sha256_file(requests),
               retrieval_sha256=p.sha256_file(retrieval),max_side=96,frames_per_candidate=4,extra_frames=8)
    p.write_json(base,cfg);prepared=tmp_path/'outputs/old_prepared';run=tmp_path/'outputs/old_run'
    p.prepare(requests,retrieval,base,prepared,True);p.run(prepared,base,run,mock=True)
    cfg.update(base_config_sha256=p.sha256_file(base),prepared_sha256=p.sha256_file(prepared/'prepared.json'),
               old_records_sha256=p.sha256_file(run/'records.jsonl'),max_new_verifications=3)
    p.write_json(config,cfg);out=tmp_path/'outputs/strict'
    result=strict.run(prepared,run,base,config,out,mock=True)
    assert result['backend_calls']==0 and result['refinement_attempts']==1
    row=p.lines(out/'run/records.jsonl')[0]
    assert row['joint_forward']['reused_control'] and row['joint_forward']['model_calls']==[]
    assert all(v['status']=='ok' for v in row['verdicts'])
    assert row['refinement']['verdict']['status']=='ok'
    assert row['final_action']['reason']=='observation_budget_exhausted'
