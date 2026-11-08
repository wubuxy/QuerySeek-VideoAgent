import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
from queryseek.eval import compact_reader as c


def request():return {'question':'What color is the requested object?','choices':['red','blue']}


def test_source_selection_excludes_used_large_and_short_without_question_access():
    meta=[{'qid':str(i),'video_id':str(i),'duration':130.,'choice_count':4} for i in range(7)]
    media={str(i):{'bytes':100} for i in range(7)}
    media['1']['bytes']=385*2**20;meta[2]['duration']=119.;meta[3]['choice_count']=9
    a=c.select_sources(meta,media,{'0'},3)
    assert {r['video_id'] for r in a}=={'4','5','6'} and a==c.select_sources(meta[::-1],media,{'0'},3)
    assert all(set(r)=={'qid','video_id','duration'} for r in a)
    with pytest.raises(ValueError):c.select_sources(meta,media,{'0'},4)


def test_fact_step_has_no_separate_choices_and_explicit_output_contract():
    r=request();r['choices']=['hidden-first','hidden-second'];prompt,schema=c.fact_prompt(r,4)
    assert all(x not in prompt for x in r['choices']) and '<OUTPUT_CONTRACT>' in prompt
    assert set(schema['properties'])=={'fact','evidence'}
    mapped,_=c.map_prompt(r,{'fact':'An observed blue object.','evidence':[1]})
    assert 'hidden-second' in mapped and '<OBSERVATION_DATA>' in mapped


@pytest.mark.parametrize('evidence',[[],[True],[4],[1,1],['1'],[0,1,2,3,4]])
def test_compact_answer_requires_valid_nonempty_distinct_indices(evidence):
    with pytest.raises(ValueError):c.decode_compact(json.dumps({'answer':'A','evidence':evidence}),request(),4)


def test_abstention_and_fact_mapping_do_not_bypass_evidence():
    assert c.decode_compact('{"answer":"abstain","evidence":[]}',request(),4)['answer']=='abstain'
    with pytest.raises(ValueError):c.decode_compact('{"answer":"abstain","evidence":[0]}',request(),4)
    observation=c.decode_fact('{"fact":"Not established.","evidence":[]}',4)
    with pytest.raises(ValueError):c.decode_map('{"answer":"A"}',request(),observation)
    assert c.decode_map('{"answer":"abstain"}',request(),observation)['evidence']==[]


@pytest.mark.parametrize('raw',['```json\n{"answer":"A","evidence":[0]}\n```','{"answer":"A","evidence":[0],"extra":1}',
    '{"answer":"Z","evidence":[0]}','[]'])
def test_no_repair_unknown_fields_or_choices(raw):
    with pytest.raises(ValueError):c.decode_compact(raw,request(),4)


def test_agreement_preserves_invalid_failures_and_disagreement_abstains():
    a={'status':'ok','answer':'A','evidence':[1]};b={**a,'evidence':[2]}
    assert c.agreement(a,b)['evidence']==[1,2]
    assert c.agreement(a,{**b,'answer':'B'})['answer']=='abstain'
    assert c.agreement(a,{**b,'status':'error'})['status']=='error'
    m=c.metrics([{'qid':'x',**a},{'qid':'y','status':'error','answer':'abstain'}],['x','y'],{'x':'B','y':'A'})
    assert m=={'n':2,'valid':1,'accepted':1,'correct':0,'wrong_accepted':1,'accuracy_on_accepted':0.}
    with pytest.raises(ValueError):c.metrics([{'qid':'x',**a}]*2,['x','y'])


def test_real_compact_parser_has_both_answer_and_abstention_paths():
    lm=pytest.importorskip('lmformatenforcer');_,schema=c.compact_prompt(request(),4)
    for value in [{'answer':'B','evidence':[2]},{'evidence':[],'answer':'abstain'}]:
        parser=lm.JsonSchemaParser(schema)
        for char in json.dumps(value):
            assert char in parser.get_allowed_characters();parser=parser.add_character(char)
        assert parser.can_end()


def test_synthetic_full_matrix_reuse_and_source_separation(demo,tmp_path,monkeypatch):
    from scripts import probe_compact_reader as p
    s=p.s;monkeypatch.setattr(s.study.baseline,'ROOT',tmp_path);monkeypatch.setattr(s.study,'ROOT',tmp_path)
    out=tmp_path/'outputs';out.mkdir()
    frames=s.study.extract_scene_frames(demo['video'],out/'images',
        [s.Scene(scene_id=str(i),start_sec=t-.001,end_sec=t+.001) for i,t in enumerate([6.,18.])],max_side=320)
    rows=[]
    for q in ['x','y']:
        rows.append({'request':{**request(),'qid':q,'video_id':q,'video_path':str(demo['video']),'duration_sec':60.},
            'frames':[f.model_dump() for f in frames],'frame_sha256':[s.sha(f.image_path) for f in frames],
            'plan':{'target':'object','relation':'color','scope':'single_view'}})
    pc=out/'parent_config.json';s.write(pc,{'frames':2,'conditions':[{'name':'bf16_high_json','precision':'bf16',
        'pixels':262144,'constrained':True,'input':'oracle','prompt':'minimal'}]})
    pp=s.begin(out/'parent_prepared',pc,True);s.write(pp/'prepared.json',{'rows':rows,'controls':[]});s.finish(pp,['prepared.json'],{'mock':True})
    parent_run=out/'parent_run';s.run(SimpleNamespace(config=pc,prepared=pp,output=parent_run,mock=True,precision='bf16'))
    lc=out/'legacy_config.json';s.write(lc,{'parent_config_sha256':s.sha(pc),'sample_count':2,'frames':2,
        'conditions':[{'name':'bf16_needs_json','precision':'bf16','constrained':True}]})
    lr=out/'legacy_run';p.legacy.run(SimpleNamespace(config=lc,parent_config=pc,prepared=pp,output=lr,mock=True,precision='bf16'))
    fresh=[{**r['request'],'qid':'new_'+r['request']['qid'],'video_id':'new_'+r['request']['video_id']} for r in rows]
    reqs=out/'requests.json';fl=out/'fresh_labels.json';dl=out/'dev_labels.json'
    s.write(reqs,fresh);s.write(fl,[{'qid':r['qid'],'right_answer':'A','gt_intervals_sec':[[5.,8.]]} for r in fresh])
    s.write(dl,[{'qid':r['request']['qid'],'right_answer':'A'} for r in rows])
    cfg=out/'config.json';s.write(cfg,{'parent_config_sha256':s.sha(pc),'legacy_config_sha256':s.sha(lc),
        'fresh_requests_sha256':s.sha(reqs),'fresh_labels_sha256':s.sha(fl),'dev_labels_sha256':s.sha(dl),
        'sample_count':2,'frames':2,'pixels':262144,'max_generations':32})
    a=SimpleNamespace(config=cfg,parent_config=pc,legacy_config=lc,parent_prepared=pp,parent_run=parent_run,
        legacy_run=lr,requests=reqs,labels=fl,output=out/'prepared',mock=True)
    assert p.prepare(a)['fresh_preparation_errors']==0
    a.prepared=a.output;a.output=out/'run';p.run(a)
    a.runs=a.output;a.output=out/'eval';a.dev_labels=dl;p.evaluate(a)
    result=s.read(a.output/'evaluation.json');assert result['mock'] and result['new_physical_cost']['generation_attempted']==0
    for group in result['groups'].values():
        for kind in ['matched','mismatched']:
            assert set(group[kind])==set(c.METHODS) and all(x['n']==2 for x in group[kind].values())
    records=s.previous.lines(a.runs/'records.jsonl');calls=s.previous.lines(a.runs/'calls.jsonl')
    assert len(records)==26 and len(calls)==32
    assert sum(r['calls'][0]['reused'] for r in records)==2
    assert all(r['donor_qid']!=r['qid'] for r in records if r['input']=='mismatched')
    assert all(x['image_sha256']==[] and x['source_times_sec']==[] for x in calls if x['step']=='map')
    with pytest.raises(ValueError):p.verify(a.prepared,cfg,False)
    bad=deepcopy(rows[0]);bad['preparation_error']='test';engine=p.Engine({},None,out,True)
    assert p.task(bad,rows[1],'compact','matched','development',engine)['status']=='error' and not engine.calls
