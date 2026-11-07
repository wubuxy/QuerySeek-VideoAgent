import json
import pytest
from scripts.probe_verifier_diagnostic import decode,metrics


def row():
    return {'request':{'question':'What color?','choices':['red','blue'],'duration_sec':60.},
        'plan':{'target':'object','relation':'color','scope':'single_view'},
        'frames':[{'timestamp_sec':1.},{'timestamp_sec':2.}], 'frame_sha256':['a','b']}


def raw(**changes):
    return json.dumps({'decision':'answer:A','evidence':'0','region':'full',
        'target_description':'Visible object','observed_fact':'The object is red',**changes})


def test_raw_intent_does_not_bypass_contract():
    good=decode(raw(),row());assert good['status']=='ok' and good['answer']=='A'
    bad=decode(raw(observed_fact='x'*241),row())
    assert bad['status']=='error' and bad['answer']=='abstain' and bad['raw_answer']=='A'
    m=metrics([{'qid':'x',**bad}],{'x':'A'})
    assert m['correct']==0 and m['raw_decision_correct']==1 and m['valid']==0


@pytest.mark.parametrize('text',['```json\n{}\n```','[]','not json','{"decision":"answer:Z"}'])
def test_no_output_repair_or_unknown_answer(text):
    result=decode(text,row())
    assert result['status']=='error' and result['answer']==result['raw_answer']=='abstain'


def test_missing_and_duplicate_queries_remain_errors():
    result={'qid':'x',**decode(raw(),row())}
    with pytest.raises(ValueError):metrics([result],{'x':'A','y':'B'})
    with pytest.raises(ValueError):metrics([result,result],{'x':'A','y':'B'})


def test_frozen_parent_and_two_precision_mock_pipeline(demo,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import probe_verifier_diagnostic as v
    s=v.s;monkeypatch.setattr(s.study.baseline,'ROOT',tmp_path)
    out=tmp_path/'outputs';out.mkdir()
    frames=s.study.extract_scene_frames(demo['video'],out/'images',
        [s.Scene(scene_id=str(i),start_sec=t-.001,end_sec=t+.001) for i,t in enumerate([6.,18.])],max_side=320)
    rows=[]
    for q in ['x','y']:
        r=row();r['request'].update(qid=q,video_id=q,video_path=str(demo['video']))
        r['frames']=[f.model_dump() for f in frames];r['frame_sha256']=[s.sha(f.image_path) for f in frames];rows.append(r)
    parent_config=out/'parent_config.json';s.write(parent_config,{'frames':2,'conditions':[
        {'name':'nf4_high_needs','precision':'nf4','prompt':'needs','input':'oracle','pixels':262144,'constrained':True}]})
    prepared=s.begin(out/'parent_prepared',parent_config,True);s.write(prepared/'prepared.json',{'rows':rows,'controls':[]})
    s.finish(prepared,['prepared.json'],{'mock':True})
    parent_args=SimpleNamespace(config=parent_config,prepared=prepared,output=out/'parent_run',mock=True,precision='nf4')
    s.run(parent_args)
    labels=out/'labels.json';s.write(labels,[{'qid':q,'right_answer':'A'} for q in ['x','y']])
    config=out/'config.json';s.write(config,{'parent_config_sha256':s.sha(parent_config),'sample_count':2,
        'frames':2,'max_generations':4,'labels_sha256':s.sha(labels),'conditions':[
        {'name':'nf4_needs_free','precision':'nf4','constrained':False},
        {'name':'bf16_needs_json','precision':'bf16','constrained':True}]})
    args=SimpleNamespace(config=config,parent_config=parent_config,prepared=prepared,mock=True)
    for precision in ['nf4','bf16']:
        args.precision=precision;args.output=out/'runs'/precision
        assert v.run(args)['new_model_calls']==0
    args.parent_run=out/'parent_run';args.runs=out/'runs';args.labels=labels;args.output=out/'evaluation'
    v.evaluate(args);result=s.read(args.output/'evaluation.json')
    assert result['mock'] and result['new_model_calls']==0
    assert all(m['n']==2 and m['answered']==0 for m in result['metrics'].values())
    with pytest.raises(ValueError):v.verify(out/'runs/nf4',config,False)
