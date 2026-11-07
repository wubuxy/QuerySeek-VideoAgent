from copy import deepcopy
from types import SimpleNamespace
import json
import pytest

from queryseek.eval.model_diagnostic import (conditions, minimal_prompt, parse_answer,
    donor_indices, assert_visual_transport, summarize)


@pytest.mark.parametrize('raw',['A','```json\n{"answer":"A"}\n```','{"answer":"Z"}',
    '{"answer":"A","explanation":"x"}','{"answer":null}','[]'])
def test_free_outputs_are_not_repaired(raw):
    with pytest.raises((ValueError,TypeError)):parse_answer(raw,['x','y'])


def test_same_prompt_for_resolution_and_decoding_and_no_privileged_fields():
    request={'question':'Which visible object?', 'choices':['first','second'],
             'gt_intervals_sec':[[1.,2.]],'right_answer':'B'}
    prompt,schema=minimal_prompt(request)
    assert 'gt_intervals' not in prompt and 'right_answer' not in prompt
    assert parse_answer(' {"answer":"B"} ',request['choices'])=='B'
    assert len(conditions())==8 and sum(c['precision']=='nf4' for c in conditions())==7
    assert schema['properties']['answer']['enum']==['A','B']


def test_wrong_source_control_cannot_use_own_video():
    rows=[{'request':{'video_id':str(i)}} for i in range(3)]
    assert donor_indices(rows)==[1,2,0]
    rows[2]['request']['video_id']='0'
    with pytest.raises(ValueError):donor_indices(rows)


def audit():
    return {'image_grid_thw':[[1,16,16]],'image_token_count':64,'visual_tokens':64,
            'pixel_tensor_sha256':'a'*64,'pixel_tensor_std':1.}


@pytest.mark.parametrize('field,value',[('image_token_count',63),('visual_tokens',32),
    ('pixel_tensor_sha256',None),('pixel_tensor_std',0.),('image_grid_thw',[])])
def test_visual_transport_detects_missing_or_wrong_images(field,value):
    a=audit();a[field]=value
    with pytest.raises(ValueError):assert_visual_transport(a,1)


def test_visual_transport_and_true_text_control():
    assert_visual_transport(audit(),1)
    a={'image_grid_thw':[],'image_token_count':0,'visual_tokens':0,
       'pixel_tensor_sha256':None,'pixel_tensor_std':0.}
    assert_visual_transport(a,0)
    a['pixel_tensor_sha256']='unexpected'
    with pytest.raises(ValueError):assert_visual_transport(a,0)


def records():
    rows=[{'qid':q,'condition':c['name'],'precision':c['precision'],'control':False,
        'status':'ok','answer':'A'} for c in conditions() for q in ['a','b']]
    rows += [{'qid':q,'condition':p+'_control','precision':p,'control':True,
        'status':'ok','answer':a} for p in ['nf4','bf16'] for q,a in [('control_0','B'),('control_1','A')]]
    return rows


def test_full_denominator_keeps_errors_abstentions_and_pair_direction():
    rows=records();rows[0].update(status='error',answer='abstain')
    result=summarize(rows,{'a':'A','b':'B'},[c['name'] for c in conditions()],
                     {'control_0':'B','control_1':'A'})
    assert result['metrics']['nf4_low_json']=={'n':2,'valid':1,'answered':1,'correct':0}
    pair=result['paired']['nf4_low_json__nf4_high_json']
    assert pair=={'n':2,'answer_changed':1,'gained':1,'lost':0}
    for bad in [rows[:-1],rows+[rows[0]],rows[1:]]:
        with pytest.raises(ValueError):summarize(bad,{'a':'A','b':'B'},[c['name'] for c in conditions()],
                                                {'control_0':'B','control_1':'A'})


def test_fixture_mock_full_matrix_image_identity_and_manifest(demo,tmp_path,monkeypatch):
    from scripts import probe_model_diagnostic as s
    monkeypatch.setattr(s.study.baseline,'ROOT',tmp_path)
    output=tmp_path/'outputs';output.mkdir()
    frames=s.study.extract_scene_frames(demo['video'],output/'images',
        [s.Scene(scene_id=str(i),start_sec=t-.001,end_sec=t+.001) for i,t in enumerate([6.,18.,30.,42.])],max_side=320)
    reqs=[dict(qid=str(i),video_id=str(i),question='Which background?',choices=['red','blue'],
              video_path=str(demo['video']),duration_sec=60.) for i in range(2)]
    paths={n:output/(n+'.json') for n in ['requests','oracle','config','plans','labels']}
    s.write(paths['requests'],reqs)
    s.write(paths['oracle'],[{'request':r,'condition':'oracle_video16',
        'frames':[f.model_dump() for f in frames],'frame_sha256':[s.sha(f.image_path) for f in frames]} for r in reqs])
    for r in reqs:s.append(paths['plans'],{'qid':r['qid'],'status':'ok',
        'plan':{'target':'background','relation':'color','scope':'single_view'}})
    s.write(paths['labels'],[{'qid':r['qid'],'right_answer':'A'} for r in reqs])
    cfg={'sample_count':2,'frames':4,'max_generations':20,'conditions':conditions(),
        'requests_sha256':s.sha(paths['requests']),'oracle_prepared_sha256':s.sha(paths['oracle']),
        'plans_sha256':s.sha(paths['plans']),'labels_sha256':s.sha(paths['labels'])}
    s.write(paths['config'],cfg)
    a=SimpleNamespace(config=paths['config'],requests=paths['requests'],oracle_prepared=paths['oracle'],
        plans=paths['plans'],output=output/'prepared',mock=True)
    s.prepare(a);prepared=a.output
    for precision in ['nf4','bf16']:
        a.prepared=prepared;a.precision=precision;a.output=output/'runs'/precision
        summary=s.run(a);assert summary['new_model_calls']==0
    a.runs=output/'runs';a.labels=paths['labels'];a.output=output/'evaluation'
    s.evaluate(a);assert s.read(a.output/'evaluation.json')['mock']
    rows=s.previous.lines(output/'runs/nf4/records.jsonl')
    wrong=[r for r in rows if r['condition']=='nf4_high_wrong']
    assert all(r['qid']!=r['donor_qid'] for r in wrong)
    text=[r for r in rows if r['condition']=='nf4_high_text']
    assert all(r['image_sha256']==[] and r['source_times_sec']==[] for r in text)
    with pytest.raises(ValueError):s.verify(prepared,paths['config'],False)
    frame=deepcopy(s.read(prepared/'prepared.json')['rows'][0]);frame['frame_sha256'][0]='wrong'
    with pytest.raises(ValueError):s.validate_row(frame,4)
