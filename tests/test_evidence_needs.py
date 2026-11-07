from copy import deepcopy
import json
import pytest
from queryseek.retrieval.evidence_needs import (plan_prompt, validate_plan, verdict_prompt,
    decode_verdict, select_action, context_interval)


def inputs():
    request={'question':'What pattern is on the requested object?','choices':['striped','plain'],'duration_sec':60.}
    packet={'candidate_id':'c0','interval_sec':[10.,30.],
        'frames':[{'frame_id':str(i),'timestamp_sec':11.+i,'image_path':str(i)} for i in range(4)],
        'frame_sha256':['a','b','c','d']}
    plan={'target':'Requested object','relation':'Requested context','scope':'single_view'}
    return request,packet,plan


def value(decision='relation_unverified',evidence='1',region='full'):
    return dict(decision=decision,evidence=evidence,region=region,
                target_description='Visible panel at the left.',observed_fact='Panel visible; owner not established.')


@pytest.mark.parametrize('need',['target_ambiguous','relation_unverified','attribute_unreadable','temporal_unverified','global_unverified'])
def test_insufficient_observations_retain_actual_partial_evidence(need):
    req,packet,plan=inputs();r=decode_verdict(value(need,region='center'),req,packet,plan)
    assert r['state']=='insufficient' and r['evidence_frames']==[1] and r['evidence_source_times_sec']==[12.]


def test_static_answer_uses_one_frame_but_temporal_needs_distinct_source_times():
    req,packet,plan=inputs()
    assert decode_verdict(value('answer:A'),req,packet,plan)['answer']=='A'
    temporal={**plan,'scope':'temporal'}
    with pytest.raises(ValueError):decode_verdict(value('answer:A'),req,packet,temporal)
    assert decode_verdict(value('answer:A','0,1'),req,packet,temporal)['evidence_source_times_sec']==[11.,12.]
    packet['detail_views']=[{'parent_index':0,'parent_sha256':'a','region':'center','sha256':'crop',
                            'frame':{**packet['frames'][0],'image_path':'crop'}}]
    with pytest.raises(ValueError):decode_verdict(value('answer:A','0,4'),req,packet,temporal)


@pytest.mark.parametrize('bad',[value('target_unobserved','1'),value('attribute_unreadable','1','full'),
    value('answer:C'),value('relation_unverified','9'),value('answer:A','0,1'),value('unknown')])
def test_invalid_states_refs_regions_and_choices_rejected(bad):
    req,packet,plan=inputs()
    with pytest.raises(ValueError):decode_verdict(bad,req,packet,plan)


def test_global_local_clip_cannot_claim_whole_video_answer():
    req,packet,plan=inputs();plan['scope']='global'
    with pytest.raises(ValueError):decode_verdict(value('answer:A'),req,packet,plan)


def test_policy_prioritizes_visible_target_and_preserves_uncertainty():
    req,packet,plan=inputs();other={**deepcopy(packet),'candidate_id':'c1'}
    a={'candidate_id':'c0','status':'ok',**decode_verdict(value('target_unobserved','','full'),req,packet,plan)}
    b={'candidate_id':'c1','status':'ok',**decode_verdict(value('relation_unverified'),req,other,plan)}
    assert select_action([packet,other],[a,b],1,'rank_uniform')['candidate_id']=='c0'
    action=select_action([packet,other],[a,b],1,'evidence_routed')
    assert action['candidate_id']=='c1' and action['action']=='context' and action['anchor_index']==1
    assert a['state']=='insufficient'  # Non-observation never becomes a contradiction.
    assert select_action([packet,other],[a,b],0,'evidence_routed')['action']=='abstain'


def test_conflicting_supported_answers_abstain_in_both_policies():
    req,packet,plan=inputs();other={**deepcopy(packet),'candidate_id':'c1'}
    a={'candidate_id':'c0','status':'ok',**decode_verdict(value('answer:A'),req,packet,plan)}
    b={'candidate_id':'c1','status':'ok',**decode_verdict(value('answer:B'),req,other,plan)}
    for policy in ['rank_uniform','evidence_routed']:
        assert select_action([packet,other],[a,b],1,policy)['reason']=='conflicting_supported_answers'


def test_context_clamps_to_video_and_plan_has_no_choices_or_labels():
    assert context_interval(2.,60.)==[0.,10.]
    assert context_interval(59.,60.)==[35.,60.]
    prompt,_=plan_prompt('What pattern?');assert 'choices' not in prompt and 'right_answer' not in prompt
    with pytest.raises(ValueError):validate_plan({'target':'x','relation':'y','scope':'single_view','answer':'A'})


def test_real_parser_cross_field_and_field_order():
    lm=pytest.importorskip('lmformatenforcer');req,packet,plan=inputs();_,schema=verdict_prompt(req,packet,plan)
    examples=[(value('answer:A'),True),(value('target_unobserved','','full'),True),
              (value('relation_unverified'),True),(value('attribute_unreadable','1','center'),True),
              (value('attribute_unreadable','1','full'),False),(value('target_unobserved','1'),False)]
    for item,expected in examples:
        keys=list(item)
        for order in [keys,keys[::-1],keys[2:]+keys[:2]]:
            parser=lm.JsonSchemaParser(schema);accepted=True
            for char in json.dumps({k:item[k] for k in order}):
                if char not in parser.get_allowed_characters():accepted=False;break
                parser=parser.add_character(char)
            assert (accepted and parser.can_end())==expected


def prepared_fixture(demo,tmp_path,monkeypatch):
    from scripts import probe_evidence_needs as s
    monkeypatch.setattr(s.p.cache_paths,'ROOT',tmp_path)
    req={'qid':'synthetic','video_id':'sample','video_path':str(demo['video']),'duration_sec':60.,
         'question':'What pattern is on the object?','choices':['striped','plain']}
    asset=s.p.asset_from_probe('sample',demo['video']);cfg={'max_side':96,'pixels':65536,'extra_frames':8,
        'context_before_sec':24.,'context_after_sec':8.,'sample_count':1,'candidate_count':1,'max_generations':4}
    packet,_=s.p.observe(asset,{'candidate_id':'c0','interval_sec':[20.,30.],'origin':'retrieval'},4,cfg)
    return s,req,asset,cfg,packet


def test_crops_preserve_parent_pts_and_identity_cache_and_context(demo,tmp_path,monkeypatch):
    s,req,asset,cfg,packet=prepared_fixture(demo,tmp_path,monkeypatch)
    action={'action':'detail','candidate_id':'c0','anchor_index':0,'region':'center'}
    detail,cost=s.observe_action(req,packet,action,asset.sha256,cfg)
    assert cost['new_crops_created']==1 and detail['frames']==packet['frames']
    assert detail['detail_views'][0]['frame']['timestamp_sec']==packet['frames'][0]['timestamp_sec']
    assert s.observe_action(req,packet,action,asset.sha256,cfg)[1]['cache_hit']
    changed=s.observe_action(req,packet,{**action,'region':'upper_left'},asset.sha256,cfg)[1]
    assert not changed['cache_hit'] and changed['manifest_sha256']!=cost['manifest_sha256']
    context,_=s.observe_action(req,packet,{**action,'action':'context'},asset.sha256,cfg)
    assert context['interval_sec'][0]==0. and context['interval_sec'][1]==30.
    assert set(packet['frame_sha256'])<=set(context['frame_sha256'])
    from pathlib import Path
    Path(detail['detail_views'][0]['frame']['image_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):s.observe_action(req,packet,action,asset.sha256,cfg)


def test_synthetic_mock_end_to_end_reuses_identical_actions_and_is_not_accuracy(demo,tmp_path,monkeypatch):
    s,req,asset,cfg,packet=prepared_fixture(demo,tmp_path,monkeypatch)
    prepared,requests,config=[tmp_path/n for n in ['prepared.json','requests.json','config.json']]
    s.p.write_json(prepared,[{'request':req,'video_sha256':asset.sha256,'packets':[packet]}]);s.p.write_json(requests,[req])
    cfg.update(prepared_sha256=s.p.sha256_file(prepared),requests_sha256=s.p.sha256_file(requests))
    s.p.write_json(config,cfg);out=tmp_path/'outputs/study'
    summary=s.run(prepared,requests,config,out,mock=True)
    assert summary['backend_calls']==0 and summary['mock']
    row=s.p.lines(out/'records.jsonl')[0]
    assert row['branches']['evidence_routed']['observation']['reused_from']=='rank_uniform'
    assert row['branches']['rank_uniform']['observation']['verdict']['status']=='ok'
    with pytest.raises(ValueError):s.verify(out,config,False)
