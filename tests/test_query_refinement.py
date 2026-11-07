import json
from dataclasses import replace

import numpy as np
import pytest

from queryseek.agent.temporal_tree import NodeObservation
from queryseek.agent import tree_observations as cache
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.query_refinement import METHODS, merge_observations, padded_window, rank_regions, retrieve, select_review_frames
from queryseek.schemas import VideoFrame, TimeWindow
from scripts import probe_query_refinement as probe


def obs(a,b,count=12):
    times=[a+(i+.5)*(b-a)/count for i in range(count)]
    frames=[VideoFrame(frame_id=f't{t:.6f}',timestamp_sec=t,image_path='unused',source='uniform') for t in times]
    features=np.array([[.1,.99] if t<60 else [.99,.1] for t in times],dtype=np.float32)
    return NodeObservation(frames,features)


def settings():
    return {'root_frames':12,'dense_frames':36,'local_frames':12,'selected_regions':2,
            'region_pool':3,'review_frames':4,'top_k':5,'nms_iou':.9,
            'split':{'min_children':3,'max_children':3}}


def test_query_scores_select_relevant_regions_and_context_preserves_time_contract():
    ranked=rank_regions([10.,30.,50.,70.],[.1,.2,.9,.8],[(0.,40.),(40.,80.)],80.)
    assert ranked[0]['region_index']==1 and ranked[0]['score']==pytest.approx(.85)
    assert padded_window([2.,12.],120.,10.,64.)==(0.,64.)
    assert padded_window([110.,119.],120.,10.,64.)==(56.,120.)
    assert padded_window([10.,20.],30.,10.,64.)==(0.,30.)
    with pytest.raises(ValueError):padded_window([20.,40.],30.,10.)


def test_merging_source_time_duplicates_and_review_selection_remain_observed():
    first=obs(0.,120.)
    merged=merge_observations([first,first])
    assert len(merged.frames)==12
    windows=[TimeWindow(start_sec=0.,end_sec=20.,score=1.),TimeWindow(start_sec=10.,end_sec=30.,score=.9)]
    selected,fallback=select_review_frames(merged,np.arange(12),windows,8)
    assert len(selected)==len({f.frame_id for f in selected})==8
    assert fallback>0
    assert {f.frame_id for f in selected}<={f.frame_id for f in first.frames}
    assert all(a.timestamp_sec<b.timestamp_sec for a,b in zip(selected,selected[1:]))


@pytest.mark.parametrize('method',METHODS)
def test_methods_share_observation_caps_and_do_not_certify_relevance(method):
    calls=[]
    def provider(a,b,budget):
        calls.append((a,b,budget));return obs(a,b,budget)
    record,review=retrieve(method,120.,np.array([1.,0.]),provider,settings())
    assert record['charged_image_observations']==36
    assert sum(c[2] for c in calls)==36
    assert len(review)==4 and record['evidence_status']=='unverified'
    assert all(0<=a<b<=120. for a,b,_ in calls)
    assert all(0<=w['start_sec']<w['end_sec']<=120. for w in record['windows'])


def setup_inputs(demo,tmp_path,monkeypatch):
    monkeypatch.setattr(cache,'ROOT',tmp_path)
    monkeypatch.setattr(probe.study,'ROOT',tmp_path)
    monkeypatch.setattr(probe.study.baseline,'ROOT',tmp_path)
    requests,labels,config=(tmp_path/n for n in ['requests.json','labels.json','config.json'])
    cache.write_json(requests,[{'qid':'x','video_id':'sample','video_path':str(demo['video']),
        'duration_sec':60.,'question':'Find the architecture section.','choices':['one','two']}])
    cache.write_json(labels,[{'qid':'x','right_answer':'A','gt_intervals_sec':[[12.,24.]]}])
    cfg=probe.read(probe.ROOT/'configs/query_refinement_v1.json')
    cfg.update(requests_sha256=sha256_file(requests),labels_sha256=sha256_file(labels),
               retrieval=settings(),max_generations=4)
    cfg['sampling']['max_side']=96
    for condition in cfg['conditions']:condition['frames']=4
    cache.write_json(config,cfg)
    return requests,labels,config,cfg


def test_mock_end_to_end_isolation_input_hashes_and_no_relabeling_as_real(demo,tmp_path,monkeypatch):
    requests,labels,config,cfg=setup_inputs(demo,tmp_path,monkeypatch)
    prepared=tmp_path/'outputs/retrieval'
    result=probe.prepare(requests,config,prepared,mock=True)
    assert result['completed']==3 and result['failed']==0 and result['new_vlm_calls']==0
    trace=(prepared/'retrieval.jsonl').read_text()
    assert 'right_answer' not in trace and 'gt_intervals' not in trace
    assert all(r['charged_image_observations']==36 for r in map(json.loads,trace.splitlines()))
    context=tmp_path/'outputs/context'
    result=probe.prepare_context(prepared,labels,config,context)
    assert result['rows']==4 and result['new_vlm_calls']==0
    rows=probe.read(context/'prepared.json')
    assert all(set(r['request'])=={'qid','video_id','video_path','duration_sec','question','choices'} for r in rows)
    with pytest.raises(ValueError,match='provenance/mock'):
        probe.run(context,config,tmp_path/'outputs/false_real')
    run=tmp_path/'outputs/run'
    assert probe.run(context,config,run,mock=True)['completed']==4
    with pytest.raises(ValueError,match='Real frozen'):
        probe.evaluate(requests,labels,config,prepared,run,tmp_path/'outputs/eval')
    with pytest.raises(FileExistsError):probe.prepare(requests,config,prepared,mock=True)
    rows[0]['frames'][0]['timestamp_sec']+=.5
    cache.write_json(context/'prepared.json',rows)
    with pytest.raises(ValueError,match='provenance/mock'):
        probe.run(context,config,tmp_path/'outputs/tampered',mock=True)


def test_label_fields_rejected_before_preparation(demo,tmp_path,monkeypatch):
    requests,labels,config,cfg=setup_inputs(demo,tmp_path,monkeypatch)
    rows=probe.read(requests);rows[0]['gt_intervals_sec']=[[12.,24.]]
    cache.write_json(requests,rows);cfg['requests_sha256']=sha256_file(requests)
    cache.write_json(config,cfg)
    with pytest.raises(ValueError,match='whitelisted'):
        probe.prepare(requests,config,tmp_path/'outputs/leaky',mock=True)


def test_failed_retrieval_remains_in_records_and_blocks_oracle_stage(demo,tmp_path,monkeypatch):
    requests,labels,config,cfg=setup_inputs(demo,tmp_path,monkeypatch)
    original=probe.retrieve
    def failure(method,*a,**kw):
        if method=='uniform_refine':raise RuntimeError('must not enter trace')
        return original(method,*a,**kw)
    monkeypatch.setattr(probe,'retrieve',failure)
    prepared=tmp_path/'outputs/failed'
    result=probe.prepare(requests,config,prepared,mock=True)
    assert result['scheduled']==3 and result['completed']==1 and result['failed']==2
    assert not result['cost_accounting_complete']
    text=(prepared/'retrieval.jsonl').read_text()
    assert len(text.splitlines())==3 and 'must not enter trace' not in text
    with pytest.raises(ValueError,match='finish and freeze'):
        probe.prepare_context(prepared,labels,config,tmp_path/'outputs/context')


def test_evaluator_separates_observed_review_and_answer_failures(tmp_path,monkeypatch):
    monkeypatch.setattr(cache,'ROOT',tmp_path)
    requests,labels,config=(tmp_path/n for n in ['requests.json','labels.json','config.json'])
    cache.write_json(requests,[{'qid':q,'video_id':'sample','video_path':'unused','duration_sec':120.,
        'question':'Find the event','choices':['one','two']} for q in ['a','b']])
    cache.write_json(labels,[{'qid':q,'right_answer':'A','gt_intervals_sec':[[20.,40.]]} for q in ['a','b']])
    cfg=probe.read(probe.ROOT/'configs/query_refinement_v1.json')
    cfg.update(requests_sha256=sha256_file(requests),labels_sha256=sha256_file(labels),max_generations=8)
    cache.write_json(config,cfg)
    retrieval,run=tmp_path/'outputs/retrieval',tmp_path/'outputs/run'
    retrieval.mkdir(parents=True);run.mkdir()
    records=[];predictions=[]
    for q in ['a','b']:
        for method in METHODS:
            records.append({'qid':q,'method':method,'status':'ok' if q=='a' else 'error',
                'coarse_source_times_sec':[1.],'observed_source_times_sec':[30.],'review_source_times_sec':[1.],
                'windows':[{'start_sec':20.,'end_sec':40.}]})
        for condition in cfg['conditions']:
            predictions.append({'qid':q,'condition':condition['name'],'status':'ok' if q=='a' else 'error',
                                'answer':'A','intervals':[[20.,40.]],'model_calls':[]})
    (retrieval/'retrieval.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    (run/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in predictions))
    cache.write_json(retrieval/'manifest.json',{'mock':False,'config_sha256':sha256_file(config),
                                             'retrieval_sha256':sha256_file(retrieval/'retrieval.jsonl')})
    cache.write_json(run/'run_summary.json',{'mock':False,'model_load_sec':0.})
    result=probe.evaluate(requests,labels,config,retrieval,run,tmp_path/'outputs/eval')
    for metrics in result['methods'].values():
        assert metrics['coarse_hit']==metrics['review_hit']==0.
        assert metrics['observed_hit']==metrics['rank1_iou']==.5
        assert metrics['qa']['n']==2 and metrics['qa']['joint_at_05']==.5
    assert result['oracle']['n']==2 and result['oracle']['qa_correct']==.5
