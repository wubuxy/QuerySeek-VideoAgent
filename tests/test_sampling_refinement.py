import json
from pathlib import Path

import pytest

from queryseek.retrieval.sampling_refinement import plan_supplemental_times, supplemental_proposals


def test_matched_budget_exploration_and_query_dependence():
    times=list(range(0,120,4))
    scores=[1 if t==60 else 0 for t in times]
    plan=plan_supplemental_times(times,scores,120)
    assert len(plan["uniform"])==len(plan["query"])==32
    assert len(set(plan["query"]))==32
    assert all(0<=t<120 and min(abs(t-x) for x in times)>=.2 for t in plan["query"])
    assert sum(56<=t<=64 for t in plan["query"])>=8
    assert min(plan["query"])<10 and max(plan["query"])>110
    moved=plan_supplemental_times(times,[1 if t==20 else 0 for t in times],120)
    assert moved["uniform"]==plan["uniform"]
    assert moved["query"]!=plan["query"]
    assert plan_supplemental_times(times,scores,120)==plan


def test_tiny_duration_and_zero_budget():
    plan=plan_supplemental_times([0],[.5],.7,32)
    assert len(plan["query"])==len(plan["uniform"])==1
    assert plan_supplemental_times([0],[0],10,0)=={"uniform":[],"query":[]}


@pytest.mark.parametrize("times,scores,duration,budget,grid",[
    ([4,0],[1,0],10,32,.5),([0],[float("nan")],10,32,.5),
    ([0],[0],float("inf"),32,.5),([0],[0],10,-1,.5),
    ([0],[0],10,32,0),([10],[0],10,32,.5),
])
def test_invalid_sampling_inputs(times,scores,duration,budget,grid):
    with pytest.raises(ValueError):
        plan_supplemental_times(times,scores,duration,budget,grid)


def test_short_event_coverage_can_improve_with_new_observations():
    from queryseek.eval.qvhighlights import temporal_iou
    coarse=[0,4,8,12,16]
    scores=[0,.1,.4,.1,0]
    before,_=supplemental_proposals(coarse,scores,20,True)
    points=sorted(zip(coarse,scores))+[(9,.9),(10,1),(11,.9)]
    points.sort()
    after,_=supplemental_proposals([t for t,s in points],[s for t,s in points],20,True)
    best=lambda ws:max(temporal_iou(w.start_sec,w.end_sec,9,11) for w in ws)
    assert best(after)>best(before)
    assert all(0<=w.start_sec<w.end_sec<=20 for w in after)


def test_cache_key_and_output_guard():
    from types import SimpleNamespace
    from scripts.sampling_probe import ROOT,checked_output,feature_signature
    idx=SimpleNamespace(asset=SimpleNamespace(sha256="a"*64))
    assert feature_signature(idx,"model-a",[1],512)!=feature_signature(idx,"model-b",[1],512)
    assert feature_signature(idx,"model-a",[1],512)!=feature_signature(idx,"model-a",[2],512)
    assert feature_signature(idx,"model-a",[1],512)!=feature_signature(idx,"model-a",[1],256)
    with pytest.raises(ValueError):checked_output(ROOT/"outputs/../data/unsafe")


def test_mock_video_probe_and_cache(demo,qs_env,monkeypatch):
    from queryseek.config import Config
    from queryseek.eval.runner import dataset_from_labels
    import scripts.sampling_probe as probe
    # Restrict the experimental output root to the isolated fixture workspace.
    monkeypatch.setattr(probe,"ROOT",qs_env)
    cfg=Config.load().resolved()
    cfg.embedding.backend="mock"
    samples=dataset_from_labels(demo["labels"],demo["video"],demo["srt"])[:1]
    first=probe.run_probe(samples,cfg,qs_env/"outputs/first",True,budget=4,grid_sec=.46)
    assert first["coverage"]==1 and first["vlm_calls"]==0
    assert first["physical_work"]["encoded_frames"]>0
    def forbidden(*args,**kwargs):raise AssertionError("Cached extraction must be reused")
    monkeypatch.setattr(probe,"extract_scene_frames",forbidden)
    second=probe.run_probe(samples,cfg,qs_env/"outputs/second",True,budget=4,grid_sec=.46)
    assert second["metrics"]==first["metrics"]
    assert second["physical_work"]["encoded_frames"]==0
    protocol=json.loads((qs_env/"outputs/first/protocol.json").read_text())
    assert "gt_windows" not in protocol


def test_supplemental_requests_preserve_actual_source_timestamps():
    from types import SimpleNamespace
    from scripts.sampling_probe import supplemental_observations
    frames=[SimpleNamespace(timestamp_sec=1.083333),SimpleNamespace(timestamp_sec=2.083333)]
    observed=supplemental_observations([1.013,2.017],frames,[.3,.8])
    assert observed[1.013]==(1.083333,.3)
    assert observed[2.017]==(2.083333,.8)
    with pytest.raises(RuntimeError):supplemental_observations([1.013],frames,[.3,.8])


def test_failed_query_remains_in_all_metric_denominators(demo,qs_env,monkeypatch):
    from queryseek.config import Config
    from queryseek.eval.runner import dataset_from_labels
    import scripts.sampling_probe as probe
    monkeypatch.setattr(probe,"ROOT",qs_env)
    cfg=Config.load().resolved();cfg.embedding.backend="mock"
    samples=dataset_from_labels(demo["labels"],demo["video"],demo["srt"])[:1]
    def fail(*args,**kwargs):raise RuntimeError("fixture extraction failure")
    monkeypatch.setattr(probe,"extra_features",fail)
    result=probe.run_probe(samples,cfg,qs_env/"outputs/failed",True,budget=4)
    assert result["incomplete"] and result["coverage"]==0
    assert len(result["errors"])==1
    assert all(m["MR-full-R1@0.5"]==0 for m in result["metrics"].values())


def test_time_alignment_invariant_to_redundant_linear_observations():
    from queryseek.retrieval.sampling_refinement import time_aligned_proposals
    coarse=time_aligned_proposals([0,4,8,12],[0,1,0,0],16,True)
    denser=time_aligned_proposals([0,2,4,6,8,10,12],[0,.5,1,.5,0,0,0],16,True)
    assert coarse==denser
    assert all(0<=w.start_sec<w.end_sec<=16 for w in denser[0])
    assert time_aligned_proposals([],[],10)==([],[])


def test_clustered_interval_retains_source_pairing():
    from scripts.analyze_sampling_probe import clustered_delta
    result=clustered_delta([0,0,0],[1,1,1],["same","same","other"],repeats=100)
    assert result["delta"]==1 and result["ci95"]==[1,1]
    assert result["n_queries"]==3 and result["n_source_videos"]==2
    unchanged=clustered_delta([0,1,1],[0,1,1],["a","a","b"],repeats=100)
    assert unchanged["delta"]==0 and unchanged["ci95"]==[0,0]
