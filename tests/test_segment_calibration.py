import numpy as np
import pytest

from queryseek.retrieval.segment_calibration import calibrated_windows,time_weighted_mean
from queryseek.schemas import TimeWindow


def window(a,b,score=1):return TimeWindow(start_sec=a,end_sec=b,score=score)


def test_time_weighting_does_not_count_redundant_local_samples_as_support():
    sparse=time_weighted_mean([0,1,9],[1,1,0],10,[(0,10)])
    dense=time_weighted_mean([0,.2,.4,.6,.8,1,9],[1,1,1,1,1,1,0],10,[(0,10)])
    assert sparse==dense==.5


def test_original_geometry_and_inputs_are_preserved():
    original=[window(2,14,5),window(18,28,3)];short=[window(5,7,10)]
    saved=[w.model_dump() for w in original+short]
    times=[0,4,6,8,12,16,20,24,28]
    pool,selected,features=calibrated_windows(original,short,times,[0,1,2,1,1,0,1,1,0],
                                             times,[0,1,2,1,1,0,1,1,0],30,"contrast")
    assert {(w.start_sec,w.end_sec) for w in original}.issubset({(w.start_sec,w.end_sec) for w in pool})
    assert saved==[w.model_dump() for w in original+short]
    assert len(selected)<=5 and all(np.isfinite(w.score) for w in pool)
    assert all(0<=w.start_sec<w.end_sec<=30 for w in pool)
    assert features[2]["contrast"]>0


def test_locked_anchor_cannot_be_displaced_by_short_peak():
    anchor=window(0,20,5);short=window(8,10,999)
    pool,selected,features=calibrated_windows([anchor],[short],[0,4,8,12,16],[0,0,1,0,0],
                       [0,4,8,9,10,12,16],[0,0,1,1,0,0,0],20,"contrast",anchor)
    assert (selected[0].start_sec,selected[0].end_sec)==(0,20)
    assert selected[0].score>max(w.score for w in pool if w.start_sec!=0)
    assert any(f.get("locked_anchor") for f in features)


def test_constant_scores_and_absent_local_context():
    pool,_,features=calibrated_windows([window(0,10)],[],[0,5],[.3,.3],[0,5],[.3,.3],10,"contrast")
    assert pool[0].score==0 and features[0]["context_observations"]==0
    assert features[0]["context_mean"]==.3


def test_no_actual_inside_observation_prevents_imputed_promotion():
    pool,_,features=calibrated_windows([window(3,4)],[],[0,5,9],[0,1,0],[0,5,9],[0,1,0],10,"shared")
    assert features[0]["inside_observations"]==0 and pool[0].score==0


@pytest.mark.parametrize("times,scores,duration,ranges",[
    ([0,0],[1,1],10,[(0,10)]),([0],[float("nan")],10,[(0,10)]),
    ([10],[1],10,[(0,10)]),([0],[1],10,[(0,11)]),([0],[1],10,[(5,5)]),
])
def test_invalid_time_contract(times,scores,duration,ranges):
    with pytest.raises(ValueError):time_weighted_mean(times,scores,duration,ranges)


def test_unknown_mode_invalid_candidate_and_nonoriginal_anchor():
    args=([window(0,5)],[],[0,5],[0,1],[0,5],[0,1],10)
    with pytest.raises(ValueError):calibrated_windows(*args,mode="unknown")
    with pytest.raises(ValueError):calibrated_windows(*args,locked_top1=window(1,2))
    with pytest.raises(ValueError):calibrated_windows([window(0,11)],[],*args[2:])


def test_group_builder_never_discards_original_pool():
    from scripts.calibration_probe import build_groups
    observations={"base":([0,4,8,12,16],[0,.1,1,.3,0]),
                  "uniform":([0,4,6,8,10,12,16],[0,.1,.5,1,.7,.3,0])}
    groups,_=build_groups(observations,20,["shared_base","uniform_contrast","uniform_locked_contrast"])
    original={(w["start_sec"],w["end_sec"]) for w in groups["E4"]["all_windows"]}
    assert all(original.issubset({(w["start_sec"],w["end_sec"]) for w in g["all_windows"]}) for g in groups.values())
    assert groups["uniform_locked_contrast"]["top5"][0][:2]==groups["E4"]["top5"][0][:2]


def test_mock_probe_runs_without_model_and_includes_failures(demo,qs_env,monkeypatch):
    from queryseek.config import Config
    from queryseek.eval.runner import dataset_from_labels
    from queryseek.ingest.pipeline import ingest
    import scripts.calibration_probe as probe
    import scripts.sampling_probe as sampling
    monkeypatch.setattr(sampling,"ROOT",qs_env)
    cfg=Config.load().resolved();cfg.embedding.backend="mock"
    samples=dataset_from_labels(demo["labels"],demo["video"],demo["srt"])[:1]
    ingest(samples[0].video_path,video_id=samples[0].video_id,srt_path=samples[0].srt_path,config=cfg)
    def forbidden(*args,**kwargs):raise AssertionError("Mock must not construct a real model")
    monkeypatch.setattr(probe,"build_embedder",forbidden)
    result=probe.run_experiment(samples,cfg,qs_env/"outputs/calibration",use_mock=True,spec_names=["contrast_base"])
    assert result["coverage"]==1 and result["vlm_calls"]==result["image_encoding_calls"]==0
    def failure(*args,**kwargs):raise RuntimeError("fixture cache failure")
    monkeypatch.setattr(probe,"load_observations",failure)
    failed=probe.run_experiment(samples,cfg,qs_env/"outputs/calibration_failed",use_mock=True,spec_names=["contrast_base"])
    assert failed["coverage"]==0 and failed["incomplete"]
    assert failed["development_winner"] is None
    assert all(m["MR-full-R1@0.5"]==0 for m in failed["metrics"].values())
