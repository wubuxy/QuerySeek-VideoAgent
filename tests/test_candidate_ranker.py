from pathlib import Path
import copy
import numpy as np
import pytest

pytest.importorskip('sklearn')
from threadpoolctl import threadpool_limits
from queryseek.retrieval.candidate_ranker import (candidate_table,candidate_features,training_targets,
    source_equal_weights,model_inputs,rank_candidates,fit_ranker,FEATURE_NAMES)
from queryseek.retrieval.curve_spans import curve_spans,select_curve_windows
from scripts import candidate_ranker_study as study


def sample():
    t=np.arange(12,dtype=float);s=np.array([0,0,0,1,1,1,1,0,0,.5,.5,0.])
    return t,s


def test_baseline_pool_geometry_scores_and_top5_are_preserved():
    t,s=sample();w,x=candidate_table(t,s,12.)
    expected=[(a.start_sec,a.end_sec,a.score) for a in select_curve_windows(curve_spans(t,s,12.),5,.9)]
    assert rank_candidates(w)==expected and x.shape==(len(w),14)
    replacement=rank_candidates(w,np.arange(len(w),dtype=float))
    assert all((a,b) in {(float(v[0]),float(v[1])) for v in w} for a,b,_ in replacement)
    assert all(isinstance(a,float) and 0<=a<b<=12. for a,b,_ in replacement)


@pytest.mark.parametrize('mode',['nan','unordered','outside','zero_duration','bad_window'])
def test_time_and_feature_contracts_reject_invalid_inputs(mode):
    t,s=sample();duration=12.;windows=[(2.,7.,1.)]
    if mode=='nan':s[2]=np.nan
    elif mode=='unordered':t[1]=t[0]
    elif mode=='outside':t[-1]=12.
    elif mode=='zero_duration':duration=0.
    else:windows=[(-1.,3.,1.)]
    with pytest.raises(ValueError):candidate_features(t,s,duration,windows)


def test_utility_target_matches_iou_thresholds_and_uses_train_labels_only():
    w=np.array([[0.,10.,0.],[0.,20.,0.],[12.,18.,0.]])
    np.testing.assert_array_equal(training_targets(w,[[0.,10.]],20.),[1.,.1,0.])
    with pytest.raises(ValueError):training_targets(w,[[0.,21.]],20.)
    t,s=sample();first=candidate_table(t,s,12.)
    # There is no GT parameter in candidate extraction or scoring.
    training_targets(first[0],[[0.,2.]],12.)
    second=candidate_table(t,s,12.)
    assert all(np.array_equal(a,b) for a,b in zip(first,second))


def test_structure_control_uses_only_geometry_on_fixed_query_generated_pool():
    t,s=sample();windows=[(2.,7.,1.),(9.,11.,.5)]
    a=candidate_features(t,s,12.,windows);b=candidate_features(t,s[::-1],12.,windows)
    np.testing.assert_array_equal(model_inputs(a,'structure'),model_inputs(b,'structure'))
    assert not np.array_equal(model_inputs(a,'full'),model_inputs(b,'full'))
    with pytest.raises(ValueError):model_inputs(a,'no_query')


def test_source_weights_equal_sources_then_queries_not_candidate_counts():
    w=source_equal_weights(['A','A','B'],[2,4,3])
    assert np.isclose(w[:6].sum(),w[6:].sum())
    assert np.isclose(w[:2].sum(),w[2:6].sum()) and np.isclose(w.mean(),1.)
    with pytest.raises(ValueError):source_equal_weights(['A'],[0])


def test_group_folds_keep_multiple_queries_of_source_together_and_full_coverage():
    items=[{'qid':i,'source':str(i//2)} for i in range(12)]
    folds=study.source_folds(items,3)
    assert sorted(q for f in folds for q in f['test_qids'])==list(range(12))
    lookup={r['qid']:r['source'] for r in items}
    assert all(not {lookup[q] for q in f['train_qids']}&{lookup[q] for q in f['test_qids']} for f in folds)


def test_real_sklearn_fit_fixed_budget_save_reload_and_ranking(tmp_path):
    import joblib
    cfg=study.read(study.ROOT/'configs/candidate_ranker_v1.json')['model']
    cfg={**cfg,'max_iter':15,'min_samples_leaf':2,'l2_regularization':0.}
    x=np.zeros((100,len(FEATURE_NAMES)));x[:,5]=np.linspace(0,1,100)
    y=(x[:,5]>.5).astype(float)
    with threadpool_limits(limits=1):
        model=fit_ranker(x,y,np.ones(100),cfg)
        path=tmp_path/'own_model.joblib';joblib.dump(model,path);restored=joblib.load(path)
        assert model.n_iter_==15 and not model.early_stopping
        np.testing.assert_array_equal(model.predict(x),restored.predict(x))
        assert model.predict(x)[-1]>model.predict(x)[0]


def test_candidate_cache_separates_version_config_source_and_input_hashes():
    original={'pipeline_version':'2','config':{'min_leaf':20},'source_sha256':{'a':'sha'},'input_sha256':{'video':'sha'}}
    for field in original:
        changed=copy.deepcopy(original);changed[field]='changed'
        assert study.digest(original)!=study.digest(changed)


def test_validation_labels_never_enter_candidate_training_table(tmp_path):
    t,s=sample()
    records=[{'entry':{'qid':i,'vid':f'Synth{i:06d}_0.0_12.0','relevant_windows':[[2.,7.]]},
              'times':t,'images':s[:,None],'query_embedding':np.ones(1),'duration_sec':12.,
              'split':'train' if i==0 else 'validation'} for i in range(2)]
    p={'config':{},'source_sha256':{}}
    _,arrays=study.prepare_records(records,tmp_path,p)
    assert 'y_0' in arrays and 'y_1' not in arrays
    records[1]['entry']['relevant_windows']=[[-5.,1000.]]
    other=tmp_path/'other';other.mkdir()
    _,altered=study.prepare_records(records,other,p)
    assert arrays.keys()==altered.keys()
    assert all(np.array_equal(arrays[k],altered[k]) for k in arrays)


def test_failed_query_keeps_empty_predictions_and_candidate_denominator(tmp_path):
    t,s=sample();bad=s.copy();bad[0]=np.nan
    record={'entry':{'qid':1,'vid':'Synth000001_0.0_12.0','relevant_windows':[[2.,7.]]},
            'times':t,'images':bad[:,None],'query_embedding':np.ones(1),'duration_sec':12.,'split':'validation'}
    items,arrays=study.prepare_records([record],tmp_path,{'config':{},'source_sha256':{}})
    assert items[0]['candidates']==0
    assert study.infer(None,items,arrays,'full',{'top_k':5,'nms_iou':.9})=={1:[]}
    assert len(study.read(tmp_path/'failures.json'))==1


def test_protocol_refuses_changed_frozen_artifact(tmp_path,monkeypatch):
    from queryseek.ingest.cache import write_json_atomic
    from queryseek.ingest.ffprobe import sha256_file
    monkeypatch.setattr(study,'ROOT',tmp_path)
    source=tmp_path/'old.json';source.write_text('original')
    out=tmp_path/'run';out.mkdir()
    write_json_atomic(out/'protocol.json',{'preserved_sha256':{'old.json':sha256_file(source)},'source_sha256':{}})
    study.verify_inputs(out);source.write_text('changed')
    with pytest.raises(ValueError,match='historical'):study.verify_inputs(out)


def test_output_traversal_is_rejected():
    with pytest.raises(ValueError):study.checked_output(study.ROOT/'outputs/../../escape')
