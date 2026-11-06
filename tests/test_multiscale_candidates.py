import inspect
import numpy as np
import pytest
from queryseek.retrieval.multiscale_candidates import multiscale_table
from queryseek.retrieval.candidate_ranker import candidate_table,rank_candidates
from scripts.multiscale_ranker_study import coverage,infer


def test_grid_preserves_original_candidates_and_covers_both_video_edges():
    t=np.arange(30,dtype=float);s=np.sin(t)
    old,_=candidate_table(t,s,30.);w,x=multiscale_table(t,s,30.,[2,4,8,16,32])
    np.testing.assert_array_equal(w[:len(old)],old)
    assert x.shape==(len(w),14) and np.isfinite(x).all()
    assert len({(a,b) for a,b,_ in w})==len(w)
    assert all(isinstance(a,float) and 0<=a<b<=30. for a,b,_ in w.tolist())
    assert (0.,2.) in {(a,b) for a,b,_ in w} and (28.,30.) in {(a,b) for a,b,_ in w}
    assert (0.,30.) in {(a,b) for a,b,_ in w}
    assert all((a,b) in {(u,v) for u,v,_ in w} for a,b,_ in rank_candidates(w))


@pytest.mark.parametrize('lengths,stride,minimum', [([], .25,1),([0],.25,1),([np.nan],.25,1),([2],0,1),([2],.25,0),([2],np.nan,1)])
def test_grid_rejects_invalid_configuration(lengths,stride,minimum):
    with pytest.raises(ValueError):multiscale_table([0.,1.],[0.,1.],2.,lengths,stride,minimum)


def test_fractional_short_video_and_irregular_source_pts():
    w,x=multiscale_table([0.,.033,.35],[0.,.3,.1],.51,[2,4])
    assert all(0<=a<b<=.51 for a,b,_ in w) and np.isfinite(x).all()
    with pytest.raises(ValueError):multiscale_table([0.,.51],[0.,1.],.51,[2])


def test_pool_build_and_inference_have_no_label_argument_or_target_read():
    assert 'gt' not in inspect.signature(multiscale_table).parameters
    class M:
        def predict(self,x):return x[:,5]
    arrays={'union_w_1':np.array([[0.,2.,0.],[3.,5.,0.]]),'union_x_1':np.ones((2,14)),'union_y_1':'must not read'}
    pp=infer(M(),[{'qid':1}],arrays,'union','full',{'top_k':5,'nms_iou':.9})
    assert len(pp[1])==2


def test_event_coverage_keeps_long_and_short_separate():
    pool={1:[[0.,4.,1.],[20.,60.,1.]]};gt={1:{'relevant_windows':[[0.,4.],[20.,60.],[6.,18.]]}}
    report=coverage(pool,gt)
    assert report['short']['hits_0.9']==1 and report['long']['hits_0.9']==1
    assert report['middle']['events']==1 and report['middle']['hits_0.5']==0
