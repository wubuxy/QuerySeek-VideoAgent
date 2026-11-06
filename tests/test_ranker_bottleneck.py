import numpy as np
from scripts.audit_ranker_bottleneck import event_rows,matched_redundancy

def test_short_event_can_be_globally_buried_but_locally_ranked():
    w=np.array([[0.,50.,0.],[1.,51.,0.],[60.,64.,0.],[70.,74.,0.]])
    rr=event_rows(w,np.array([.9,.8,.3,.1]),[[60.,64.]],w[:2].tolist())[0]
    assert rr['pool_hit'] and rr['raw_rank']==3 and rr['within_short_rank']==1 and not rr['top5_hit']

def test_redundancy_counts_same_event_not_number_of_positive_windows():
    r=matched_redundancy([[0.,20.,.9],[1.,21.,.8],[30.,34.,.7]],[[0.,20.],[30.,34.]])
    assert r=={'matched':3,'distinct_events':2,'duplicate_matches':1}

def test_missing_event_and_empty_selection_are_retained():
    rr=event_rows(np.array([[0.,2.,0.]]),np.array([.2]),[[5.,8.]],[])[0]
    assert not rr['pool_hit'] and rr['raw_rank'] is None and rr['within_short_rank'] is None
    assert matched_redundancy([],[[5.,8.]])['matched']==0

def test_existing_stricter_nms_preserves_top1_and_removes_near_duplicates():
    from scripts.cheap_ranker_decode_probe import decode
    w=np.array([[0.,20.,0.],[2.,22.,0.],[30.,34.,0.]])
    scores=np.array([.9,.8,.7])
    original=decode(w,scores,.9);dedupe=decode(w,scores,.5)
    assert original[0]==dedupe[0] and len(original)==3 and len(dedupe)==2
    assert dedupe[1][:2]==(30.,34.)

def test_balanced_loss_keeps_query_mass_and_equal_events_despite_duplicate_counts():
    from scripts.cheap_ranker_weight_probe import balanced_weights
    w=np.array([[0.,4.,0.],[.5,4.5,0.],[10.,30.,0.],[40.,44.,0.]])
    ww=balanced_weights(w,np.array([1.,.6,1.,0.]),[[0.,4.],[10.,30.]])
    assert np.isclose(ww.sum(),1.) and np.isclose(ww[-1],.5)
    assert np.isclose(ww[0]+ww[1],.25) and np.isclose(ww[2],.25)

def test_balanced_loss_degenerate_class_is_finite_and_targets_unchanged():
    from scripts.cheap_ranker_weight_probe import balanced_weights
    w=np.array([[0.,4.,0.],[4.,8.,0.]])
    y=np.zeros(2);before=y.copy()
    np.testing.assert_array_equal(balanced_weights(w,y,[[9.,10.]]),[.5,.5])
    np.testing.assert_array_equal(y,before)
