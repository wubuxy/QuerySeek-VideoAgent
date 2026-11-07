from copy import deepcopy
import pytest
from queryseek.retrieval.anchor_observations import retain_seed


def inputs():
    p={'candidate_id':'c5','origin':'seed_window','interval_sec':[0.,64.],'seed_time_sec':32.,
        'frames':[{'frame_id':str(i),'timestamp_sec':float(4+8*i),'image_path':str(i),'source':'scene'} for i in range(8)],
        'frame_sha256':[str(i) for i in range(8)]}
    f={'frame_id':'seed-original','timestamp_sec':32.,'image_path':'actual_observed.jpg','source':'uniform'}
    return p,f


def test_anchor_is_retained_without_increasing_budget_or_mutating_prior_record():
    p,f=inputs();old=deepcopy(p);new,audit=retain_seed(p,f,'actual-sha',100.)
    assert p==old and len(new['frames'])==8
    selected=next(i for i,x in enumerate(new['frames']) if x['timestamp_sec']==32.)
    assert new['frames'][selected]['image_path']=='actual_observed.jpg'
    assert new['frame_sha256'][selected]=='actual-sha'
    assert audit['replaced_source_time_sec']==28. and not audit['frame_budget_changed']
    assert retain_seed(new,f,'actual-sha',100.)[0]==new


@pytest.mark.parametrize('change', ['wrong_time','outside_window','not_seed','missing_hash'])
def test_unobserved_or_invalid_seeds_cannot_enter_packet(change):
    p,f=inputs();sha='actual-sha'
    if change=='wrong_time':f['timestamp_sec']=30.
    if change=='outside_window':p['interval_sec']=[0.,32.]
    if change=='not_seed':p['origin']='retrieval'
    if change=='missing_hash':sha=''
    with pytest.raises(ValueError):retain_seed(p,f,sha,100.)
