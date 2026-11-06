import pytest
from queryseek.retrieval.evidence_spans import evidence_spans

def obs(t,r): return {'seconds':t,'relevant':r}
def test_midpoint_boundaries():
    ws=evidence_spans([obs(0,False),obs(4,True),obs(8,True),obs(12,False)],20)
    assert [(w.start_sec,w.end_sec) for w in ws]==[(2,10)]
def test_negative_breaks_event():
    assert len(evidence_spans([obs(4,True),obs(8,False),obs(12,True)],20))==2
def test_unobserved_gap_not_bridged():
    assert len(evidence_spans([obs(4,True),obs(40,True)],50))==2
def test_edge_clamp():
    ws=evidence_spans([obs(0,True),obs(4,True)],5)
    assert all(0<=w.start_sec<w.end_sec<=5 for w in ws)
def test_empty(): assert evidence_spans([],10)==[]
def test_invalid_timestamp():
    with pytest.raises(ValueError): evidence_spans([obs(float('nan'),True)],10)
def test_rank_support_not_purity():
    ws=evidence_spans([obs(1,True),obs(5,False),obs(9,True),obs(13,True)],20)
    assert ws[0].score==2
@pytest.mark.parametrize('kwargs',[{'max_gap':0},{'max_width':0},{'max_width':float('nan')}])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):evidence_spans([obs(2,True)],10,**kwargs)
