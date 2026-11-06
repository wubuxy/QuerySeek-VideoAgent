from queryseek.retrieval.curve_spans import curve_spans, select_curve_windows
import pytest

def test_curve_ranges():
    ws=curve_spans([0,4,8,12,16],[0,.1,.9,.8,0],20)
    assert ws and all(0<=w.start_sec<w.end_sec<=20 for w in ws)
def test_curve_order_invalid():
    with pytest.raises(ValueError):curve_spans([4,0],[0,1],10)
def test_curve_empty():assert curve_spans([],[],10)==[]
def test_curve_single():assert len(curve_spans([2],[.5],10))==1

def test_curve_search_no_vlm_and_topk():
    from types import SimpleNamespace
    import numpy as np
    from queryseek.retrieval.curve_spans import curve_search
    from queryseek.schemas import VideoFrame
    class Emb:
        def encode_texts(self,queries):return np.array([[1.]])
        def encode_images(self,paths):raise AssertionError('Cached features must be reused')
    fs=[VideoFrame(frame_id=str(i),timestamp_sec=i*4,image_path='unused') for i in range(5)]
    idx=SimpleNamespace(frames=fs,asset=SimpleNamespace(duration_sec=20))
    cfg=SimpleNamespace(retrieval=SimpleNamespace(top_k_windows=2))
    r=curve_search('video','event',idx,Emb(),cfg,np.array([[0],[.1],[.9],[.8],[0]]))
    assert r.vlm_calls==0 and not r.used_frame_ids and 0<len(r.windows)<=2
    assert r.indexed_frames == 5 and r.online_encoded_frames == 0 and r.embedding_cache_hit is True
    assert r.evidence and all(w.evidence_ids for w in r.windows)
    evidence = {e.evidence_id: e for e in r.evidence}
    for w in r.windows:
        assert all(w.start_sec <= evidence[eid].timestamp_sec < w.end_sec for eid in w.evidence_ids)
    expected = select_curve_windows(
        curve_spans([i*4 for i in range(5)], [0,.1,.9,.8,0], 20), 2)
    assert [(w.start_sec, w.end_sec, w.score) for w in r.windows] == [(w.start_sec,w.end_sec,w.score) for w in expected]

def test_e4_runner_does_not_build_vlm(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from queryseek.eval import runner
    from queryseek.models import factory
    from queryseek.config import Config
    from queryseek.schemas import VideoFrame
    class Emb:
        def encode_texts(self,queries):return np.array([[1.]])
    fs=[VideoFrame(frame_id=str(i),timestamp_sec=i*4,image_path='unused') for i in range(5)]
    idx=SimpleNamespace(frames=fs,asset=SimpleNamespace(duration_sec=20),video_id='v',content_hints=[])
    monkeypatch.setattr(runner,'_real_embedder',lambda cfg,index:(Emb(),np.array([[0],[.1],[.9],[.8],[0]])))
    def forbidden(*args,**kwargs):raise AssertionError('E4 must not construct an API client')
    monkeypatch.setattr(factory,'build_vlm',forbidden)
    sample=SimpleNamespace(query='event',qid=0)
    r=runner._run_method(Config.load(),idx,sample,'E4',16,False)
    assert r.vlm_calls==0 and r.used_frames==0 and r.windows

def test_curve_selection_retains_top_and_distinct_event():
    from queryseek.retrieval.curve_spans import select_curve_windows
    from queryseek.schemas import TimeWindow
    a=TimeWindow(start_sec=0,end_sec=20,score=3)
    duplicate=TimeWindow(start_sec=0,end_sec=19.9,score=2)
    other=TimeWindow(start_sec=40,end_sec=45,score=1)
    selected=select_curve_windows([duplicate,other,a],2)
    assert selected==[a,other]

def test_curve_selection_rejects_invalid_settings():
    from queryseek.retrieval.curve_spans import select_curve_windows
    with pytest.raises(ValueError):select_curve_windows([],5,1.1)

def test_study_selection_matches_production():
    from scripts.short_event_study import rerank
    from queryseek.retrieval.curve_spans import select_curve_windows
    from queryseek.schemas import TimeWindow
    raw=[[0,20,3],[0,18,2],[40,45,1]]
    expected=[[w.start_sec,w.end_sec,w.score] for w in select_curve_windows([TimeWindow(start_sec=a,end_sec=b,score=sc) for a,b,sc in raw])]
    assert rerank(raw,diversity=.9)==expected
