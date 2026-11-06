from types import SimpleNamespace
import pytest
from queryseek.schemas import TimeWindow, VideoFrame, SearchResult
from queryseek.retrieval.visual_spans import validated_adjustment, verify_curve_result


def observations():
    return [VideoFrame(frame_id=str(i), timestamp_sec=t, image_path="unused")
            for i, t in enumerate((2., 5., 8., 11.))]


def reply(**changes):
    return dict(verdict="confirmed", confidence=.9, start_sec=2., end_sec=12.,
                evidence_frame_ids=["0", "1"], **changes)


@pytest.mark.parametrize("bounds", [(2, 12), (5, 8)])
def test_expand_and_contract(bounds):
    obj = reply(); obj.update(start_sec=bounds[0], end_sec=bounds[1], evidence_frame_ids=["1"])
    result = validated_adjustment(obj, TimeWindow(start_sec=4,end_sec=10,score=3),
                                  observations(), TimeWindow(start_sec=0,end_sec=14))
    assert (result.start_sec,result.end_sec)==bounds and result.score==3


@pytest.mark.parametrize("changes", [{"start_sec":-1},{"end_sec":99},
    {"evidence_frame_ids":["invented"]},{"confidence":float("nan")},
    {"start_sec":True},{"start_sec":9,"end_sec":10}])
def test_invalid_visual_output(changes):
    obj=reply();obj.update(changes)
    with pytest.raises(ValueError):
        validated_adjustment(obj,TimeWindow(start_sec=4,end_sec=10),observations(),TimeWindow(start_sec=0,end_sec=14))


def test_insufficient_retains_original():
    original=TimeWindow(start_sec=4,end_sec=10)
    obj=reply();obj["verdict"]="insufficient"
    assert validated_adjustment(obj,original,observations(),TimeWindow(start_sec=0,end_sec=14)) is original


def test_failure_fallback_and_global_budget(monkeypatch):
    from queryseek.retrieval import visual_spans
    monkeypatch.setattr(visual_spans,"extract_local_frames",lambda *a,**k:observations())
    class VLM:
        max_frames_per_call=4
        def verify_span(self,*a):raise TimeoutError("do not log provider content")
    ws=[TimeWindow(start_sec=4,end_sec=10,score=3),TimeWindow(start_sec=5,end_sec=11,score=2)]
    base=SearchResult(video_id="v",query="event",method="clip_curve",windows=ws)
    idx=SimpleNamespace(asset=SimpleNamespace(path="unused",duration_sec=14))
    out=verify_curve_result(base,idx,VLM(),"outputs/test_visual",budget_frames=4)
    assert out.windows==ws and out.vlm_calls==1 and len(out.used_frame_ids)==4
    assert out.refinement_rounds[0]["error_type"]=="TimeoutError"
    assert "provider content" not in str(out)


def test_output_path_guard():
    with pytest.raises(ValueError):
        verify_curve_result(None,None,None,"outputs/../../outside")


def test_short_peaks_are_bounded_and_unsmoothed():
    from queryseek.retrieval.curve_spans import short_peak_spans
    ws=short_peak_spans([0,4,8,12],[0,1,0,0],14)
    assert len(ws)==3 and all(0<=w.start_sec<w.end_sec<=14 for w in ws)
    assert {w.duration_sec for w in ws}=={2,4,8}


def test_api_span_sends_images_and_thinking_flag(tmp_path):
    import json
    import httpx
    from queryseek.models.compatible_api import CompatibleAPIVLM
    captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(reply())}}]})
    img=tmp_path/"sample.jpg";img.write_bytes(b"fixture image bytes")
    frames=[f.model_copy(update={"image_path":str(img)}) for f in observations()]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        vlm=CompatibleAPIVLM("test-key","https://test.invalid/v1","test-model",
                              client=client,enable_thinking=False,max_retries=0)
        obj=vlm.verify_span("opens door",TimeWindow(start_sec=4,end_sec=10),frames,
                            TimeWindow(start_sec=0,end_sec=14))
    payload=captured[0];parts=payload["messages"][1]["content"]
    assert payload["enable_thinking"] is False and obj["verdict"]=="confirmed"
    assert sum(p["type"]=="image_url" for p in parts)==4
    assert "0 @ 2.000s" in parts[0]["text"]


@pytest.mark.parametrize("method", ["E4V", "E5"])
def test_visual_runner_mock_uses_visual_path(monkeypatch, method):
    from queryseek.eval import runner
    from queryseek.config import Config
    from queryseek.retrieval import visual_spans
    monkeypatch.setattr(visual_spans,"extract_local_frames",lambda *a,**k:observations())
    idx=SimpleNamespace(asset=SimpleNamespace(path="unused",duration_sec=14),
                        frames=observations(),content_hints=[],video_id="v")
    sample=SimpleNamespace(query="opens door",qid=0)
    result=runner._run_method(Config.load(),idx,sample,method,4,True)
    assert result.windows and result.vlm_calls==1 and result.used_frames==4


def test_original_visual_control_disables_short_candidates(monkeypatch):
    from queryseek.eval import runner
    from queryseek.config import Config
    from queryseek.retrieval import visual_spans, curve_spans
    observed=[]
    def search(*a,**kwargs):
        observed.append(kwargs["include_short"])
        return SearchResult(video_id="v",query="event",method="clip_curve",windows=[])
    monkeypatch.setattr(curve_spans,"curve_search",search)
    monkeypatch.setattr(visual_spans,"verify_curve_result",lambda result,*a,**kw:result)
    idx=SimpleNamespace(asset=SimpleNamespace(duration_sec=14),frames=observations(),content_hints=[],video_id="v")
    for method in ("E4V","E5"):
        runner._run_method(Config.load(),idx,SimpleNamespace(query="event",qid=0),method,16,True)
    assert observed==[False,True]
