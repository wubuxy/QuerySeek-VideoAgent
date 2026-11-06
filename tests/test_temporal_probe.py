from scripts.temporal_probe import proposals, grow, select_frames, validate_scores
from queryseek.schemas import TimeWindow, VideoFrame
import pytest

def f(i,t): return VideoFrame(frame_id=str(i),timestamp_sec=t,image_path='unused.jpg')
def test_proposals_clamped():
    assert all(0<=a<b<=10 for a,b in proposals(TimeWindow(start_sec=1,end_sec=4),10))
def test_selection_unique_budget_order():
    fs=[f(i,i) for i in range(40)]; ws=[TimeWindow(start_sec=8,end_sec=16),TimeWindow(start_sec=12,end_sec=22)]
    picked=select_frames(fs,ws)
    assert len(picked)<=16 and len({x.frame_id for x in picked})==len(picked)
    assert [x.timestamp_sec for x in picked]==sorted(x.timestamp_sec for x in picked)
def test_grow_stops_at_negative():
    w=TimeWindow(start_sec=8,end_sec=12)
    assert grow(w,[f(0,0),f(1,4),f(2,8),f(3,12),f(4,16)],{'0','2','4'},20)==(8,12,1)
def test_no_inside_support():
    assert grow(TimeWindow(start_sec=8,end_sec=12),[f(1,4),f(2,8)],{'1'},20) is None
@pytest.mark.parametrize('scores',[[1],[float('nan'),0],[True,0],[-1,0]])
def test_bad_scores_rejected(scores):
    with pytest.raises(ValueError): validate_scores({'scores':scores},2)
def test_valid_scores(): assert validate_scores({'scores':[0,1]},2)==[0.,1.]
