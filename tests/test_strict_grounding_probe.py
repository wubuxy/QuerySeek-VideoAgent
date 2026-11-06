from scripts.strict_grounding_probe import parse_labels
from queryseek.schemas import VideoFrame
import pytest
f=VideoFrame(frame_id='a',timestamp_sec=1,image_path='unused')
def test_strict_complete():assert parse_labels({'observations':[{'frame_id':'a','match':'uncertain'}]},[f])['a']['match']=='uncertain'
@pytest.mark.parametrize('rows',[[],[{'frame_id':'x','match':'yes'}],[{'frame_id':'a','match':'maybe'}],[{'frame_id':'a','match':'yes'},{'frame_id':'a','match':'no'}]])
def test_strict_reject(rows):
    with pytest.raises(ValueError):parse_labels({'observations':rows},[f])

def test_frame_keyed_response():
    assert parse_labels({'a':{'match':'no'}},[f])['a']['match']=='no'
