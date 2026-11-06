import numpy as np
import pytest
pytest.importorskip('torch')
pytest.importorskip('scipy')
pytest.importorskip('sklearn')
from scripts.replay_localizer_curves import curve_predictions


def test_decoder_interchange_uses_only_timeline_scores_and_legal_seconds():
    record={'times':np.arange(12.),'duration_sec':12.}
    scores=np.array([0,0,0,0,1,2,3,2,1,0,0,0.],dtype=float)
    first,pool=curve_predictions(record,scores)
    other,_=curve_predictions({**record,'entry':{'relevant_windows':[[999,1000]]}},scores)
    assert first==other and first and pool
    assert len(first)<=5 and all(0<=a<b<=12 for a,b,_ in pool)
    with pytest.raises(ValueError): curve_predictions(record,np.full(12,np.nan))
