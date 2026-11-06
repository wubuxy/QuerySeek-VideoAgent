import numpy as np
import pytest
from scripts import repaired_pool_ranker as ranker


def test_freeze_requires_prevalidation_generator_choice(tmp_path,monkeypatch):
    monkeypatch.setattr(ranker,'ROOT',tmp_path);up=tmp_path/'up';up.mkdir();(up/'metrics.json').write_text('{}')
    with pytest.raises(ValueError,match='before upstream'):ranker.freeze({'upstream_run':'up'},tmp_path/'out')


def test_prediction_receives_features_and_geometry_only():
    class Model:
        def predict(self,x):return x[:,0]
    table={1:{'windows':np.array([[0.,2.,.5],[4.,6.,.4]]),'features':np.array([[1.]*14,[2.]*14]),'target':'must not be read'}}
    result=ranker.predict(Model(),table,'full',{'top_k':5,'nms_iou':.7})
    assert result[1][0][:2]==(4.,6.)
