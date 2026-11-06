import sys
from pathlib import Path
import pytest
from scripts import benchmark
from queryseek.config import Config

@pytest.mark.parametrize('mock,expected',[(False,.15),(True,.3)])
def test_benchmark_respects_real_threshold(monkeypatch,mock,expected):
    captured={}
    monkeypatch.setattr(benchmark,'ensure_dotenv',lambda:None)
    monkeypatch.setattr(benchmark,'load_dataset',lambda path:[])
    def fake_run(cfg,*args,**kwargs):
        captured['threshold']=cfg.retrieval.semantic.min_sim
        return {'run_id':'test','per_method':{}}
    monkeypatch.setattr(benchmark,'run_benchmark',fake_run)
    argv=['benchmark','--dataset','unused','--config',str(Path('configs/qvhighlights.yaml'))]
    if mock:argv.append('--mock')
    monkeypatch.setattr(sys,'argv',argv)
    benchmark.main()
    assert captured['threshold']==expected
