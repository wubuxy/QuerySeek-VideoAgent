import json
import io
import time
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip('torch')

from queryseek.retrieval.univtg_inputs import text_hidden_states, valid_text_features, upstream_temporal_inputs, numeric_difference
from scripts.univtg_source_audit import identity, audit_rows, bounded_download


class TinyClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_final = torch.nn.LayerNorm(4)
        self.embed = torch.nn.Embedding(12,4)
        self.projection = torch.nn.Linear(4,4,bias=False)
    def encode_text(self,tokens):
        h = self.ln_final(self.embed(tokens))
        return self.projection(h[:,0])


def test_capture_native_ln_final_hidden_states_not_pool_and_cleanup_hook():
    torch.manual_seed(1); model = TinyClip().eval(); tokens = torch.tensor([[9,1,10,0],[9,2,3,10]])
    hidden,pool = text_hidden_states(model,tokens)
    assert hidden.shape == (2,4,4) and pool.shape==(2,4)
    assert not model.ln_final._forward_hooks
    torch.testing.assert_close(hidden,model.ln_final(model.embed(tokens)))
    features = valid_text_features(hidden,tokens,eos_id=10)
    assert [len(h) for h in features]==[3,4]
    assert all(torch.allclose(h.norm(dim=-1),torch.ones(len(h))) for h in features)


def test_length_policy_preserves_zero_vocabulary_token_and_exposes_upstream_difference():
    hidden = torch.randn(1,5,4); tokens = torch.tensor([[9,0,2,10,0]])
    assert len(valid_text_features(hidden,tokens,eos_id=10)[0])==4
    assert len(valid_text_features(hidden,tokens,eos_id=10,length_policy='upstream_nonzero')[0])==3
    with pytest.raises(ValueError,match='Missing EOT'): valid_text_features(hidden,torch.zeros_like(tokens))
    with pytest.raises(ValueError): valid_text_features(hidden,tokens,length_policy='guess')


def test_temporal_inputs_reproduce_published_tef_and_anchor_without_fixing_it():
    tef,anchor = upstream_temporal_inputs(3,2.)
    torch.testing.assert_close(tef,torch.tensor([[0,1/3],[1/3,2/3],[2/3,1.]]))
    torch.testing.assert_close(anchor[:,0],torch.tensor([1/3,2/3,1.]))
    # Published (i+clip_length/2)/L is not (i+0.5)/L; keep this distinction explicit.
    assert not torch.allclose(anchor[:,0],(torch.arange(3)+.5)/3)
    for count,length in [(0,2.),(True,2.),(3,float('nan'))]:
        with pytest.raises(ValueError): upstream_temporal_inputs(count,length)


def test_complete_identity_intersection_and_incomplete_scope(tmp_path):
    source='Ab_cd-Ef123'
    p=tmp_path/'rows.jsonl'; p.write_text(json.dumps({'vid':source+'_0.0_10.0','query':'ignored'})+'\n')
    report,ids=audit_rows(p,{'validation':{source}})
    assert ids=={source} and report['overlaps']['validation']['sources']==1
    assert report['complete_identity_parse']
    with p.open('a') as f:f.write(json.dumps({'vid':'unrecognized'})+'\n')
    report,_=audit_rows(p,{'validation':{source}})
    assert not report['interval_subset_negative_claim_allowed'] and report['invalid_rows']==1
    assert 'ignored' not in json.dumps(report)
    with pytest.raises(ValueError):identity('../source')


@pytest.mark.parametrize('mode',['html','oversize','empty','deadline'])
def test_metadata_transport_caps_and_html_rejection(tmp_path,monkeypatch,mode):
    import scripts.univtg_source_audit as audit
    class Response(io.BytesIO):
        headers = {'Content-Type':'text/html' if mode=='html' else 'application/octet-stream'}
    monkeypatch.setattr(audit.urllib.request,'urlopen',lambda *args,**kwargs:Response(b'' if mode=='empty' else b'abcdef'))
    with pytest.raises((ValueError,TimeoutError)):
        bounded_download('https://example.invalid',tmp_path/'metadata',4,1,0 if mode=='deadline' else time.perf_counter()+1)


def test_numeric_comparison_rejects_shapes_and_nonfinite():
    assert numeric_difference([1,2],[1,3])['max_abs']==1
    with pytest.raises(ValueError):numeric_difference([1],[1,2])
    with pytest.raises(ValueError):numeric_difference([float('nan')],[0])


def test_source_pts_are_distinguished_from_fps_output_grid():
    from scripts.univtg_input_audit import parse_showinfo
    lines = b'[showinfo@src @ 0x1] n: 29 pts: 29000 pts_time:0.966667 checksum:ABCD1234\n[showinfo@sample @ 0x2] n: 0 pts: 0 pts_time:0 checksum:ABCD1234\n'
    source=parse_showinfo(lines,'src'); output=parse_showinfo(lines,'sample')
    assert source[0]['source_or_output_pts_sec']==.966667
    assert output[0]['source_or_output_pts_sec']==0.
    assert source[0]['checksum']==output[0]['checksum']


def test_generated_fixture_supports_bounded_official_pixel_audit(demo, tmp_path, monkeypatch):
    import scripts.univtg_input_audit as audit
    # Keep production output confinement while using an isolated pytest workspace.
    monkeypatch.setattr(audit,'checked_output',lambda p:Path(p))
    pixels,phase=audit.sample_pixels(demo['video'],320,240,3,tmp_path/'pixels')
    assert pixels.shape==(3,224,224,3) and pixels.dtype==np.uint8
    assert [p['official_output_grid_sec'] for p in phase]==[0.,2.,4.]
    assert all(p['source_pts_candidates_sec'] for p in phase)
    assert (tmp_path/'pixels/ffmpeg_showinfo.log').exists()
