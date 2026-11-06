import math
import subprocess
import sys
from pathlib import Path

import pytest

from queryseek.ingest.frames import source_timestamps


def test_pts_parser_accepts_real_showinfo_precision_and_rejects_missing_lines():
    data=b"[Parsed_showinfo_2 @ 000] n:   0 pts: 1 pts_time:0.0833333 duration:1\n[Parsed_showinfo_2] n: 1 pts: 2 pts_time:4.001e+0\n"
    assert source_timestamps(data)==[(0,.0833333),(1,4.001)]
    assert source_timestamps(b"unrelated pts_time:0")==[]


def test_timing_fixture_matches_image_content(has_ffmpeg,tmp_path,monkeypatch):
    if not has_ffmpeg:pytest.skip("FFmpeg unavailable")
    import scripts.audit_frame_timing as audit
    import scripts.sampling_probe as sampling
    monkeypatch.setattr(sampling,"ROOT",tmp_path)
    result=audit.audit(tmp_path/"outputs/timing")
    assert result["groups"]["legacy_uniform"]["max_abs_error_sec"]>1.8
    for group in ("current_uniform","current_local","scene_seek"):
        # MKV's millisecond timebase + showinfo precision, independent image-based match.
        assert result["groups"][group]["max_abs_error_sec"]<.002
    assert [r["source_frame_index"] for r in result["groups"]["current_uniform"]["frames"]]==[0,48]


def test_cache_namespace_preserves_v1(qs_env):
    from queryseek.ingest.cache import cache_dir_for
    from queryseek import PIPELINE_VERSION
    original=qs_env/"cache"/("a"*64);original.mkdir(parents=True)
    (original/"legacy.txt").write_text("keep")
    current=cache_dir_for(qs_env/"cache","a"*64)
    assert current!=original and f"pipeline_{PIPELINE_VERSION}" in str(current)
    assert (original/"legacy.txt").read_text()=="keep"


def test_openai_activation_policy_is_explicit_and_cache_identity_changes(monkeypatch):
    from types import SimpleNamespace
    from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
    flags=[]
    def fake_load(self):
        flags.append(self.force_quick_gelu)
        return SimpleNamespace(visual=SimpleNamespace(output_dim=512)),None
    monkeypatch.setattr(OpenCLIPEmbedding,"_load",fake_load)
    correct=OpenCLIPEmbedding()
    historical=OpenCLIPEmbedding(force_quick_gelu=False)
    other=OpenCLIPEmbedding(pretrained="laion2b_s34b_b79k")
    assert flags==[True,False,False]
    assert correct.model_id!=historical.model_id and "quickgelu" in correct.model_id


def test_local_output_tag_and_intervals_are_validated_before_ffmpeg(tmp_path):
    from queryseek.ingest.frames import extract_local_frames
    for start,end,fps,tag in ((-1,2,1,"x"),(2,1,1,"x"),(0,2,0,"x"),(0,2,1,"../bad")):
        with pytest.raises(ValueError):extract_local_frames("missing",tmp_path,start,end,fps,tag=tag)


def test_scene_missing_pts_preserves_log_without_fabricating_time(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from queryseek.ingest import frames
    from queryseek.schemas import Scene
    monkeypatch.setattr(frames, '_require_ffmpeg', lambda: 'ffmpeg')
    def command(args, **kwargs):
        Path(args[-1]).touch()
        return SimpleNamespace(returncode=0, stderr=b'no source timestamp available')
    monkeypatch.setattr(frames.subprocess, 'run', command)
    with pytest.raises(RuntimeError, match='no source PTS'):
        frames.extract_scene_frames('unused.mp4', tmp_path, [Scene(scene_id='x',start_sec=1.,end_sec=2.)])
    assert (tmp_path/'s_000001.ffmpeg.stderr.log').read_bytes() == b'no source timestamp available'
