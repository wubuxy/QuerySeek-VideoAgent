"""导出测试：FFmpeg 结构化剪辑 + ffprobe 验证 + manifest + 路径安全。"""

from __future__ import annotations

import json

import pytest

from queryseek.config import Config
from queryseek.ingest.pipeline import ingest, load_index
from queryseek.schemas import TimeWindow
from queryseek.tools.export import export_clip, sanitize_job_id


def _ingest(qs_env, demo, vid="demo"):
    cfg = Config.load().resolved()
    ingest(demo["video"], video_id=vid, srt_path=demo["srt"], config=cfg)
    return cfg, load_index(vid, cfg)


def test_export_clip_valid(qs_env, demo):
    cfg, idx = _ingest(qs_env, demo)
    windows = [
        TimeWindow(start_sec=12.0, end_sec=24.0, score=1.0, evidence_ids=["a"]),
        TimeWindow(start_sec=36.0, end_sec=48.0, score=0.8, evidence_ids=["b"]),
    ]
    out, manifest = export_clip(idx.asset, windows, cfg.output_root, cfg)
    assert out.is_file() and out.stat().st_size > 1000
    # 时长 = 两段之和（12+12），容忍 3s
    assert abs(manifest.output_duration_sec - 24.0) <= cfg.export.target_duration_tolerance_sec
    assert len(manifest.segments) == 2
    # manifest 的源区间映射正确
    assert (manifest.segments[0].source_start_sec, manifest.segments[0].source_end_sec) == (12.0, 24.0)
    assert manifest.source_sha256 == idx.asset.sha256
    # 输出限制在 outputs/<job_id>/ 内
    assert str(cfg.output_root) in str(out)


def test_export_target_seconds_greedy(qs_env, demo):
    cfg, idx = _ingest(qs_env, demo)
    windows = [
        TimeWindow(start_sec=12.0, end_sec=24.0, score=1.0),
        TimeWindow(start_sec=36.0, end_sec=48.0, score=0.8),
    ]
    out, manifest = export_clip(idx.asset, windows, cfg.output_root, cfg, target_seconds=10.0)
    # 贪心：第一个窗口 12s 已满足 10s 目标
    assert len(manifest.segments) == 1
    assert abs(manifest.output_duration_sec - 12.0) <= 3.0
    assert out.is_file()


def test_export_no_windows_raises(qs_env, demo):
    cfg, idx = _ingest(qs_env, demo)
    with pytest.raises(RuntimeError):
        export_clip(idx.asset, [], cfg.output_root, cfg)


def test_job_id_sanitize():
    assert sanitize_job_id("job_20261009_001_abc123")
    for bad in ("../evil", "a/b", "job name", ""):
        with pytest.raises(ValueError):
            sanitize_job_id(bad)


def test_export_overlapping_windows_merged(qs_env, demo):
    """重叠窗口在导出前必须被合并（说明书 §12.3）。"""
    cfg, idx = _ingest(qs_env, demo)
    windows = [
        TimeWindow(start_sec=10.0, end_sec=16.0, score=0.9),
        TimeWindow(start_sec=14.0, end_sec=20.0, score=0.7),
    ]
    out, manifest = export_clip(idx.asset, windows, cfg.output_root, cfg)
    assert len(manifest.segments) == 1
    assert abs(manifest.output_duration_sec - 10.0) <= 3.0
    assert out.is_file()


def test_manifest_json_roundtrip(qs_env, demo):
    cfg, idx = _ingest(qs_env, demo)
    out, manifest = export_clip(idx.asset, [TimeWindow(start_sec=0.0, end_sec=6.0)], cfg.output_root, cfg)
    from queryseek.schemas import ExportManifest

    parsed = ExportManifest.model_validate(json.loads(manifest.model_dump_json()))
    assert parsed.job_id == manifest.job_id
    assert parsed.segments
