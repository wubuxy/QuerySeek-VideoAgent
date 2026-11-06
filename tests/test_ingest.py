"""ingest 集成测试：真实 FFmpeg/PySceneDetect 流程 + 缓存 + 注册表。"""

from __future__ import annotations

import pytest

from queryseek.config import Config
from queryseek.ingest.pipeline import ingest, load_index


def _cfg() -> Config:
    cfg = Config.load().resolved()
    cfg.embedding.backend = "mock"  # 测试离线化：不触发 OpenCLIP 权重下载
    return cfg


def test_ingest_and_cache_hit(qs_env, demo):
    cfg = _cfg()
    idx = ingest(demo["video"], video_id="demo1", srt_path=demo["srt"], config=cfg)
    assert not idx.cache_hit
    # 元信息（合成样本固定参数：60s / 320x240 / 12fps / 含音轨）
    assert abs(idx.asset.duration_sec - 60.0) < 2.0
    assert (idx.asset.width, idx.asset.height) == (320, 240)
    assert idx.asset.has_audio
    assert len(idx.asset.sha256) == 64
    # 5 段纯色卡片 => 镜头数在 [4, 7]（允许切分器在过渡处有微小噪声）
    assert 4 <= len(idx.scenes) <= 7
    # 0.5fps 粗抽帧：>= 25 帧
    assert len(idx.frames) >= 25
    # SRT 5 段
    assert len(idx.subtitles) == 5
    # labels 被复制为 content hints（供 Mock VLM）
    assert len(idx.content_hints) == 5

    # 第二次 ingest：缓存命中，产物一致
    idx2 = ingest(demo["video"], video_id="demo1", srt_path=demo["srt"], config=cfg)
    assert idx2.cache_hit
    assert len(idx2.frames) == len(idx.frames)
    assert idx2.cache_key == idx.cache_key


def test_second_video_id_reuses_cache(qs_env, demo):
    cfg = _cfg()
    ingest(demo["video"], video_id="demo_a", srt_path=demo["srt"], config=cfg)
    idx = ingest(demo["video"], video_id="demo_b", srt_path=demo["srt"], config=cfg)
    assert idx.cache_hit  # 同 sha256 + 同配置 => 复用缓存


def test_load_index_and_registry(qs_env, demo):
    cfg = _cfg()
    ingest(demo["video"], video_id="demo1", srt_path=demo["srt"], config=cfg)
    idx = load_index("demo1", cfg)
    assert idx.video_id == "demo1"
    assert idx.frames
    with pytest.raises(KeyError):
        load_index("not_registered", cfg)


def test_config_change_invalidates_cache(qs_env, demo):
    cfg = _cfg()
    ingest(demo["video"], video_id="demo1", srt_path=demo["srt"], config=cfg)
    cfg2 = _cfg()
    cfg2.video.coarse_fps = 1.0  # 抽帧配置变化 => 缓存键变化
    idx = ingest(demo["video"], video_id="demo1", srt_path=demo["srt"], config=cfg2)
    assert not idx.cache_hit


def test_ingest_without_srt_no_crash(qs_env, demo):
    """无 SRT 且无 faster-whisper：字幕为空但 ingest 成功（说明书 §12.1）。"""
    cfg = _cfg()
    idx = ingest(demo["video"], video_id="demo_noasr", config=cfg)
    assert not idx.cache_hit
    assert idx.frames
    assert idx.subtitles == []  # faster-whisper 未安装 => 自动跳过


def test_video_id_sanitize(qs_env, demo):
    from queryseek.ingest.registry import sanitize_video_id

    assert sanitize_video_id("my_video-01") == "my_video-01"
    # QVHighlights 的 vid 含小数点（{youtube_id}_{start}_{end}）
    assert sanitize_video_id("7QNJyAIprhg_60.0_210.0") == "7QNJyAIprhg_60.0_210.0"
    for bad in ("../evil", "a/b", "a\\b", "..", "视频", "", "x" * 65):
        with pytest.raises(ValueError):
            sanitize_video_id(bad)
