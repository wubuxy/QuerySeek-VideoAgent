"""语义检索测试（Mock 后端，离线确定性）。"""

from __future__ import annotations

import pytest

from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.ingest.pipeline import ingest, load_index
from queryseek.retrieval.semantic import semantic_search
from queryseek.schemas import TimeWindow
from queryseek.tools.validate import iou

GT = {
    "架构": TimeWindow(start_sec=12.0, end_sec=24.0),
    "检索": TimeWindow(start_sec=24.0, end_sec=36.0),
    "export": TimeWindow(start_sec=48.0, end_sec=60.0),
}


def _setup(qs_env, demo, vid="demo"):
    cfg = Config.load().resolved()
    cfg.embedding.backend = "mock"  # 测试走离线 Mock 后端
    cfg.retrieval.semantic.min_sim = 0.3  # 单桶碰撞相似度 ~0.18 < 0.3，防误检
    ingest(demo["video"], video_id=vid, srt_path=demo["srt"], config=cfg)
    idx = load_index(vid, cfg)
    embedder = MockEmbedding(idx.content_hints, idx.frames)
    return cfg, idx, embedder


def test_top1_hits_gt(qs_env, demo):
    cfg, idx, embedder = _setup(qs_env, demo)
    res = semantic_search(idx.video_id, "寻找讲解系统架构的片段", idx.frames,
                          idx.asset.duration_sec, embedder, cfg)
    assert res.windows, res.warnings
    assert iou(res.windows[0], GT["架构"]) >= 0.5
    assert res.windows[0].score > 0.5  # 相关帧簇的平均相似度高


def test_negative_query_empty(qs_env, demo):
    cfg, idx, embedder = _setup(qs_env, demo)
    res = semantic_search(idx.video_id, "寻找关于美食烹饪的片段", idx.frames,
                          idx.asset.duration_sec, embedder, cfg)
    assert res.windows == []
    assert res.warnings  # 证据不足


def test_english_query(qs_env, demo):
    cfg, idx, embedder = _setup(qs_env, demo)
    res = semantic_search(idx.video_id, "Where is the export pipeline demonstrated?",
                          idx.frames, idx.asset.duration_sec, embedder, cfg)
    assert res.windows
    assert iou(res.windows[0], GT["export"]) >= 0.5


def test_frame_count_mismatch_raises(qs_env, demo):
    import numpy as np

    cfg, idx, embedder = _setup(qs_env, demo)
    with pytest.raises(RuntimeError):
        semantic_search(idx.video_id, "q", idx.frames, idx.asset.duration_sec,
                        embedder, cfg, frame_embeddings=np.zeros((3, 256)))


def test_backend_mismatch_guard(qs_env, demo):
    from queryseek.embeddings.cache import check_backend

    check_backend({"backend": "openclip"}, "openclip")  # 一致：通过
    with pytest.raises(RuntimeError):
        check_backend({"backend": "openclip"}, "mock")
