"""Embedding 层测试：Mock 确定性 + 语义可分性 + OpenCLIP（可选，已安装时运行）。"""

from __future__ import annotations

import numpy as np
import pytest

from queryseek.embeddings.base import MockEmbedding
from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
from queryseek.schemas import VideoFrame

HINTS = [
    {"start_sec": 0.0, "end_sec": 12.0, "label": "intro welcome 欢迎"},
    {"start_sec": 12.0, "end_sec": 24.0, "label": "architecture 系统架构 讲解"},
    {"start_sec": 24.0, "end_sec": 36.0, "label": "retrieval 语义检索 demo"},
]

FRAMES = [
    VideoFrame(frame_id="f0", timestamp_sec=2.0, image_path="p0.jpg"),
    VideoFrame(frame_id="f1", timestamp_sec=18.0, image_path="p1.jpg"),
    VideoFrame(frame_id="f2", timestamp_sec=30.0, image_path="p2.jpg"),
]


class TestMockEmbedding:
    def test_deterministic(self):
        e = MockEmbedding(HINTS, FRAMES)
        a = e.encode_texts(["寻找讲解系统架构的片段"])
        b = e.encode_texts(["寻找讲解系统架构的片段"])
        assert np.array_equal(a, b)
        assert a.shape == (1, e.dim)

    def test_unit_norm(self):
        e = MockEmbedding(HINTS, FRAMES)
        v = e.encode_texts(["hello world"])[0]
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6

    def test_semantic_separation(self):
        """语义相关帧的相似度必须显著高于无关帧。"""
        e = MockEmbedding(HINTS, FRAMES)
        q = e.encode_texts(["寻找讲解系统架构的片段"])[0]
        img = e.encode_images(["p0.jpg", "p1.jpg", "p2.jpg"])
        sims = img @ q
        assert sims[1] > 0.5  # 架构帧
        assert sims[0] < 0.2 and sims[2] < 0.2  # 无关帧（允许偶然低相似）
        assert sims[1] > sims[0] + 0.3

    def test_no_hints_zero_vectors(self):
        e = MockEmbedding(None, FRAMES)
        img = e.encode_images(["p0.jpg"])
        q = e.encode_texts(["any query"])[0]
        assert float((img @ q)[0]) == 0.0


@pytest.mark.skipif(not OpenCLIPEmbedding.is_available(), reason="open_clip/torch 未安装")
class TestOpenCLIPOptional:
    def test_load_and_encode(self):
        """真实 OpenCLIP：模型加载 + 文本/图像编码（首次运行会下载权重，可能较慢）。"""
        e = OpenCLIPEmbedding()
        t = e.encode_texts(["a photo of architecture", "a photo of food"])
        assert t.shape == (2, e.dim)
        sim = float(t[0] @ t[1])
        assert -1.0 <= sim <= 1.0
        # 不同语义文本相似度应低于相同文本
        same = float(e.encode_texts(["a photo of architecture"])[0] @ t[0])
        assert same > sim
