"""检索测试：subtitle（BM25 基线）+ uniform（E0 视觉基线，Mock VLM）。"""

from __future__ import annotations

from queryseek.config import Config
from queryseek.ingest.pipeline import ingest, load_index
from queryseek.models.mock import MockVLM
from queryseek.retrieval.subtitle import subtitle_search
from queryseek.tokenize import tokenize_query
from queryseek.retrieval.uniform import select_budget_frames, uniform_search
from queryseek.schemas import TimeWindow
from queryseek.tools.validate import iou

GT = {
    "架构": TimeWindow(start_sec=12.0, end_sec=24.0),
    "检索": TimeWindow(start_sec=24.0, end_sec=36.0),
    "智能体": TimeWindow(start_sec=36.0, end_sec=48.0),
    "export": TimeWindow(start_sec=48.0, end_sec=60.0),
}


def _ingest(qs_env, demo, vid="demo"):
    cfg = Config.load().resolved()
    ingest(demo["video"], video_id=vid, srt_path=demo["srt"], config=cfg)
    return cfg, load_index(vid, cfg)


def _uniform(qs_env, demo, query, budget, vid="demo", hints=None):
    cfg, idx = _ingest(qs_env, demo, vid=vid)
    vlm = MockVLM(hints if hints is not None else idx.content_hints)
    res = uniform_search(idx.video_id, query, idx.frames, idx.asset.duration_sec, vlm, cfg, budget_frames=budget)
    return idx, res


class TestTokenize:
    def test_cjk_bigram(self):
        toks = tokenize_query("寻找讲解系统架构的片段")
        assert "架构" in toks and "系统" in toks and "讲解" in toks
        assert "寻找" not in toks and "片段" not in toks  # 停用词
        assert toks == list(dict.fromkeys(toks))  # 去重

    def test_english(self):
        toks = tokenize_query("Where is the export pipeline demonstrated?")
        assert "export" in toks and "pipeline" in toks and "demonstrated" in toks
        assert "the" not in toks and "where" not in toks

    def test_all_stopwords(self):
        assert tokenize_query("寻找 片段 的 是") == []


class TestSubtitleSearch:
    def test_top1_hits_gt(self, qs_env, demo):
        cfg, idx = _ingest(qs_env, demo)
        res = subtitle_search(idx.video_id, "寻找讲解系统架构的片段", idx.subtitles, idx.asset.duration_sec, cfg)
        assert res.windows
        assert iou(res.windows[0], GT["架构"]) >= 0.5
        assert res.evidence, "应附带字幕证据"

    def test_english_query(self, qs_env, demo):
        cfg, idx = _ingest(qs_env, demo)
        res = subtitle_search(idx.video_id, "Where is the export pipeline demonstrated?", idx.subtitles, idx.asset.duration_sec, cfg)
        assert res.windows
        assert iou(res.windows[0], GT["export"]) >= 0.5

    def test_no_subtitles_graceful(self, qs_env, demo):
        cfg = Config.load().resolved()
        ingest(demo["video"], video_id="noasr", config=cfg)  # 无 SRT，无 whisper
        idx = load_index("noasr", cfg)
        res = subtitle_search(idx.video_id, "任意查询", idx.subtitles, idx.asset.duration_sec, cfg)
        assert res.windows == []
        assert res.warnings


class TestUniformSearch:
    def test_budget_16_recovers_gt(self, qs_env, demo):
        _idx, res = _uniform(qs_env, demo, "寻找讲解系统架构的片段", budget=16)
        assert res.windows, res.warnings
        assert iou(res.windows[0], GT["架构"]) >= 0.5
        assert len(res.used_frame_ids) == 16  # 预算生效
        assert res.vlm_calls >= 1
        assert all(0 <= w.start_sec < w.end_sec <= 60.0 for w in res.windows)

    def test_budget_is_respected(self, qs_env, demo):
        _idx, res = _uniform(qs_env, demo, "寻找讲解系统架构的片段", budget=8)
        assert len(res.used_frame_ids) == 8
        assert res.windows  # 仍能找到内容
        assert iou(res.windows[0], GT["架构"]) > 0.0

    def test_negative_query_no_windows(self, qs_env, demo):
        _idx, res = _uniform(qs_env, demo, "寻找关于美食烹饪的片段", budget=16)
        assert res.windows == []
        assert res.warnings  # 证据不足

    def test_no_hints_returns_nothing(self, qs_env, demo):
        cfg, idx = _ingest(qs_env, demo)
        vlm = MockVLM(None)  # 显式无提示：Mock 判全部不相关
        res = uniform_search(idx.video_id, "寻找讲解系统架构的片段", idx.frames,
                             idx.asset.duration_sec, vlm, cfg, budget_frames=16)
        assert res.windows == []
        assert res.warnings

    def test_select_budget_frames(self):
        from queryseek.schemas import VideoFrame

        frames = [VideoFrame(frame_id=f"f{i}", timestamp_sec=float(i), image_path=f"p{i}.jpg") for i in range(10)]
        assert len(select_budget_frames(frames, 4)) == 4
        assert select_budget_frames(frames, 1) == [frames[5]]  # 中位帧
        assert len(select_budget_frames(frames, 100)) == 10  # 超过索引则全取
        assert select_budget_frames([], 5) == []
