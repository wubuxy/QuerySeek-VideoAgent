"""Adaptive 检索测试（E2，Mock 后端离线确定性）：粗召回 → 粗观察 → 细化 → 预算守卫。"""

from __future__ import annotations

from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.ingest.pipeline import ingest, load_index
from queryseek.models.mock import MockVLM
from queryseek.retrieval.adaptive import (
    _WinState,
    _grow_window,
    adaptive_search,
    estimate_uncertainty,
)
from queryseek.schemas import Evidence, TimeWindow
from queryseek.tools.validate import iou

GT = TimeWindow(start_sec=12.0, end_sec=24.0)


def _setup(qs_env, demo, vid="demo"):
    cfg = Config.load().resolved()
    cfg.embedding.backend = "mock"
    cfg.retrieval.semantic.min_sim = 0.3
    ingest(demo["video"], video_id=vid, srt_path=demo["srt"], config=cfg)
    idx = load_index(vid, cfg)
    embedder = MockEmbedding(idx.content_hints, idx.frames)
    vlm = MockVLM(idx.content_hints)
    return cfg, idx, embedder, vlm


class TestEstimateUncertainty:
    def _cfg(self):
        return Config.load().resolved()

    def test_unobserved_wide_window_high(self):
        cfg = self._cfg()
        st = _WinState(window=TimeWindow(start_sec=0.0, end_sec=30.0), observed=0)
        assert estimate_uncertainty(st, 60.0, cfg) > 0.7

    def test_confirmed_narrow_observed_low(self):
        cfg = self._cfg()
        st = _WinState(
            window=TimeWindow(start_sec=10.0, end_sec=15.0),
            asr_support=True, visual_support=True, observed=5,
        )
        assert estimate_uncertainty(st, 60.0, cfg) < 0.3

    def test_single_modality_mid(self):
        cfg = self._cfg()
        st = _WinState(
            window=TimeWindow(start_sec=10.0, end_sec=15.0),
            asr_support=True, observed=5,
        )
        u = estimate_uncertainty(st, 60.0, cfg)
        assert 0.15 < u < 0.45


class TestWindowGrowth:
    """证据驱动的窗口生长（EXPERIMENTS.md §6 诊断 3b：62% 有重叠未命中=过窄）。"""

    def _setup(self, qs_env, demo):
        cfg = Config.load().resolved()
        cfg.embedding.backend = "mock"
        cfg.retrieval.semantic.min_sim = 0.3
        cfg.retrieval.adaptive.grow.enabled = True
        ingest(demo["video"], video_id="grow", srt_path=demo["srt"], config=cfg)
        idx = load_index("grow", cfg)
        vlm = MockVLM(idx.content_hints)
        return cfg, idx, vlm

    def _state(self, start, end, n_rel=2, observed=4):
        st = _WinState(
            window=TimeWindow(start_sec=start, end_sec=end, score=n_rel / observed),
            visual_support=True, observed=observed,
        )
        st.evidence = [
            Evidence(evidence_id=f"e{i}", type="frame", timestamp_sec=start + i,
                     source_file="x", model_id="mock", score=1.0)
            for i in range(n_rel)
        ]
        return st

    def _call(self, cfg, idx, vlm, st, query="寻找讲解系统架构的片段", used=None, budget=16):
        return _grow_window(
            st, idx.frames, idx.asset.duration_sec, vlm, query, cfg,
            used=list(used or []), frame_budget=budget, vlm_calls=0,
            max_vlm_calls=None, trace=None, method="adaptive",
        )

    def test_grows_right_into_relevant_adjacent(self, qs_env, demo):
        """窗口 [12,18) 右侧相邻帧仍属"架构"段 → 向右扩展；左侧是 intro → 不动。"""
        cfg, idx, vlm = self._setup(qs_env, demo)
        st = self._state(12.0, 18.0)
        used, calls, errs, log = self._call(cfg, idx, vlm, st)
        assert st.window.end_sec > 18.0 + 1.0  # 向右生长了
        assert st.window.start_sec == 12.0  # 左侧相邻帧不相关：不生长
        assert log is not None and log["round"] == -2
        assert used and calls >= 1  # 消耗了预算与调用
        assert st.score == (2 + 2) / (4 + len(used))  # (粗观察相关+生长相关)/总观察

    def test_no_growth_when_adjacent_irrelevant(self, qs_env, demo):
        """窗口已覆盖整个相关段：相邻帧属其他段 → 窗口不变、无生长日志。"""
        cfg, idx, vlm = self._setup(qs_env, demo)
        st = self._state(12.0, 24.0)
        used, calls, errs, log = self._call(cfg, idx, vlm, st)
        assert (st.window.start_sec, st.window.end_sec) == (12.0, 24.0)
        assert log is None
        assert used  # 相邻帧被观察（预算如实消耗），但未扩展

    def test_min_relevant_rate_gate(self, qs_env, demo):
        """粗观察相关率低于阈值 → 不生长、零消耗。"""
        cfg, idx, vlm = self._setup(qs_env, demo)
        st = self._state(12.0, 18.0, n_rel=1, observed=4)  # 0.25 < 0.5
        used, calls, errs, log = self._call(cfg, idx, vlm, st)
        assert used == [] and calls == 0 and log is None

    def test_frame_budget_cap(self, qs_env, demo):
        """帧预算只剩 2：生长最多观察 2 帧。"""
        cfg, idx, vlm = self._setup(qs_env, demo)
        st = self._state(12.0, 18.0)
        pre_used = [f"u{i}" for i in range(14)]
        used, calls, errs, log = self._call(cfg, idx, vlm, st, used=pre_used, budget=16)
        assert len(used) <= 2

    def test_disabled_by_default(self, qs_env, demo):
        """默认关闭：adaptive_search（E1 模式）不产生生长日志。"""
        cfg = Config.load().resolved()
        cfg.embedding.backend = "mock"
        cfg.retrieval.semantic.min_sim = 0.3
        ingest(demo["video"], video_id="nogrow", srt_path=demo["srt"], config=cfg)
        idx = load_index("nogrow", cfg)
        vlm = MockVLM(idx.content_hints)
        embedder = MockEmbedding(idx.content_hints, idx.frames)
        res = adaptive_search(
            idx.video_id, "寻找讲解系统架构的片段", idx, vlm, embedder, cfg,
            budget_frames=32, max_rounds=0,
        )
        assert not [r for r in res.refinement_rounds if r.get("round") == -2]


class TestAdaptiveSearch:
    def test_recovers_gt_with_refinement(self, qs_env, demo):
        cfg, idx, embedder, vlm = _setup(qs_env, demo)
        res = adaptive_search(
            idx.video_id, "寻找讲解系统架构的片段", idx, vlm, embedder, cfg,
            budget_frames=32, max_rounds=2,
        )
        assert res.windows, res.warnings
        assert iou(res.windows[0], GT) >= 0.5
        assert len(res.used_frame_ids) <= 32  # 预算硬约束
        # 细化发生过：轮次日志（round>=0，过滤剪枝记录）字段完整
        refines = [r for r in res.refinement_rounds if r.get("round", 0) >= 0]
        assert refines
        r0 = refines[0]
        for key in ("before_window", "sampled_frames", "after_window", "reason", "uncertainty"):
            assert key in r0
        # 细化确实收窄了窗口（before 宽度 >= after 宽度）
        b0, a0 = r0["before_window"], r0["after_window"]
        assert (a0[1] - a0[0]) <= (b0[1] - b0[0]) + 1e-6

    def test_zero_rounds_equals_e1_no_refine(self, qs_env, demo):
        """max_rounds=0 => E1（语义召回+粗观察，无细化轮次）。"""
        cfg, idx, embedder, vlm = _setup(qs_env, demo)
        res = adaptive_search(
            idx.video_id, "寻找讲解系统架构的片段", idx, vlm, embedder, cfg,
            budget_frames=32, max_rounds=0,
        )
        assert res.windows
        # 无细化轮次（剪枝记录 round=-1 不算细化）
        assert not [r for r in res.refinement_rounds if r.get("round", 0) >= 0]
        assert iou(res.windows[0], GT) >= 0.5

    def test_tiny_budget_flags_exhaustion(self, qs_env, demo):
        cfg, idx, embedder, vlm = _setup(qs_env, demo)
        res = adaptive_search(
            idx.video_id, "寻找讲解系统架构的片段", idx, vlm, embedder, cfg,
            budget_frames=4, max_rounds=2,
        )
        assert len(res.used_frame_ids) <= 4
        assert res.budget_exhausted or res.warnings  # 预算耗尽被标记或警告

    def test_negative_query_graceful(self, qs_env, demo):
        cfg, idx, embedder, vlm = _setup(qs_env, demo)
        res = adaptive_search(
            idx.video_id, "寻找关于美食烹饪的片段", idx, vlm, embedder, cfg,
            budget_frames=16,
        )
        assert res.windows == []

    def test_deterministic(self, qs_env, demo):
        cfg, idx, embedder, vlm = _setup(qs_env, demo)
        args = dict(budget_frames=32, max_rounds=1)
        r1 = adaptive_search(idx.video_id, "寻找讲解系统架构的片段", idx, vlm, embedder, cfg, **args)
        r2 = adaptive_search(idx.video_id, "寻找讲解系统架构的片段", idx, vlm, embedder, cfg, **args)
        assert [w.model_dump() for w in r1.windows] == [w.model_dump() for w in r2.windows]
        assert r1.refinement_rounds == r2.refinement_rounds

    def test_no_subtitles_no_embeddings_fallback(self, qs_env, demo):
        """无字幕 + 无嵌入后端：退化为 uniform 粗检索（VideoMind 风格 fallback）。"""
        cfg = Config.load().resolved()
        cfg.embedding.backend = "mock"
        cfg.retrieval.semantic.min_sim = 0.3
        ingest(demo["video"], video_id="noasr", config=cfg)  # 无 SRT
        idx = load_index("noasr", cfg)
        vlm = MockVLM(idx.content_hints)
        res = adaptive_search(
            idx.video_id, "寻找讲解系统架构的片段", idx, vlm, None, cfg, budget_frames=16
        )
        assert res.windows
        assert any("uniform" in wmsg for wmsg in res.warnings)
