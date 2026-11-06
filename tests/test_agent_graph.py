"""LangGraph Agent 图测试（P3 验收，说明书 §10）：

- 模拟"第一次候选错误、第二次成功"：Verifier 正确触发 ground 恢复
- 模拟永不收敛：预算限制下退出，不无限循环
- 同一 thread_id 可恢复（SQLite Checkpointer + interrupt/resume）
- Mock 端到端（MockVLM + MockEmbedding，确定性）
"""

from __future__ import annotations

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph 未安装")
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from queryseek.agent.graph import run_agent  # noqa: E402
from queryseek.agent.state import initial_state, default_budget  # noqa: E402
from queryseek.config import Config  # noqa: E402
from queryseek.embeddings.base import MockEmbedding  # noqa: E402
from queryseek.ingest.pipeline import ingest, load_index  # noqa: E402
from queryseek.models.interface import (  # noqa: E402
    FrameJudgment,
    InspectionResult,
    WindowVerification,
)
from queryseek.models.mock import MockVLM  # noqa: E402
from queryseek.schemas import TimeWindow  # noqa: E402
from queryseek.tools.validate import iou  # noqa: E402

GT_ARCH = TimeWindow(start_sec=12.0, end_sec=24.0)


def _setup(qs_env, demo, vid="demo"):
    cfg = Config.load().resolved()
    cfg.embedding.backend = "mock"
    cfg.retrieval.semantic.min_sim = 0.3
    ingest(demo["video"], video_id=vid, srt_path=demo["srt"], config=cfg)
    return cfg, load_index(vid, cfg)


def _configurable(cfg, thread_id, vlm=None, embedder=None, mock=True):
    d = {"thread_id": thread_id, "cfg": cfg.model_dump(), "mock": mock}
    if vlm is not None:
        d["vlm"] = vlm
    if embedder is not None:
        d["embedder"] = embedder
    return d


class ScriptedVLM:
    """测试替身：按脚本依次返回 inspect/verify 结果（模拟"先错后对"等路径）。"""

    model_id = "scripted-vlm-1"
    max_frames_per_call = 64

    def __init__(self, inspect_rel: list[list[str]], verify_seq: list[dict]):
        self._insp = inspect_rel
        self._ver = verify_seq
        self._i = 0
        self._v = 0
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def inspect_frames(self, query, frames, context=""):
        rel = self._insp[min(self._i, len(self._insp) - 1)]
        self._i += 1
        ids = [fid for fid in rel if any(f.frame_id == fid for f in frames)]
        return InspectionResult(
            relevant_frame_ids=ids,
            judgments=[FrameJudgment(frame_id=f.frame_id, relevant=f.frame_id in ids,
                                     reason="scripted") for f in frames],
            summary="scripted",
        )

    def verify_window(self, query, window, evidence):
        v = self._ver[min(self._v, len(self._ver) - 1)]
        self._v += 1
        return WindowVerification(**v)


class TestGraphControlFlow:
    def test_first_wrong_then_recover(self, qs_env, demo):
        """第一次候选错误（refuted）→ ground 重采样 → 第二次验证成功。"""
        cfg, idx = _setup(qs_env, demo)
        # subtitle 策略：查询"欢迎来到"只命中 seg0 → 单一候选 [0,11.5]
        # （该候选由脚本 VLM 裁决为错误，触发恢复路径）
        state = initial_state("demo", "欢迎来到", strategy="subtitle",
                              budget=default_budget({"max_frames": 32, "max_vlm_calls": 6,
                                                     "max_iterations": 3}))
        # 帧索引 t ∈ [36,48) 的 frame_id（重采样后脚本 VLM 只认这些帧）
        target_ids = [f.frame_id for f in idx.frames if 36.0 <= f.timestamp_sec < 48.0]
        vlm = ScriptedVLM(
            inspect_rel=[target_ids],  # reground_uniform 时命中 seg4
            verify_seq=[
                {"verdict": "refuted", "confidence": 0.1, "reason": "scripted: 与查询无关"},
                {"verdict": "confirmed", "confidence": 0.9, "reason": "scripted: 命中目标"},
            ],
        )
        final = run_agent(state, _configurable(cfg, "t1", vlm=vlm))
        assert final["status"] == "ok", final["errors"]
        verdicts = [r["verdict"] for r in final["verifier_results"]]
        assert verdicts == ["refuted", "confirmed"]
        actions = [o.get("action") for o in final["observations"]]
        assert "reground_uniform" in actions  # 失败恢复路径被触发
        assert final["selected_windows"]
        top = final["selected_windows"][0]
        assert 36.0 <= top["start_sec"] <= 48.0 and 36.0 <= top["end_sec"] <= 60.0
        assert final["iteration"] == 1  # 一轮 ground

    def test_never_converges_exits_bounded(self, qs_env, demo):
        """Verifier 永远 insufficient：迭代预算耗尽后必须退出。"""
        cfg, idx = _setup(qs_env, demo)
        state = initial_state("demo", "寻找讲解系统架构的片段", strategy="subtitle",
                              budget=default_budget({"max_frames": 32, "max_vlm_calls": 6,
                                                     "max_iterations": 2}))
        vlm = ScriptedVLM(
            inspect_rel=[[]],  # 细化永远找不到相关帧
            verify_seq=[{"verdict": "insufficient", "confidence": 0.3,
                         "reason": "scripted: 永远证据不足"}],
        )
        final = run_agent(state, _configurable(cfg, "t2", vlm=vlm))
        assert final["status"] in ("ok", "no_result")  # 有候选兜底 → ok + 未验证警告
        assert final["iteration"] <= 2  # 有界退出
        assert any("未经验证" in w or "预算" in w for w in final["warnings"])
        assert final["artifacts"]["validated"] is True  # 程序校验通过


class TestCheckpoint:
    def test_thread_id_resume(self, qs_env, demo, tmp_path):
        """interrupt 暂停 → 丢弃结果（模拟崩溃）→ 同 thread_id 恢复续跑。"""
        cfg, idx = _setup(qs_env, demo)
        state = initial_state("demo", "寻找讲解系统架构的片段", strategy="adaptive",
                              budget=default_budget({"max_frames": 32}))
        embedder = MockEmbedding(idx.content_hints, idx.frames)
        vlm = MockVLM(idx.content_hints)
        conf = _configurable(cfg, "resume-1", vlm=vlm, embedder=embedder)
        ckpt = tmp_path / "ckpt.sqlite"

        with SqliteSaver.from_conn_string(str(ckpt)) as saver:
            partial = run_agent(state, conf, checkpointer=saver, interrupt_before=["verify"])
        # 暂停点：planner + initial_retrieve 已执行，verify 尚未执行
        assert "candidate_windows" in partial
        assert not partial.get("verifier_results")

        # 模拟崩溃后重新打开 checkpoint，同一 thread_id 恢复
        with SqliteSaver.from_conn_string(str(ckpt)) as saver:
            final = run_agent(None, conf, checkpointer=saver)
        assert final["status"] in ("ok", "no_result"), final["errors"]
        assert final.get("verifier_results")  # 恢复后 verify 已执行

    def test_checkpoint_persisted_state(self, qs_env, demo, tmp_path):
        cfg, idx = _setup(qs_env, demo)
        state = initial_state("demo", "寻找讲解系统架构的片段", strategy="adaptive",
                              budget=default_budget({"max_frames": 32}))
        embedder = MockEmbedding(idx.content_hints, idx.frames)
        vlm = MockVLM(idx.content_hints)
        conf = _configurable(cfg, "persist-1", vlm=vlm, embedder=embedder)
        ckpt = tmp_path / "ckpt.sqlite"
        with SqliteSaver.from_conn_string(str(ckpt)) as saver:
            final = run_agent(state, conf, checkpointer=saver)
        # 重新打开：状态可从 checkpoint 读取
        with SqliteSaver.from_conn_string(str(ckpt)) as saver:
            from queryseek.agent.graph import build_agent_graph

            snap = build_agent_graph(saver).get_state({"configurable": {"thread_id": "persist-1"}})
        assert snap is not None
        assert snap.values["status"] == final["status"]


class TestRouteDefense:
    """状态边界防御（DECISIONS #43）：last_verifier / verifier_results 异常形态不崩溃。"""

    def _route(self, state):
        from queryseek.agent.graph import route_after_verify

        base = initial_state("demo", "q", strategy="adaptive",
                             budget=default_budget({"max_frames": 16, "max_vlm_calls": 8}))
        return route_after_verify({**base, **state})

    def test_last_verifier_as_list_does_not_crash(self):
        """last_verifier 为 list（曾致 'list' object has no attribute 'get'）
        时按"无上次判定"处理：有候选路由到 finalize，不崩溃。"""
        cand = [{"start_sec": 1.0, "end_sec": 5.0, "score": 0.8, "evidence_ids": []}]
        assert self._route({"last_verifier": [{"verdict": "refuted"}],
                            "candidate_windows": cand}) == "finalize"

    def test_last_verifier_as_list_grounds_without_candidates(self):
        """同样异常形态 + 无候选：给一次重采样恢复机会（ground），不崩溃。"""
        assert self._route({"last_verifier": [], "candidate_windows": []}) == "ground"


class TestMockEndToEnd:
    def test_negative_query_no_candidates_terminates(self, qs_env, demo):
        """回归测试：查询内容不存在（无任何候选 + 恢复仍无候选）必须终止，
        不得 verify->ground 死循环（曾触发 LangGraph recursion limit）。"""
        cfg, idx = _setup(qs_env, demo)
        state = initial_state("demo", "寻找关于美食烹饪的片段", strategy="adaptive",
                              budget=default_budget({"max_frames": 16, "max_vlm_calls": 8,
                                                     "max_iterations": 3}))
        embedder = MockEmbedding(idx.content_hints, idx.frames)
        vlm = MockVLM(idx.content_hints)
        final = run_agent(state, _configurable(cfg, "neg-1", vlm=vlm, embedder=embedder))
        assert final["status"] == "no_result", (final["status"], final["warnings"])
        assert final["selected_windows"] == []
        assert final["budget_exhausted"]  # 恢复路径失败后标记耗尽并退出

    def test_adaptive_confirmed(self, qs_env, demo):
        """Mock 全链路：adaptive 检索 → VLM 验证 confirmed → finalize ok。"""
        cfg, idx = _setup(qs_env, demo)
        state = initial_state("demo", "寻找讲解系统架构的片段", strategy="adaptive",
                              budget=default_budget({"max_frames": 32}))
        embedder = MockEmbedding(idx.content_hints, idx.frames)
        vlm = MockVLM(idx.content_hints)
        final = run_agent(state, _configurable(cfg, "e2e-1", vlm=vlm, embedder=embedder))
        assert final["status"] == "ok", (final["errors"], final["warnings"])
        assert final["selected_windows"]
        top = TimeWindow(**{k: final["selected_windows"][0][k]
                            for k in ("start_sec", "end_sec", "score")})
        assert iou(top, GT_ARCH) >= 0.5
        # Verifier 已确认（不是"未验证"兜底）
        assert any(r["verdict"] == "confirmed" for r in final["verifier_results"])
        # 预算未超限
        assert len(final["used_frame_ids"]) <= 32
        assert final["vlm_calls"] <= 6
