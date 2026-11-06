"""Agent 状态定义（LangGraph StateGraph 使用，说明书 §7.4）。

Checkpoint（SQLite）要求状态可序列化：所有字段为纯 JSON 类型
（dict/list/str/float/int/bool），TimeWindow 一律以 dict 形式存储。
运行时对象（VideoIndex/VLM/Embedder/Config）不进入状态，由节点通过
RunnableConfig["configurable"] 重建。
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..schemas import TimeWindow

# 状态字段：说明书 §7.4 建议字段的落点
# video_id, query, requested_duration_sec, candidate_windows, evidence,
# iteration, used_frames, vlm_calls, budget, last_observation,
# verifier_result, status, errors, selected_windows, artifacts


class AgentState(TypedDict, total=False):
    # -- 输入 --
    video_id: str
    query: str
    requested_duration_sec: float | None
    strategy: str  # 初始检索策略：subtitle | uniform | semantic | adaptive
    budget: dict[str, Any]  # {"max_iterations","max_vlm_calls","max_frames","wall_time_limit_sec"}

    # -- 运行状态 --
    iteration: int
    status: str  # running | ok | no_result | failed
    plan: dict[str, Any]  # planner 输出：{"query_type","strategy","rationale"}
    candidate_windows: list[dict[str, Any]]  # [{"start_sec","end_sec","score","evidence_ids"}]
    selected_windows: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    used_frame_ids: list[str]
    vlm_calls: int
    observations: list[dict[str, Any]]  # 每轮 ground 观察记录
    verifier_results: list[dict[str, Any]]  # 每轮 verify 结果
    last_observation: dict[str, Any]
    last_verifier: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    fatal_errors: list[str]  # 不可恢复错误（如 VLM 连续失败）
    budget_exhausted: bool
    started_at: float
    elapsed_sec: float
    artifacts: dict[str, Any]


def window_to_dict(w: TimeWindow) -> dict[str, Any]:
    return {
        "start_sec": round(w.start_sec, 3),
        "end_sec": round(w.end_sec, 3),
        "score": w.score,
        "evidence_ids": list(w.evidence_ids),
    }


def dict_to_window(d: dict[str, Any]) -> TimeWindow:
    return TimeWindow(
        start_sec=float(d["start_sec"]),
        end_sec=float(d["end_sec"]),
        score=d.get("score"),
        evidence_ids=list(d.get("evidence_ids") or []),
    )


def windows_to_dicts(ws: list[TimeWindow]) -> list[dict[str, Any]]:
    return [window_to_dict(w) for w in ws]


def dicts_to_windows(ds: list[dict[str, Any]]) -> list[TimeWindow]:
    return [dict_to_window(d) for d in ds]


def default_budget(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    b = {
        "max_iterations": 4,
        "max_vlm_calls": 6,
        "max_frames": 32,
        "wall_time_limit_sec": 180.0,
    }
    if overrides:
        b.update(overrides)
    return b


def initial_state(
    video_id: str = "",
    query: str = "",
    strategy: str = "adaptive",
    requested_duration_sec: float | None = None,
    budget: dict[str, Any] | None = None,
) -> AgentState:
    return AgentState(
        video_id=video_id,
        query=query,
        requested_duration_sec=requested_duration_sec,
        strategy=strategy,
        budget=default_budget(budget),
        iteration=0,
        status="running",
        plan={},
        candidate_windows=[],
        selected_windows=[],
        evidence=[],
        used_frame_ids=[],
        vlm_calls=0,
        observations=[],
        verifier_results=[],
        last_observation={},
        last_verifier={},
        warnings=[],
        errors=[],
        fatal_errors=[],
        budget_exhausted=False,
        started_at=0.0,
        elapsed_sec=0.0,
        artifacts={},
    )
