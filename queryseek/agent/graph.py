"""LangGraph 条件图：plan → initial_retrieve → verify ⇄ ground → finalize → validate。

- 路由（route_after_verify）：confirmed → finalize；refuted/insufficient 且预算
  允许 → ground（细化/恢复）；fatal → fail；预算耗尽 → finalize 兜底。
- Checkpointer：SQLite（langgraph-checkpoint-sqlite），按 thread_id 持久化/
  恢复；同一 thread_id 可复现与续跑（说明书 §7.4）。
- BudgetGuard 硬限制：任何失败路径必须终止（说明书 §12.10）。
"""

from __future__ import annotations

import time
from typing import Any

from .budget import BudgetGuard
from .nodes import (
    fail_node,
    finalize,
    ground,
    initial_retrieve,
    planner,
    validate_artifact,
    verify,
)
from .state import AgentState, initial_state

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover — 未安装 langgraph 时给出明确错误
    SqliteSaver = StateGraph = None  # type: ignore[assignment]

try:
    from langgraph.types import RetryPolicy
except ImportError:  # 旧版无节点重试接口，仅用 timeout
    RetryPolicy = None  # type: ignore[assignment]

# 节点重试策略（DECISIONS #42）：禁用 VLM 节点的自动重试——网络间歇性故障时
# 有界客户端每次调用 ≤ ~73s 必失败（分项超时），失败直接冒泡 → 评测记录 error
# 后继续下一项，杜绝"默认节点重试 × 多轮迭代"把单次网络故障放大成数分钟阻塞。
# 注：LangGraph 的节点 timeout= 仅支持异步节点（sync 节点 ValueError），故用
# 禁用重试 + 有界客户端 + BudgetGuard 三者组合实现等效的硬上界。
def _vlm_node_opts() -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if RetryPolicy is not None:
        opts["retry_policy"] = RetryPolicy(max_attempts=1)  # 1 次尝试 = 不重试
    return opts


class LangGraphUnavailable(RuntimeError):
    pass


def _require_langgraph():
    if StateGraph is None:
        raise LangGraphUnavailable(
            "langgraph 未安装：pip install -e '.[agent]'"
        )


def route_after_verify(state: dict) -> str:
    if state.get("fatal_errors"):
        return "fail"
    # 预算耗尽（含 ground 恢复路径返回的 budget_exhausted）必须终止，
    # 否则"无候选 + 恢复失败"会形成 verify->ground 死循环（说明书 §12.10）
    if state.get("budget_exhausted"):
        return "finalize"
    guard = BudgetGuard(state.get("budget", {}))
    if not guard.can_iterate(int(state.get("iteration", 0))):
        return "finalize"
    if not guard.can_vlm_call(int(state.get("vlm_calls", 0))):
        return "finalize"
    last = state.get("last_verifier") or {}
    if not isinstance(last, dict):
        last = {}  # 状态边界防御（DECISIONS #43）：非 dict 一律按"无上次判定"处理
    verdict = last.get("verdict")
    if verdict == "confirmed":
        return "finalize"
    if verdict in ("refuted", "insufficient"):
        return "ground"
    if not state.get("candidate_windows"):
        return "ground"  # 初次检索无候选：给一次重采样恢复机会
    return "finalize"


def build_agent_graph(checkpointer=None):
    _require_langgraph()
    g = StateGraph(AgentState)
    g.add_node("planner", planner, **_vlm_node_opts())
    g.add_node("initial_retrieve", initial_retrieve, **_vlm_node_opts())
    g.add_node("verify", verify, **_vlm_node_opts())
    g.add_node("ground", ground, **_vlm_node_opts())
    g.add_node("finalize", finalize)
    g.add_node("fail", fail_node)
    g.add_node("validate_artifact", validate_artifact)
    g.add_edge(START, "planner")
    g.add_edge("planner", "initial_retrieve")
    g.add_edge("initial_retrieve", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"ground": "ground", "finalize": "finalize", "fail": "fail"},
    )
    g.add_edge("ground", "verify")
    g.add_edge("finalize", "validate_artifact")
    g.add_edge("fail", "validate_artifact")
    g.add_edge("validate_artifact", END)
    return g.compile(checkpointer=checkpointer)


def run_agent(
    state: dict[str, Any] | None,
    configurable: dict[str, Any],
    checkpointer=None,
    interrupt_before: list[str] | None = None,
) -> dict[str, Any]:
    """执行（或续跑）一次 agent 任务。

    - state=None：从 checkpoint 恢复续跑（须与之前的 thread_id 相同）。
    - configurable 需包含：thread_id, cfg（Config.model_dump()）, mock, 可选
      vlm/embedder 替身（测试注入）、可选 trace_path。
    """
    if state is not None:
        state = {**initial_state(), **state}
        state.setdefault("started_at", time.time())
    graph = build_agent_graph(checkpointer)
    config = {"configurable": configurable}
    return graph.invoke(state, config, interrupt_before=interrupt_before)


def resume_agent(
    checkpointer,
    thread_id: str,
    configurable_extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """按 thread_id 恢复：返回已检查点状态；可再次 run_agent(None, ...) 续跑。"""
    _require_langgraph()
    graph = build_agent_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id, **(configurable_extra or {})}}
    snapshot = graph.get_state(config)
    return snapshot.values if snapshot else None
