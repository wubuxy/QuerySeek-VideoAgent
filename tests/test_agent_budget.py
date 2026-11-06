"""BudgetGuard 单元测试（说明书 §7.4、§12.10：任何失败路径必须终止）。"""

from __future__ import annotations

import time

from queryseek.agent.budget import BudgetGuard


def _guard(**over) -> BudgetGuard:
    b = {"max_iterations": 4, "max_vlm_calls": 6, "max_frames": 32, "wall_time_limit_sec": 180.0}
    b.update(over)
    return BudgetGuard(b)


def test_capacity_queries():
    g = _guard()
    assert g.frames_left(10) == 22
    assert g.frames_left(40) == 0  # 不为负
    assert g.vlm_calls_left(6) == 0
    assert g.iterations_left(4) == 0
    assert g.time_left(time.time()) > 100


def test_can_observe():
    g = _guard(max_frames=8)
    assert g.can_observe(0, 8)
    assert not g.can_observe(1, 8)  # 只剩 7 帧
    assert not g.can_observe(0, 0)


def test_exhausted_reasons():
    g = _guard(max_frames=8, max_vlm_calls=2, max_iterations=1)
    state = {"used_frame_ids": list(range(8)), "vlm_calls": 2, "iteration": 1,
             "started_at": time.time()}
    reasons = g.exhausted_reasons(state)
    assert "max_frames" in reasons
    assert "max_vlm_calls" in reasons
    assert "max_iterations" in reasons
    assert "wall_time" not in reasons


def test_wall_time():
    g = _guard(wall_time_limit_sec=0.0)
    assert not g.on_time(time.time())
    assert "wall_time" in g.exhausted_reasons({"used_frame_ids": [], "vlm_calls": 0,
                                               "iteration": 0, "started_at": time.time()})
