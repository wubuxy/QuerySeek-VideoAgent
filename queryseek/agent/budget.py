"""BudgetGuard：迭代/调用/帧/时间硬限制（说明书 §7.4、§12.10）。

任何失败路径必须终止而非无限循环；预算检查在 router 与每次 VLM 调用前执行。
"""

from __future__ import annotations

import time
from typing import Any


class BudgetGuard:
    def __init__(self, budget: dict[str, Any]):
        self.budget = budget

    # -- 能力查询 ------------------------------------------------------------

    def frames_left(self, used_frames: int) -> int:
        return max(0, int(self.budget["max_frames"]) - used_frames)

    def vlm_calls_left(self, vlm_calls: int) -> int:
        return max(0, int(self.budget["max_vlm_calls"]) - vlm_calls)

    def iterations_left(self, iteration: int) -> int:
        return max(0, int(self.budget["max_iterations"]) - iteration)

    def time_left(self, started_at: float) -> float:
        return max(0.0, float(self.budget["wall_time_limit_sec"]) - (time.time() - started_at))

    # -- 判定 ------------------------------------------------------------------

    def can_observe(self, used_frames: int, n_frames: int) -> bool:
        return self.frames_left(used_frames) >= n_frames and n_frames > 0

    def can_vlm_call(self, vlm_calls: int) -> bool:
        return self.vlm_calls_left(vlm_calls) > 0

    def can_iterate(self, iteration: int) -> bool:
        return self.iterations_left(iteration) > 0

    def on_time(self, started_at: float) -> bool:
        return self.time_left(started_at) > 0

    def exhausted_reasons(self, state: dict[str, Any]) -> list[str]:
        """返回当前已触发的预算耗尽原因列表。"""
        reasons = []
        if self.frames_left(len(state.get("used_frame_ids", []))) <= 0:
            reasons.append("max_frames")
        if self.vlm_calls_left(int(state.get("vlm_calls", 0))) <= 0:
            reasons.append("max_vlm_calls")
        if self.iterations_left(int(state.get("iteration", 0))) <= 0:
            reasons.append("max_iterations")
        if self.time_left(float(state.get("started_at", time.time()))) <= 0:
            reasons.append("wall_time")
        return reasons
