"""窗口对齐扩张（P4 真实数据驱动的算法改进，见 docs/DECISIONS.md #36）。

动机（QVHighlights 子集第一轮实测诊断）：K=16 帧预算在 150s 视频上每 9.4s 才
观察一帧，单帧命中 + 固定 1.5s 外扩产生的 3s 窗口对 10-40s 的真实时刻 IoU
结构性偏低（实测 CLIP top 帧命中 GT，但窗口太小）。时刻（moment）通常包含在
镜头（shot）内部——把候选窗口扩张到"证据帧最集中的那个镜头"的边界，是任务
结构驱动的原理性改进，不是验证集调参。

注意（第二轮诊断）：若直接取"首帧镜头到尾帧镜头"的并集，跨镜头簇（相邻镜头
边缘帧因时间间隙小被聚到一起）会被过度扩张；因此采用**多数投票**——选择包含
证据帧最多的镜头；全簇帧在单一镜头内则退化为该镜头。
"""

from __future__ import annotations

from ..schemas import Scene, TimeWindow

__all__ = ["expand_to_shots"]


def shot_of(t: float, scenes: list[Scene]) -> Scene | None:
    for s in scenes:
        if s.start_sec <= t < s.end_sec:
            return s
    return None


def expand_to_shots(
    window: TimeWindow,
    scenes: list[Scene],
    max_window_sec: float = 120.0,
    cluster_ts: list[float] | None = None,
) -> TimeWindow:
    """把窗口扩张到证据帧最集中的镜头（多数投票）；无镜头数据时原样返回。

    - cluster_ts：证据帧时间戳列表（默认取窗口首尾）。
    - 选中镜头过长（> max_window_sec）时退回原窗口，防过度扩张。
    """
    if not scenes:
        return window
    ts = cluster_ts or [window.start_sec, max(window.start_sec, window.end_sec - 1e-6)]
    votes: dict[str, int] = {}
    for t in ts:
        s = shot_of(t, scenes)
        if s is not None:
            votes[s.scene_id] = votes.get(s.scene_id, 0) + 1
    if not votes:
        return window
    best_id = max(votes, key=lambda k: votes[k])
    best = next(s for s in scenes if s.scene_id == best_id)
    if best.end_sec - best.start_sec > max_window_sec:
        return window
    return TimeWindow(
        start_sec=best.start_sec,
        end_sec=best.end_sec,
        score=window.score,
        evidence_ids=list(window.evidence_ids),
    )
