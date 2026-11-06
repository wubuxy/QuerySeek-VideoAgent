"""Experimental, GT-blind supplemental CLIP sampling; no production dispatch change."""

from __future__ import annotations

import math

import numpy as np

from .curve_spans import curve_spans, select_curve_windows, short_peak_spans


def _evenly(pool: list[float], count: int) -> list[float]:
    if count <= 0 or not pool:
        return []
    count = min(count, len(pool))
    indices = np.linspace(0, len(pool) - 1, count).round().astype(int)
    return [pool[i] for i in indices]


def plan_supplemental_times(
    times: list[float], scores: list[float], duration_sec: float,
    budget: int = 32, grid_sec: float = .5,
) -> dict[str, list[float]]:
    """Match additional-frame counts, with 1/4 global exploration in the query arm.

    Only coarse observations enter this planner. The eligible grid excludes points
    within 0.2 seconds of existing observations. Query arm: three score-ranked
    coarse anchors >=8s apart, each receiving an equal share of local observations
    within +/-4s; any unspent observations are allocated globally. No dense scores,
    labels, or future image observations are consulted.
    """
    t, s = np.asarray(times, dtype=float), np.asarray(scores, dtype=float)
    if (not math.isfinite(duration_sec) or duration_sec <= 0
            or not math.isfinite(grid_sec) or grid_sec < .01
            or not isinstance(budget, int) or not 0 <= budget <= 256
            or len(t) != len(s) or not np.isfinite(t).all()
            or not np.isfinite(s).all() or np.any(np.diff(t) <= 0)
            or np.any(t < 0) or np.any(t >= duration_sec)):
        raise ValueError("Invalid sampling inputs")
    grid = np.arange(grid_sec / 2, duration_sec - .001, grid_sec)
    pool = [round(float(x), 3) for x in grid
            if not len(t) or np.min(np.abs(t - x)) >= .2]
    count = min(budget, len(pool))
    uniform = _evenly(pool, count)
    explore = min(count, max(1, count // 4)) if count else 0
    chosen = set(_evenly(pool, explore))
    anchors = []
    for i in sorted(range(len(t)), key=lambda i: (-s[i], t[i])):
        if all(abs(t[i] - anchor) >= 8 for anchor in anchors):
            anchors.append(float(t[i]))
        if len(anchors) == 3:
            break
    remaining = count - len(chosen)
    for k, anchor in enumerate(anchors):
        allocation = remaining // len(anchors) + (k < remaining % len(anchors))
        local = [x for x in pool if abs(x - anchor) <= 4 and x not in chosen]
        chosen.update(_evenly(local, allocation))
    chosen.update(_evenly([x for x in pool if x not in chosen], count - len(chosen)))
    return {"uniform": uniform, "query": sorted(chosen)}


def supplemental_proposals(times, scores, duration_sec: float, include_short: bool = False):
    """Same candidate construction/ranking for both acquisition policies."""
    windows = curve_spans(times, scores, duration_sec)
    if include_short:
        windows += short_peak_spans(times, scores, duration_sec)
    return windows, select_curve_windows(windows, top_k=5, nms_iou=.9)


def time_aligned_proposals(times, scores, duration_sec: float, include_short: bool = False):
    """Mechanism control: interpolate observed scores onto a fixed 1-second axis.

    Interpolation is imputation, not a new image observation. The original
    three-tap smoother/quantile ranking now sees the same physical-time grid
    under all acquisition policies. This also changes E4's effective bandwidth,
    so the aligned baseline must be reported alongside aligned query/uniform.
    """
    curve_spans(times, scores, duration_sec)  # validate before interpolation
    if not len(times):
        return [], []
    axis = np.arange(0., duration_sec, 1.)
    interpolated = np.interp(axis, times, scores)
    return supplemental_proposals(axis, interpolated, duration_sec, include_short)
