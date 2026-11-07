"""Lazy visual-change tree and multi-child navigation, without evaluator labels.

This research interface leaves the historical binary/SFT policy unchanged.
VTS supplies the boundary algorithm; this module supplies time contracts,
observation budgets, node identity and navigation. Nodes are candidate regions,
not certified events or automatically valid answer evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Callable

import numpy as np

from queryseek.schemas import VideoFrame
from queryseek.vendor.vts.boundaries import (
    _boundaries_to_segments, _compute_frame_deltas, _detect_boundaries_threshold)


@dataclass(frozen=True)
class SplitConfig:
    k: float = 1.5
    min_children: int = 3
    max_children: int = 8
    leaf_seconds: float = 64.0
    max_depth: int = 6
    max_observed_nodes: int = 3

    def __post_init__(self):
        integers = [self.min_children, self.max_children, self.max_depth, self.max_observed_nodes]
        if (any(type(x) is not int for x in integers)
                or not 2 <= self.min_children <= self.max_children
                or self.max_depth < 1 or self.max_observed_nodes < 1
                or not math.isfinite(self.k) or self.k < 0
                or not math.isfinite(self.leaf_seconds) or self.leaf_seconds <= 0):
            raise ValueError('Invalid temporal tree configuration')


@dataclass
class NodeObservation:
    frames: list[VideoFrame]
    embeddings: np.ndarray
    cost: dict = field(default_factory=dict)


@dataclass
class TemporalNode:
    node_id: str
    parent_id: str | None
    start_sec: float
    end_sec: float
    depth: int
    children: tuple[str, ...] | None = None
    split_info: dict = field(default_factory=dict)


class TreeBudgetExceeded(RuntimeError):
    pass


def _validate_observation(start: float, end: float, observation: NodeObservation, min_samples: int):
    times = [float(f.timestamp_sec) for f in observation.frames]
    vectors = np.asarray(observation.embeddings)
    if (not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end
            or len(times) < min_samples or any(not start <= t < end for t in times)
            or any(a >= b for a, b in zip(times, times[1:]))
            or vectors.ndim != 2 or vectors.shape[0] != len(times) or vectors.shape[1] < 1
            or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) <= 0)):
        raise ValueError('Invalid source-time observations/features')
    return times, vectors


def visual_change_split(start: float, end: float, observation: NodeObservation, cfg: SplitConfig):
    times, vectors = _validate_observation(start, end, observation, cfg.min_children)
    deltas = _compute_frame_deltas(vectors)
    indices = _detect_boundaries_threshold(deltas, cfg.k, cfg.min_children, cfg.max_children)
    spans = _boundaries_to_segments(times, indices, start, end)
    threshold = float(deltas.mean() + cfg.k*deltas.std())
    info = {'algorithm': 'vts-cosine-change-v1', 'source_times_sec': times,
            'deltas': deltas.tolist(), 'threshold': threshold,
            'boundary_indices': indices,
            'threshold_children': int(np.count_nonzero(deltas > threshold))+1,
            'forced_minimum': int(np.count_nonzero(deltas > threshold))+1 < cfg.min_children,
            'boundary_brackets_sec': [[times[i], times[i+1]] for i in indices],
            'boundary_points_are': 'midpoints of observed source timestamps, not decoded frame timestamps'}
    return spans, info


def uniform_partition(start: float, end: float, children: int):
    if (type(children) is not int or children < 2 or not math.isfinite(start)
            or not math.isfinite(end) or not 0 <= start < end):
        raise ValueError('Invalid uniform partition')
    bounds = [start+(end-start)*i/children for i in range(children)] + [end]
    return list(zip(bounds, bounds[1:]))


class TemporalTree:
    def __init__(self, duration: float, provider: Callable, config: SplitConfig | None = None):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid duration')
        self.duration, self.provider = float(duration), provider
        self.config = config or SplitConfig()
        self.nodes = {'r': TemporalNode('r', None, 0., float(duration), 0)}
        self.observations: dict[str, NodeObservation] = {}

    def observe(self, node_id: str) -> NodeObservation:
        node = self.nodes[node_id]
        if node_id not in self.observations:
            if len(self.observations) >= self.config.max_observed_nodes:
                raise TreeBudgetExceeded('Node observation budget exhausted')
            observation = self.provider(node.start_sec, node.end_sec)
            # Validate before adding to either state or the navigation image pool.
            _validate_observation(node.start_sec, node.end_sec, observation, min_samples=1)
            self.observations[node_id] = observation
        return self.observations[node_id]

    def expand(self, node_id: str) -> tuple[str, ...]:
        node = self.nodes[node_id]
        if node.children is not None:
            return node.children
        if node.end_sec-node.start_sec < self.config.leaf_seconds or node.depth >= self.config.max_depth:
            node.children = ()
            node.split_info = {'stop_reason': 'duration' if node.depth < self.config.max_depth else 'depth'}
            return ()
        observation = self.observe(node_id)
        spans, info = visual_change_split(node.start_sec, node.end_sec, observation, self.config)
        children = []
        for i, (start, end) in enumerate(spans):
            if not node.start_sec <= start < end <= node.end_sec:
                raise ValueError('Invalid child interval')
            child_id = f'{node_id}.{i}'
            self.nodes[child_id] = TemporalNode(child_id, node_id, float(start), float(end), node.depth+1)
            children.append(child_id)
        node.children, node.split_info = tuple(children), info
        return node.children

    def to_dict(self):
        return {'version': 'visual-change-tree-v1', 'duration_sec': self.duration,
                'config': asdict(self.config), 'observed_nodes': len(self.observations),
                'nodes': [asdict(n) for n in self.nodes.values()],
                'observation_costs': {key: obs.cost for key, obs in self.observations.items()}}


class TreeNavigator:
    """Executable navigation tools. Departing a node never marks it refuted.

    CLIP preprocessing observations are not automatically shown to the policy.
    `observe` explicitly selects new images within a separate policy-frame budget.
    This is an interface, not a newly trained or evaluated decision policy.
    """
    def __init__(self, tree: TemporalTree, max_frames: int = 16, max_moves: int = 8):
        if type(max_frames) is not int or type(max_moves) is not int or min(max_frames, max_moves) < 1:
            raise ValueError('Invalid navigation budget')
        self.tree, self.current_id = tree, 'r'
        self.max_frames, self.max_moves = max_frames, max_moves
        self.observed: dict[str, VideoFrame] = {}
        self.history: list[dict] = []

    def expand(self):
        return self.tree.expand(self.current_id)

    def observe(self, count: int = 4):
        if type(count) is not int or count < 1:
            raise ValueError('Invalid frame count')
        remaining = self.max_frames-len(self.observed)
        if not remaining:
            return []
        available = [f for f in self.tree.observe(self.current_id).frames if f.frame_id not in self.observed]
        count = min(count, remaining, len(available))
        fresh = [available[int((i+.5)*len(available)/count)] for i in range(count)] if count else []
        self.observed.update((f.frame_id, f) for f in fresh)
        return fresh

    def move(self, action: str, target_id: str | None = None):
        if len(self.history) >= self.max_moves:
            raise TreeBudgetExceeded('Navigation budget exhausted')
        current = self.tree.nodes[self.current_id]
        if action == 'zoom_in':
            allowed = current.children or ()
        elif action == 'shift' and current.parent_id is not None:
            allowed = tuple(c for c in self.tree.nodes[current.parent_id].children if c != self.current_id)
        elif action == 'zoom_out' and current.parent_id is not None and target_id is None:
            allowed = (current.parent_id,)
            target_id = current.parent_id
        else:
            raise ValueError('Illegal navigation action')
        if target_id not in allowed:
            raise ValueError('Target is not a permitted child/sibling/parent')
        self.history.append({'action': action, 'from_node': self.current_id, 'to_node': target_id})
        self.current_id = target_id

    def state(self):
        node = self.tree.nodes[self.current_id]
        def brief(node_id):
            n = self.tree.nodes[node_id]
            return {'node_id': node_id, 'window_sec': [n.start_sec, n.end_sec],
                    'expanded': n.children is not None, 'evidence_status': 'unknown'}
        siblings = () if node.parent_id is None else self.tree.nodes[node.parent_id].children
        local = sorted(f.timestamp_sec for f in self.observed.values() if node.start_sec <= f.timestamp_sec < node.end_sec)
        points = [node.start_sec] + local + [node.end_sec]
        return {'current_node': self.current_id, 'window_sec': [node.start_sec, node.end_sec],
                'children': [brief(c) for c in node.children or ()],
                'siblings': [brief(c) for c in siblings if c != self.current_id],
                'parent_id': node.parent_id, 'history': list(self.history),
                'observed_frames': [{'id': f.frame_id, 'source_time_sec': f.timestamp_sec} for f in self.observed.values()],
                'remaining_frames': self.max_frames-len(self.observed),
                'remaining_moves': self.max_moves-len(self.history),
                'largest_unsampled_gap_sec': max(b-a for a, b in zip(points, points[1:])),
                'evidence_status': 'unknown', 'gap_is_not_semantic_coverage': True}
