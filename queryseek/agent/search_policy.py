"""Bounded visual search environment; annotations never enter the policy state.

Uniform temporal tree is an experimental action interface, not VTS replication.
All public intervals use source seconds and half-open bounds.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from queryseek.models.compatible_api import _image_data_url, _parse_json_strict


PROMPT = """You are a video evidence search agent. Locate the event described by the query.
Only use the supplied images and their source timestamps. Do not assume an unseen event happened.
The current window is a node in a binary temporal tree. zoom_in_left/right enters its child;
zoom_out returns to its parent; shift enters its sibling; observe gets more unseen frames.
Use recovery actions when the current branch lacks evidence. Frames from earlier turns remain visible.
Your task is to choose ONE allowed action, not to follow a fixed action sequence.
Return exactly a JSON object with action, window, evidence_ids, reason.
For answer, window is [start_sec,end_sec), evidence_ids contains observed supporting image IDs.
For every other action window must be null and evidence_ids must be [].
Answer only if evidence supports the query. Abstain when evidence is inadequate and no useful action remains.
Keep reason under 400 characters. Final-turn actions are restricted to answer or abstain.
"""


VALIDATION_CODES = {"invalid_action_schema", "illegal_action_or_reason", "nonanswer_evidence",
                    "invalid_answer_interval", "unsupported_evidence_reference"}


class DecisionValidationError(ValueError):
    """Static codes only: safe to persist without logging an API error payload."""


def decision_shape(obj):
    """Privacy-safe structural diagnostics, without arbitrary response strings."""
    if not isinstance(obj, dict):
        return {"response_type": type(obj).__name__}
    window = obj.get("window")
    ids = obj.get("evidence_ids")
    reason = obj.get("reason")
    known_actions = {"answer", "abstain", "observe", "shift", "zoom_out", "zoom_in_left", "zoom_in_right"}
    action = obj.get("action")
    return {"field_count": len(obj), "known_fields": sorted(set(obj) & {"action", "window", "evidence_ids", "reason"}),
            "action": action if isinstance(action, str) and action in known_actions else "unknown",
            "window": window if isinstance(window, list) and len(window) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in window) else None,
            "evidence_count": len(ids) if isinstance(ids, list) else None,
            "reason_length": len(reason) if isinstance(reason, str) else None}


def initial_branch(video_id: str, query: str) -> tuple[int, ...]:
    return (int(hashlib.sha256((video_id + query).encode()).hexdigest(), 16) % 2,)


def node_window(path: tuple[int, ...], duration: float) -> tuple[float, float]:
    if not math.isfinite(duration) or duration <= 0 or any(p not in (0, 1) for p in path):
        raise ValueError("Invalid temporal tree")
    start, end = 0.0, float(duration)
    for child in path:
        mid = (start + end) / 2
        start, end = (start, mid) if child == 0 else (mid, end)
    return start, end


@dataclass
class SearchState:
    query: str
    duration: float
    method: str
    path: tuple[int, ...]
    observed: list = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


def select_observations(frames, state: SearchState, count: int):
    start, end = node_window(state.path, state.duration)
    used = {f.frame_id for f in state.observed}
    available = sorted((f for f in frames if f.frame_id not in used and start <= f.timestamp_sec < end),
                       key=lambda f: (f.timestamp_sec, f.frame_id))
    count = min(count, len(available))
    if count <= 0:
        return []
    # Mid-quantiles select distinct frames without reading pixels or relevance scores.
    return [available[min(len(available)-1, int((i+.5)*len(available)/count))] for i in range(count)]


def allowed_actions(state, frames, turn, cfg):
    actions = ["answer", "abstain"]
    if (state.method == "fixed_uniform" or turn == cfg["max_turns"]-1
            or len(state.observed) >= cfg["max_frames"]):
        return actions
    if select_observations(frames, state, 1):
        actions.append("observe")
    if len(state.path) < cfg["max_depth"]:
        for child, name in ((0, "zoom_in_left"), (1, "zoom_in_right")):
            probe = SearchState(state.query, state.duration, state.method, state.path+(child,), state.observed)
            if select_observations(frames, probe, 1):
                actions.append(name)
    if state.method == "backtracking" and state.path:
        actions.extend(["zoom_out", "shift"])
    return actions


def policy_payload(state, actions, cfg, turn):
    # Deliberately construct an allowlist, rather than serializing dataset/sample objects.
    return {"query": state.query, "duration_sec": state.duration,
            "current_window": list(node_window(state.path, state.duration)),
            "tree_path": list(state.path), "allowed_actions": actions,
            "remaining_new_frames": cfg["max_frames"]-len(state.observed),
            "remaining_calls_including_this": cfg["max_turns"]-turn,
            "history": list(state.history),
            "images": [{"id": f.frame_id, "source_time_sec": f.timestamp_sec} for f in state.observed]}


def validate_decision(obj, state, actions):
    if not isinstance(obj, dict) or set(obj) != {"action", "window", "evidence_ids", "reason"}:
        raise DecisionValidationError("invalid_action_schema")
    action = obj["action"]
    if action not in actions or not isinstance(obj["reason"], str) or len(obj["reason"]) > 800:
        raise DecisionValidationError("illegal_action_or_reason")
    ids = obj["evidence_ids"]
    if action != "answer":
        if obj["window"] is not None or ids != []:
            raise DecisionValidationError("nonanswer_evidence")
    else:
        window = obj["window"]
        if (not isinstance(window, list) or len(window) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in window)
                or not 0 <= window[0] < window[1] <= state.duration):
            raise DecisionValidationError("invalid_answer_interval")
        observed = {f.frame_id: f for f in state.observed}
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids) or any(i not in observed for i in ids)
                or any(not window[0] <= observed[i].timestamp_sec < window[1] for i in ids)):
            raise DecisionValidationError("unsupported_evidence_reference")
    return obj


def transition(state, action):
    if action.startswith("zoom_in_"):
        state.path += (0 if action.endswith("left") else 1,)
    elif action == "zoom_out":
        if not state.path:
            raise ValueError("Root has no parent")
        state.path = state.path[:-1]
    elif action == "shift":
        if not state.path:
            raise ValueError("Root has no sibling")
        state.path = state.path[:-1] + (1-state.path[-1],)


class VisualPolicy:
    """Reuse the existing bounded, secret-safe API adapter without automatic retries."""
    def __init__(self, vlm, max_calls):
        self.vlm, self.max_calls = vlm, max_calls
        self.calls = self.failures = self.presentations = 0

    def __call__(self, payload, frames):
        if self.calls >= self.max_calls or self.failures >= 3:
            raise RuntimeError("Global API budget or failure circuit exhausted")
        parts = [{"type": "text", "text": PROMPT + json.dumps(payload, ensure_ascii=False)}]
        for frame in frames:
            parts.extend([{"type": "text", "text": f"{frame.frame_id} @ {frame.timestamp_sec:.6f} seconds"},
                          {"type": "image_url", "image_url": {"url": _image_data_url(frame.image_path)}}])
        self.calls += 1
        self.presentations += len(frames)
        try:
            response = self.vlm._chat(parts, max_retries=0)
        except Exception:
            self.failures += 1
            raise
        self.failures = 0
        return _parse_json_strict(response)


def rollout(query, duration, frames, method, start_path, cfg, policy):
    if method not in {"fixed_uniform", "forward_only", "backtracking"}:
        raise ValueError("Unknown method")
    if (cfg["max_frames"] <= 0 or cfg["frames_per_turn"] <= 0 or cfg["max_turns"] <= 0
            or len({f.frame_id for f in frames}) != len(frames)
            or any(not 0 <= f.timestamp_sec < duration for f in frames)):
        raise ValueError("Invalid observations/budgets")
    state = SearchState(query, duration, method, () if method == "fixed_uniform" else start_path)
    node_window(state.path, duration)
    trace, window, status, attempts, presentations = [], None, "abstained", 0, 0
    for turn in range(1 if method == "fixed_uniform" else cfg["max_turns"]):
        n = cfg["max_frames"] if method == "fixed_uniform" else cfg["frames_per_turn"]
        fresh = select_observations(frames, state, min(n, cfg["max_frames"]-len(state.observed)))
        state.observed.extend(fresh)
        actions = allowed_actions(state, frames, turn, cfg)
        payload = policy_payload(state, actions, cfg, turn)
        record = {"turn": turn, "before_window": payload["current_window"],
                  "new_frame_ids": [f.frame_id for f in fresh], "state": payload}
        attempts += 1
        presentations += len(state.observed)
        try:
            decision = validate_decision(policy(payload, state.observed), state, actions)
        except Exception as exc:
            # Do not persist exception strings: an upstream error can include request data.
            record["error_type"] = type(exc).__name__
            record["error_code"] = str(exc) if isinstance(exc, DecisionValidationError) and str(exc) in VALIDATION_CODES else "policy_or_transport_failure"
            trace.append(record)
            status = "error"
            break
        record["decision"] = decision
        trace.append(record)
        action = decision["action"]
        if action in {"answer", "abstain"}:
            window = decision["window"]
            status = "answered" if action == "answer" else "abstained"
            break
        state.history.append({"window": payload["current_window"], "action": action,
                              "reason": decision["reason"]})
        transition(state, action)
    return {"method": method, "status": status, "window": window,
            "unique_frames": len(state.observed), "policy_attempts": attempts,
            "logical_image_presentations": presentations, "trace": trace}
