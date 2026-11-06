import json
from types import SimpleNamespace

import pytest

from queryseek.agent.anchored_policy import AnchoredPolicy, anchor_catalog, support_cells
from queryseek.agent.search_policy import SearchState, rollout, validate_decision


def pool(times=(0., 4., 8., 12.)):
    return [SimpleNamespace(frame_id=f"f{i}", timestamp_sec=t, image_path="")
            for i, t in enumerate(times)]


def test_last_anchor_is_inside_half_open_answer_and_duplicates_share_cell():
    frames = pool((0., 4., 4., 8.))
    cells = support_cells(frames, 10.)
    assert cells["f1"] == cells["f2"] == [2., 6.]
    assert cells["f3"] == [6., 10.]
    payload = {"allowed_actions": ["answer", "abstain"]}
    ordered, cat, _ = anchor_catalog(payload, frames, cells)
    state = SearchState("q", 10., "fixed_uniform", (), frames)
    for decision in cat.values():
        validate_decision(decision, state, payload["allowed_actions"])
    assert cat["answer:1:3"]["window"] == [2., 10.]


@pytest.mark.parametrize("times,duration", [((0., 10.), 10.), ((-1.,), 10.), ((float("nan"),), 10.), ((0.,), 0.)])
def test_invalid_source_times_rejected(times, duration):
    with pytest.raises(ValueError):
        support_cells(pool(times), duration)


def test_unobserved_anchors_cannot_be_selected_or_shown():
    frames = pool()
    captured = []
    def backend(prompt, seen, schema):
        captured.append((prompt, seen, schema))
        return {"choice": "answer:0:1"}
    policy = AnchoredPolicy(backend, frames, 16.)
    result = policy({"allowed_actions": ["answer", "abstain"]}, frames[:2])
    assert result["evidence_ids"] == ["f0", "f1"]
    assert "f2" not in captured[0][0]
    assert "answer:0:2" not in captured[0][2]["properties"]["choice"]["enum"]
    assert result["window"] == [0., 6.]


def test_illegal_choice_fails_without_response_leak():
    frames = pool()
    policy = AnchoredPolicy(lambda *_: {"choice": "secret-unseen-frame"}, frames, 16.)
    result = rollout("q", 16., frames, "fixed_uniform", (), {
        "max_frames": 4, "frames_per_turn": 2, "max_turns": 3, "max_depth": 3}, policy)
    assert result["status"] == "error"
    assert "secret-unseen" not in json.dumps(result)


def test_mirrored_branch_recovery_is_reachable_with_same_budget():
    frames = pool()
    def backend(prompt, observed, schema):
        return {"choice": "shift" if all(f.timestamp_sec < 8 for f in observed) else "answer:2:3"}
    policy = AnchoredPolicy(backend, frames, 16.)
    cfg = {"max_frames": 4, "frames_per_turn": 2, "max_turns": 2, "max_depth": 2}
    result = rollout("event", 16., frames, "backtracking", (0,), cfg, policy)
    assert result["status"] == "answered"
    assert result["trace"][0]["decision"]["action"] == "shift"
    assert result["window"] == [6., 16.]
