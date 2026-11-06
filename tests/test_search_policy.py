import json
from pathlib import Path

import pytest

from queryseek.agent.search_policy import (SearchState, VisualPolicy, allowed_actions, initial_branch,
    node_window, policy_payload, rollout, select_observations, transition, validate_decision)
from queryseek.schemas import VideoFrame


CFG = {"max_turns": 4, "max_frames": 16, "frames_per_turn": 4, "max_depth": 3}


def frames():
    return [VideoFrame(frame_id=f"f{i}", timestamp_sec=float(i), image_path="unused.jpg", source="uniform")
            for i in range(60)]


def decision(action, window=None, ids=None):
    return {"action": action, "window": window, "evidence_ids": ids or [], "reason": "test"}


def test_tree_backtrack_and_sibling_preserve_half_open_seconds():
    state = SearchState("q", 60., "backtracking", (0, 1))
    assert node_window(state.path, 60.) == (15., 30.)
    transition(state, "zoom_out")
    transition(state, "shift")
    assert node_window(state.path, 60.) == (30., 60.)
    with pytest.raises(ValueError):
        node_window((), float("nan"))


def test_recovery_changes_reachable_observations_without_gt():
    def policy(payload, seen):
        relevant = [f for f in seen if f.timestamp_sec >= 40]
        if relevant:
            return decision("answer", [40., 60.], [f.frame_id for f in relevant])
        return decision("shift" if "shift" in payload["allowed_actions"] else "abstain")
    full = rollout("q", 60., frames(), "backtracking", (0,), CFG, policy)
    forward = rollout("q", 60., frames(), "forward_only", (0,), CFG, policy)
    assert full["status"] == "answered" and forward["status"] == "abstained"
    assert full["unique_frames"] == 8 and full["logical_image_presentations"] == 12
    assert "gt" not in json.dumps(full["trace"]).lower()


@pytest.mark.parametrize("bad", [decision("shift"), decision("answer", [-1, 20], ["f3"]),
    decision("answer", [0, 20], ["unseen"]), decision("answer", [0, 3], ["f3"]),
    decision("answer", [0, float("nan")], ["f3"]), decision("abstain", [0, 20])])
def test_illegal_actions_and_unsupported_evidence_are_rejected(bad):
    state = SearchState("q", 60., "forward_only", (0,), [frames()[3]])
    with pytest.raises(ValueError):
        validate_decision(bad, state, ["answer", "abstain"])


def test_budget_and_final_turn_terminate_and_no_duplicate_observations():
    seen_actions = []
    def policy(payload, seen):
        seen_actions.append(payload["allowed_actions"])
        assert len({f.frame_id for f in seen}) == len(seen) <= 16
        return decision("observe" if "observe" in payload["allowed_actions"] else "abstain")
    result = rollout("q", 60., frames(), "forward_only", (0,), CFG, policy)
    assert result["policy_attempts"] == 4 and result["unique_frames"] == 16
    assert result["logical_image_presentations"] == 40
    assert seen_actions[-1] == ["answer", "abstain"]


def test_failures_stop_episode_and_do_not_leak_exception_payload():
    def fail(*_):
        raise RuntimeError("secret-sentinel-base64")
    result = rollout("q", 60., frames(), "backtracking", (0,), CFG, fail)
    assert result["status"] == "error" and result["window"] is None
    assert "secret-sentinel" not in json.dumps(result)


def test_selection_and_payload_use_only_observed_frames():
    state = SearchState("q", 60., "forward_only", (0,))
    state.observed = select_observations(frames(), state, 4)
    payload = policy_payload(state, allowed_actions(state, frames(), 0, CFG), CFG, 0)
    assert len(payload["images"]) == 4
    assert all(i["source_time_sec"] < 30 for i in payload["images"])
    assert initial_branch("v", "q") == initial_branch("v", "q")


def test_api_global_failure_circuit_without_retries():
    class Failing:
        def _chat(self, parts, max_retries):
            assert max_retries == 0
            raise RuntimeError("private")
    policy = VisualPolicy(Failing(), 54)
    for _ in range(4):
        with pytest.raises(RuntimeError):
            policy({}, [])
    assert policy.calls == 3 and policy.failures == 3


def test_synthetic_video_end_to_end_and_replay(demo, tmp_path, monkeypatch):
    import scripts.agent_policy_pilot as pilot
    import scripts.sampling_probe as sampling
    monkeypatch.setattr(sampling, "ROOT", tmp_path)
    out = tmp_path/"outputs/run"
    # Use the repository-standard fixture generator through the shared demo fixture.
    out.mkdir(parents=True)
    sample = pilot.dataset_from_labels(demo["labels"], demo["video"], None)[0]
    asset, pool, manifest = pilot.load_pool(sample, out, True)
    policy = pilot.MockPolicy(pilot.load_hints_file(demo["labels"]))
    records = []
    methods = ["fixed_uniform", "forward_only", "backtracking"]
    for method in methods:
        result = rollout(sample.query, asset.duration_sec, pool, method, (1,), CFG, policy)
        records.append(pilot.score_result(result, sample, asset.duration_sec, (1,)))
    pilot.write_json(out/"protocol.json", {"config": {"methods": methods}})
    pilot.write_json(out/"metrics.json", pilot.summarize(records, methods))
    (out/"records.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert pilot.replay(out)["exact_metrics_match"]
    assert manifest["metadata"]["pipeline_version"] == "2"
    assert all(r["status"] != "error" for r in records)


def test_output_path_escape_is_rejected(tmp_path):
    from scripts.sampling_probe import checked_output
    with pytest.raises(ValueError):
        checked_output(tmp_path/"outside")


def test_diagnostic_shape_excludes_arbitrary_response_strings():
    from queryseek.agent.search_policy import decision_shape
    obj = {"action": "secret", "reason": "secret-base64", "extra-secret": "private", "evidence_ids": ["private"]}
    summary = decision_shape(obj)
    assert "secret" not in json.dumps(summary) and "private" not in json.dumps(summary)
    assert summary["reason_length"] == 13


def test_validation_failures_have_static_codes_in_trace():
    result = rollout("q", 60., frames(), "forward_only", (0,), CFG,
                     lambda *_: decision("answer", [0., 5.], ["unseen"]))
    assert result["trace"][0]["error_code"] == "unsupported_evidence_reference"


def test_real_pool_rejects_matching_video_with_stale_sampling_config(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.agent_policy_pilot as pilot
    import scripts.baseline_integrity_probe as baseline
    cache = tmp_path/"outputs/baseline_integrity_cache/stale"
    cache.mkdir(parents=True)
    (cache/"sampling_meta.json").write_text(json.dumps({"video_sha256": "same-video",
        "pipeline_version": "2", "sampling_version": "source-pts-select-v3",
        "steps": {"coarse_fps": .25}}))
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "asset_from_probe", lambda *_: SimpleNamespace(sha256="same-video"))
    monkeypatch.setattr(pilot.Config, "load", lambda *_args, **_kwargs:
                        SimpleNamespace(snapshot_for_cache=lambda: {"coarse_fps": 1.}))
    def rebuild(*_):
        raise RuntimeError("rebuild-required")
    monkeypatch.setattr(baseline, "load_v1_index", rebuild)
    with pytest.raises(RuntimeError, match="rebuild-required"):
        pilot.load_pool(SimpleNamespace(video_id="v", video_path="v.mp4"), tmp_path, False)


def test_trace_history_is_a_snapshot_not_future_actions():
    def policy(payload, seen):
        return decision("shift") if not payload["history"] else decision("abstain")
    result = rollout("q", 60., frames(), "backtracking", (0,), CFG, policy)
    assert result["trace"][0]["state"]["history"] == []
    assert len(result["trace"][1]["state"]["history"]) == 1
