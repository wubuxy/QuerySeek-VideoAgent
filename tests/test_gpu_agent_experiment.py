import json
from types import SimpleNamespace

import pytest

from scripts.gpu_agent_experiment import summarize, within_outputs
from scripts.prepare_gpu_agent import select_confirmation, source_id


def test_confirmation_is_source_disjoint_and_label_independent():
    a = SimpleNamespace(video_id="a_b_0.0_150.0")
    same_source = SimpleNamespace(video_id="a_b_150.0_300.0")
    b = SimpleNamespace(video_id="b_0.0_150.0")
    c = SimpleNamespace(video_id="c_0.0_150.0")
    assert source_id(a.video_id) == "a_b"
    result = select_confirmation([a], [same_source, b, c, b], 2)
    assert {s.video_id for s in result} == {b.video_id, c.video_id}
    with pytest.raises(ValueError):
        select_confirmation([a], [same_source], 1)


def test_output_path_cannot_escape():
    with pytest.raises(ValueError):
        within_outputs("outputs/../outside")


def test_failures_remain_in_denominator_and_two_conditions_cluster_together():
    rows = []
    for q in range(3):
        for condition in ["hash_half", "mirrored_half"]:
            for method in ["forward_only", "backtracking"]:
                success = int(q == 0 and method == "backtracking")
                rows.append({"qid": q, "source": str(q), "condition": condition, "method": method,
                    "status": "answered" if success else "error", "top1_iou": float(success),
                    "r1_at_05": success, "r1_at_07": success, "recovery_eligible": True,
                    "trace": [], "unique_frames": 4, "logical_image_presentations": 4,
                    "generation_stats": []})
    metrics = summarize(rows, {"seed": 12, "bootstrap_replicates": 100})
    assert metrics["source_count"] == 3
    assert metrics["groups"]["backtracking/hash_half"]["r1_at_05"] == 1/3
    assert metrics["groups"]["forward_only/hash_half"]["error"] == 3
    assert metrics["training_gate_pass"] is False


def test_transition_audit_detects_tampered_timestamp_and_budget():
    import copy
    from queryseek.agent.search_policy import rollout
    from queryseek.agent.anchored_policy import AnchoredPolicy
    from queryseek.schemas import VideoFrame
    from scripts.audit_gpu_agent import audit_episode
    from scripts.gpu_agent_experiment import MockBackend
    frames = [VideoFrame(frame_id=f"f{i}",timestamp_sec=float(i),image_path="") for i in range(16)]
    cfg = {"max_frames": 4,"frames_per_turn":2,"max_turns":2,"max_depth":3}
    backend = MockBackend()
    row = rollout("event",16.,frames,"fixed_uniform",(),cfg,AnchoredPolicy(backend,frames,16.))
    row.update(condition="global",generation_stats=backend.calls)
    sample = {"video_id":"source_0_16","query":"event","duration_sec":16.,
              "frames":[f.model_dump() for f in frames]}
    audit_episode(row,sample,cfg)
    bad = copy.deepcopy(row)
    bad["trace"][0]["state"]["images"][0]["source_time_sec"] += 1.
    with pytest.raises(ValueError,match="frozen evidence"):
        audit_episode(bad,sample,cfg)
    bad = copy.deepcopy(row)
    bad["logical_image_presentations"] += 1
    with pytest.raises(ValueError,match="Budget"):
        audit_episode(bad,sample,cfg)
