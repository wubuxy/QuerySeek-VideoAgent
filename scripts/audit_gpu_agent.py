"""Check frozen source evidence, every transition, budgets, and paired failures."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from queryseek.agent.anchored_policy import support_cells
from queryseek.agent.search_policy import SearchState, allowed_actions, initial_branch, node_window, select_observations, transition, validate_decision
from queryseek.schemas import VideoFrame
from queryseek.eval.qvhighlights import temporal_iou
from scripts.gpu_agent_experiment import sha, within_outputs, write, replay


def audit_episode(row, sample, cfg):
    frames = [VideoFrame(frame_id=f["frame_id"], timestamp_sec=f["timestamp_sec"], image_path="")
              for f in sample["frames"]]
    by_id = {f.frame_id:f for f in frames}
    cells = support_cells(frames, sample["duration_sec"])
    path = initial_branch(sample["video_id"], sample["query"])
    if row["condition"] == "mirrored_half":
        path = (1-path[0],)
    if row["method"] == "fixed_uniform":
        path = ()
    state = SearchState(sample["query"], sample["duration_sec"], row["method"], path)
    presentations = 0
    if len(row["trace"]) > (1 if row["method"] == "fixed_uniform" else cfg["max_turns"]):
        raise ValueError("Turn budget exceeded")
    for turn, record in enumerate(row["trace"]):
        count = cfg["max_frames"] if row["method"] == "fixed_uniform" else cfg["frames_per_turn"]
        expected = select_observations(frames, state, min(count, cfg["max_frames"]-len(state.observed)))
        if record["new_frame_ids"] != [f.frame_id for f in expected]:
            raise ValueError("Observation selection mismatch")
        state.observed.extend(expected)
        presentations += len(state.observed)
        payload = record["state"]
        if (payload["query"] != sample["query"] or payload["duration_sec"] != sample["duration_sec"]
                or payload["tree_path"] != list(state.path)
                or payload["current_window"] != list(node_window(state.path, state.duration))
                or payload["images"] != [{"id":f.frame_id,"source_time_sec":f.timestamp_sec} for f in state.observed]
                or payload["history"] != state.history
                or payload["remaining_new_frames"] != cfg["max_frames"]-len(state.observed)):
            raise ValueError("Policy state differs from frozen evidence")
        actions = allowed_actions(state, frames, turn, cfg)
        if payload["allowed_actions"] != actions:
            raise ValueError("Action availability mismatch")
        if "decision" not in record:
            if turn != len(row["trace"])-1 or row["status"] != "error":
                raise ValueError("Invalid failure termination")
            continue
        decision = validate_decision(record["decision"], state, actions)
        action = decision["action"]
        if action == "answer":
            refs = decision["evidence_ids"]
            expected_window = [min(cells[i][0] for i in refs), max(cells[i][1] for i in refs)]
            if decision["window"] != expected_window or row["window"] != expected_window:
                raise ValueError("Anchor window mismatch")
            if row["status"] != "answered" or turn != len(row["trace"])-1:
                raise ValueError("Invalid answer termination")
        elif action == "abstain":
            if row["status"] != "abstained" or turn != len(row["trace"])-1:
                raise ValueError("Invalid abstain termination")
        else:
            state.history.append({"window":payload["current_window"], "action":action, "reason":decision["reason"]})
            transition(state, action)
    if (len(state.observed) != row["unique_frames"] or len(state.observed) > cfg["max_frames"]
            or presentations != row["logical_image_presentations"]
            or len(row["trace"]) != row["policy_attempts"]):
        raise ValueError("Budget accounting mismatch")
    if len(row["generation_stats"]) > len(row["trace"]):
        raise ValueError("Hidden generation retry")
    if row["status"] != "answered" and row["window"] is not None:
        raise ValueError("Nonanswer contains prediction")


def audit(package, out):
    out = within_outputs(out)
    manifest = json.loads((Path(package)/"manifest.json").read_text("utf8"))
    samples = {s["qid"]:s for s in manifest["samples"]}
    protocol = json.loads((out/"protocol.json").read_text("utf8"))
    if sha(Path(package)/"manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Manifest changed")
    rows = [json.loads(line) for line in (out/"records.jsonl").read_text("utf8").splitlines()]
    counts, by_method, oracle_rows = Counter(), {}, []
    for row in rows:
        if row["gt_windows"] != samples[row["qid"]]["gt_windows"] or row["source"] != samples[row["qid"]]["source"]:
            raise ValueError("Evaluation labels/source differ from frozen package")
        audit_episode(row, samples[row["qid"]], protocol["config"])
        sample = samples[row["qid"]]
        frames = [VideoFrame(frame_id=f["frame_id"],timestamp_sec=f["timestamp_sec"],image_path="")
                  for f in sample["frames"]]
        cells = support_cells(frames, sample["duration_sec"])
        observed = {f["id"] for t in row["trace"] for f in t["state"]["images"]}
        ordered = sorted(observed, key=lambda i: cells[i])
        oracle = max((temporal_iou(cells[a][0],cells[b][1],*g)
                      for i,a in enumerate(ordered) for b in ordered[i:]
                      for g in sample["gt_windows"]), default=0.)
        oracle_rows.append({"qid":row["qid"],"condition":row["condition"],"method":row["method"],
            "observed_anchor_oracle_iou":oracle,"actual_iou":row["top1_iou"],
            "failure_class": "correct" if row["r1_at_05"] else
                ("execution_error" if row["status"] == "error" else
                 "observed_anchor_coverage_insufficient" if oracle < .5 else "selection_or_stopping_failure")})
        by_method[(row["qid"],row["condition"],row["method"])] = row
        if row["method"] != "backtracking":
            continue
        recovered = any(t.get("decision",{}).get("action") in {"shift","zoom_out"} for t in row["trace"])
        if row["recovery_eligible"]:
            if row["r1_at_05"]:
                category = "wrong_start_recovered_to_correct_answer"
            elif row["status"] == "error":
                category = "wrong_start_execution_error"
            elif recovered:
                category = "wrong_start_navigation_used_but_not_correct"
            else:
                category = "wrong_start_no_recovery_action"
        elif row["r1_at_05"]:
            category = "overlapping_start_correct"
        else:
            category = "overlapping_start_not_correct"
        counts[category] += 1
    pairs = []
    for (qid,condition,method), b in by_method.items():
        if method != "backtracking":
            continue
        f = by_method[(qid,condition,"forward_only")]
        pairs.append({"qid":qid,"condition":condition,"source":b["source"],
                      "delta_iou":b["top1_iou"]-f["top1_iou"],
                      "delta_r1":b["r1_at_05"]-f["r1_at_05"],
                      "backtracking_status":b["status"],"forward_status":f["status"],
                      "recovery_eligible":b["recovery_eligible"]})
    report = {"all_episode_audits_pass":True,"records":len(rows),"diagnosis":dict(counts),
              "r1_wins":sum(p["delta_r1"]>0 for p in pairs),
              "r1_losses":sum(p["delta_r1"]<0 for p in pairs),"paired_episodes":len(pairs),
              "replay":replay(out)}
    write(out/"completion_audit.json", report)
    write(out/"paired_differences.json", pairs)
    write(out/"observed_anchor_diagnosis.json", oracle_rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.package,args.out),ensure_ascii=False))
