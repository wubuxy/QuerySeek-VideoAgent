"""Frozen, bounded, three-policy visual search pilot; --replay never calls models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from queryseek import PIPELINE_VERSION
from queryseek.agent.search_policy import PROMPT, VisualPolicy, initial_branch, node_window, rollout
from queryseek.config import Config
from queryseek.env import ensure_dotenv
from queryseek.eval.qvhighlights import temporal_iou
from queryseek.eval.runner import dataset_from_labels, load_dataset
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.ingest.frames import extract_uniform_frames, load_frames_index, save_frames_index
from queryseek.models.factory import build_vlm
from queryseek.models.mock import MockVLM, load_hints_file
from queryseek.tools.export import new_job_id
from scripts.sampling_probe import checked_output


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pool(sample, out, mock):
    asset = asset_from_probe(sample.video_id, sample.video_path)
    extraction_sec = 0.
    cache_hit = True
    if mock:
        directory = out / "mock_frames"
        directory.mkdir(exist_ok=True)
        index = directory / "frames.jsonl"
        if not index.exists():
            frames = extract_uniform_frames(asset.path, directory, 1., asset.duration_sec)
            save_frames_index(frames, index)
        frames = load_frames_index(index)
        meta = {"mock": True, "video_sha256": asset.sha256, "pipeline_version": PIPELINE_VERSION}
    else:
        runtime = Config.load(ROOT/"configs/qvhighlights_v2.yaml", apply_env=False)
        expected_steps = runtime.snapshot_for_cache()
        matches = []
        for p in sorted((ROOT / "outputs/baseline_integrity_cache").glob("*/sampling_meta.json")):
            value = json.loads(p.read_text(encoding="utf-8"))
            if (value.get("video_sha256") == asset.sha256
                    and value.get("pipeline_version") == PIPELINE_VERSION
                    and value.get("sampling_version") == "source-pts-select-v3"
                    and value.get("steps") == expected_steps):
                matches.append((p, value))
        if not matches:
            # The first twelve historical development videos used select-v2.
            # Reuse only their scene metadata; recreate images with current select-v3.
            from scripts.baseline_integrity_probe import load_v1_index, corrected_index
            old = load_v1_index(sample, runtime)
            refreshed, cache_hit, extraction_sec = corrected_index(old, runtime)
            p = refreshed.cache_dir/"sampling_meta.json"
            matches.append((p, json.loads(p.read_text(encoding="utf-8"))))
        p, meta = matches[0]  # Stable, label-independent cache choice.
        index = p.parent / "frames.jsonl"
        frames = load_frames_index(index)
    if (not frames or len({f.frame_id for f in frames}) != len(frames)
            or any(not 0 <= f.timestamp_sec < asset.duration_sec or not Path(f.image_path).is_file() for f in frames)):
        raise ValueError("Invalid source frame pool")
    # Hash all cached source images without exposing their pixels to the policy.
    manifest = {"video_sha256": asset.sha256, "duration_sec": asset.duration_sec,
                "metadata": meta, "index_path": str(index), "index_sha256": sha256_file(index),
                "offline_pool_frames": len(frames), "new_image_encodings": 0,
                "frame_cache_hit": cache_hit, "extraction_sec": extraction_sec,
                "frames": [{"id": f.frame_id, "time_sec": f.timestamp_sec,
                            "sha256": sha256_file(f.image_path)} for f in frames]}
    return asset, frames, manifest


class MockPolicy:
    def __init__(self, hints):
        self.vlm = MockVLM(hints)

    def __call__(self, payload, frames):
        relevant = self.vlm.inspect_frames(payload["query"], frames).relevant_frame_ids
        if relevant:
            selected = [f for f in frames if f.frame_id in relevant]
            start = min(f.timestamp_sec for f in selected)
            end = min(payload["duration_sec"], max(f.timestamp_sec for f in selected) + 1.)
            return {"action": "answer", "window": [start, end], "evidence_ids": relevant,
                    "reason": "Mock content hints only; no accuracy claim"}
        action = next((a for a in ("shift", "observe", "zoom_in_left", "abstain")
                       if a in payload["allowed_actions"]), "abstain")
        return {"action": action, "window": None, "evidence_ids": [], "reason": "Mock recovery plumbing"}


def score_result(result, sample, duration, path):
    gt = [[w.start_sec, w.end_sec] for w in sample.gt_windows]
    initial = node_window(path, duration)
    if any(not 0 <= a < b <= duration for a, b in gt):
        raise ValueError("GT outside video")
    pred = result.get("window")
    iou = max((temporal_iou(*pred, *g) for g in gt), default=0.) if pred else 0.
    # Annotation-dependent fields are created only after the policy has terminated.
    return {**result, "qid": sample.qid, "video_id": sample.video_id,
            "query": sample.query, "gt_windows": gt, "initial_window": list(initial),
            "initial_has_zero_gt_overlap": bool(gt) and not any(max(initial[0], a) < min(initial[1], b) for a, b in gt),
            "top1_iou": iou, "r1_at_05": int(iou >= .5), "r1_at_07": int(iou >= .7)}


def summarize(records, methods):
    groups = {}
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        n = len(rows)
        recovery = [r for r in rows if r["initial_has_zero_gt_overlap"]]
        groups[method] = {"n": n, "mean_top1_iou": sum(r["top1_iou"] for r in rows)/n,
                         "r1_at_05": sum(r["r1_at_05"] for r in rows)/n,
                         "r1_at_07": sum(r["r1_at_07"] for r in rows)/n,
                         "answered": sum(r["status"] == "answered" for r in rows),
                         "abstained": sum(r["status"] == "abstained" for r in rows),
                         "errors": sum(r["status"] == "error" for r in rows),
                         "actual_api_calls": sum(r.get("actual_api_calls", 0) for r in rows),
                         "actual_image_presentations": sum(r.get("actual_image_presentations", 0) for r in rows),
                         "mean_unique_frames": sum(r["unique_frames"] for r in rows)/n,
                         "recovery_eligible": len(recovery),
                         "recovery_r1_at_05": sum(r["r1_at_05"] for r in recovery)/len(recovery) if recovery else None,
                         "recovery_actions": sum(t.get("decision", {}).get("action") in {"zoom_out", "shift"}
                                                 for r in rows for t in r["trace"])}
    full, forward = groups["backtracking"], groups["forward_only"]
    return {"groups": groups, "paired_backtracking_minus_forward": {
        key: full[key]-forward[key] for key in ("mean_top1_iou", "r1_at_05", "r1_at_07")},
        "qa_accuracy": None, "independent_validation": False,
        "actual_bill": "TODO / unverified", "trained_policy": False,
        "interpretation": "small reused development localization pilot; not VTS reproduction or QA accuracy"}


def replay(out):
    out = checked_output(out)
    protocol = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (out/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    saved = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    for row in records:
        pred = row["window"]
        actual = max((temporal_iou(*pred, *g) for g in row["gt_windows"]), default=0.) if pred else 0.
        if actual != row["top1_iou"] or int(actual >= .5) != row["r1_at_05"]:
            raise ValueError("Saved prediction metrics mismatch")
    rebuilt = summarize(records, protocol["config"]["methods"])
    if rebuilt != saved:
        raise ValueError("Summary replay mismatch")
    report = {"exact_metrics_match": True, "records": len(records), "new_api_calls": 0,
              "records_sha256": sha256_file(out/"records.jsonl"), "metrics_sha256": sha256_file(out/"metrics.json")}
    write_json(out/"replay.json", report)
    return report


def run(out, mock=False, fixture=None):
    cfg_path = ROOT/"configs/agent_policy_pilot_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if (cfg["sample_count"] > 6 or cfg["max_turns"] != 4 or cfg["max_frames"] != 16
            or cfg["max_real_calls"] > 54 or cfg["max_retries"] != 0):
        raise ValueError("Pilot budget contract changed")
    out = checked_output(out)
    out.mkdir(parents=True, exist_ok=False)
    if mock:
        fixture = fixture or checked_output(ROOT/"outputs/agent_policy_fixture")
        fixture = checked_output(fixture)
        if not (fixture/"sample.mp4").exists():
            subprocess.run([sys.executable, str(ROOT/"scripts/make_synthetic_fixture.py"),
                            "--out-dir", str(fixture)], shell=False, timeout=600, check=True,
                           capture_output=True)
        dataset = fixture/"sample.labels.json"
        samples = dataset_from_labels(dataset, fixture/"sample.mp4", None)
        policy = MockPolicy(load_hints_file(dataset))
        vlm = None
    else:
        ensure_dotenv()
        if not os.environ.get("VLM_API_KEY"):
            raise RuntimeError("Real mode requires configured key; use --mock explicitly")
        dataset = ROOT/cfg["dataset"]
        samples = load_dataset(dataset)[:cfg["sample_count"]]
        if len(samples) != cfg["sample_count"]:
            raise ValueError("Do not reduce frozen sample denominator")
        runtime = Config.load()
        runtime.vlm.max_retries = 0
        runtime.vlm.max_output_tokens = cfg["max_output_tokens"]
        runtime.vlm.temperature = cfg["temperature"]
        runtime.vlm.enable_thinking = cfg["enable_thinking"]
        runtime.vlm.read_timeout_sec = 45.
        runtime.vlm.req_log_dir = str(out/"requests")
        vlm = build_vlm(runtime, use_mock=False)
        policy = VisualPolicy(vlm, cfg["max_real_calls"])
    sources = ["scripts/agent_policy_pilot.py", "queryseek/agent/search_policy.py",
               "scripts/baseline_integrity_probe.py", "configs/qvhighlights_v2.yaml",
               "queryseek/models/compatible_api.py", "queryseek/ingest/frames.py",
               "queryseek/eval/qvhighlights.py", "configs/agent_policy_pilot_v1.json",
               "docs/AGENT_ALGORITHM_PLAN.md"]
    snapshot = out/"source_snapshot"
    snapshot.mkdir()
    for name in sources:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    protocol = {"config": cfg, "mock": mock, "model": "mock-content-hints" if mock else vlm.model,
                "pipeline_version": PIPELINE_VERSION, "prompt": PROMPT,
                "source_hashes": {name: sha256_file(ROOT/name) for name in sources},
                "dataset_sha256": sha256_file(dataset), "qids": [s.qid for s in samples],
                "fresh_selection_sha256": sha256_file(ROOT/"data/qvhighlights_fresh80/selection.json")
                    if (ROOT/"data/qvhighlights_fresh80/selection.json").exists() else None}
    write_json(out/"protocol.json", protocol)
    write_json(out/"research_status.json", {"phase": "preparing", "mock": mock})
    started = time.perf_counter()
    records, inputs = [], []
    try:
        # Freeze/audit every input BEFORE making the first API request.
        prepared = []
        for sample in samples:
            asset, frames, manifest = load_pool(sample, out, mock)
            prepared.append((sample, asset, frames))
            inputs.append({"qid": sample.qid, **manifest})
        write_json(out/"inputs.json", inputs)
        for sample_index, (sample, asset, frames) in enumerate(prepared):
            path = initial_branch(sample.video_id, sample.query)
            # Rotate execution order to reduce confounding from provider time/order.
            order = cfg["methods"][sample_index % 3:] + cfg["methods"][:sample_index % 3]
            for method in order:
                before_calls = getattr(policy, "calls", 0)
                before_images = getattr(policy, "presentations", 0)
                before_usage = dict(vlm.total_usage) if vlm else {}
                t0 = time.perf_counter()
                result = rollout(sample.query, asset.duration_sec, frames, method, path, cfg, policy)
                result.update({"elapsed_sec": time.perf_counter()-t0,
                    "actual_api_calls": getattr(policy, "calls", 0)-before_calls,
                    "actual_image_presentations": getattr(policy, "presentations", 0)-before_images,
                    "usage": {k: v-before_usage.get(k, 0) for k, v in vlm.total_usage.items()} if vlm else {}})
                row = score_result(result, sample, asset.duration_sec, path)
                records.append(row)
                with (out/"records.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False)+"\n")
                write_json(out/"research_status.json", {"phase": "running", "completed": len(records),
                           "planned": len(samples)*3, "actual_api_calls": getattr(policy, "calls", 0)})
                print(json.dumps({k: row[k] for k in ("qid", "method", "status", "top1_iou", "actual_api_calls")}), flush=True)
        metrics = summarize(records, cfg["methods"])
        write_json(out/"metrics.json", metrics)
        write_json(out/"research_status.json", {"phase": "complete", "mock": mock,
            "completed": len(records), "elapsed_sec": time.perf_counter()-started,
            "actual_api_calls": getattr(policy, "calls", 0),
            "actual_image_presentations": getattr(policy, "presentations", 0),
            "usage": dict(vlm.total_usage) if vlm else {}, "new_clip_encodings": 0})
        return metrics
    except Exception as exc:
        write_json(out/"research_status.json", {"phase": "failed", "error_type": type(exc).__name__,
                                               "completed": len(records)})
        raise
    finally:
        if vlm:
            vlm.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.replay)))
        return
    out = args.out or ROOT/"outputs"/("agent_policy_mock" if args.mock else "agent_policy_pilot")/new_job_id()
    print(str(out), flush=True)
    run(out, args.mock)
    print(json.dumps(replay(out)), flush=True)


if __name__ == "__main__":
    main()
