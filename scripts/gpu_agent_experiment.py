"""GPU/Mock experiment on portable frozen frame packages; replay needs no GPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from queryseek.agent.anchored_policy import ANCHOR_PROMPT, ANCHOR_VERSION, AnchoredPolicy
from queryseek.agent.search_policy import initial_branch, node_window, rollout, validate_decision, SearchState
from queryseek.schemas import VideoFrame
from queryseek.eval.qvhighlights import temporal_iou


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def within_outputs(path):
    resolved = Path(path).resolve()
    resolved.relative_to((ROOT/"outputs").resolve())
    return resolved


class TransformersBackend:
    def __init__(self, cfg, root=ROOT):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        self.torch, self.cfg = torch, cfg
        torch.manual_seed(cfg["seed"])
        torch.set_num_threads(8)
        path = str(root/cfg["model_path"])
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.processor.image_processor.size = {
            "shortest_edge": cfg["min_pixels"], "longest_edge": cfg["max_pixels"]}
        # This table depends only on the frozen tokenizer, not on schema or query.
        # Each generation still gets a fresh parser/enforcer for its own schema.
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.processor.tokenizer)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation=cfg["attention"], local_files_only=True).eval()
        self.calls = []
        self.errors = 0

    def _prefix_constraint(self, schema):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        return build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, JsonSchemaParser(schema))

    def __call__(self, prompt, frames, schema):
        from PIL import Image
        if self.errors >= self.cfg["max_consecutive_errors"]:
            raise RuntimeError("backend_circuit_open")
        started = time.perf_counter()
        record = {"images": len(frames), "input_tokens": 0, "output_tokens": 0}
        images = []
        try:
            content = [{"type": "text", "text": prompt}]
            for i, f in enumerate(frames):
                content.extend([{"type": "text", "text": f"anchor_index={i}; id={f.frame_id}; source_time_sec={f.timestamp_sec}"},
                                {"type": "image"}])
                with Image.open(f.image_path) as im:
                    images.append(im.convert("RGB"))
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
            record["input_tokens"] = int(inputs.input_ids.shape[-1])
            record["image_grid_thw"] = inputs.image_grid_thw.tolist()
            prefix = self._prefix_constraint(schema)
            self.torch.cuda.reset_peak_memory_stats()
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=self.cfg["max_new_tokens"],
                    max_time=self.cfg["max_seconds_per_generation"],
                    prefix_allowed_tokens_fn=prefix)
            generated = output[0, inputs.input_ids.shape[-1]:]
            record["output_tokens"] = int(generated.shape[-1])
            record["peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated()
            text_out = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
            obj = json.loads(text_out)
            record["status"] = "complete"
            self.errors = 0
            return obj
        except Exception as exc:
            self.errors += 1
            record.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            for im in images:
                im.close()
            record["elapsed_sec"] = time.perf_counter()-started
            self.calls.append(record)


class MockBackend:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, frames, schema):
        choices = schema["properties"]["choice"]["enum"]
        choice = "shift" if "shift" in choices and len(frames) <= 4 else next(
            (c for c in choices if c.startswith("answer:")), "abstain")
        self.calls.append({"status": "complete", "images": len(frames), "input_tokens": 0,
                           "output_tokens": 0, "elapsed_sec": 0., "peak_allocated_bytes": 0})
        return {"choice": choice}


def score(result, sample, start_path):
    gt = sample["gt_windows"]
    pred = result["window"]
    initial = node_window(start_path, sample["duration_sec"])
    if any(not 0 <= a < b <= sample["duration_sec"] for a, b in gt):
        raise ValueError("Invalid evaluation labels")
    iou = max((temporal_iou(*pred, *g) for g in gt), default=0.) if pred else 0.
    return {**result, "qid": sample["qid"], "source": sample["source"],
            "gt_windows": gt, "initial_window": list(initial),
            "recovery_eligible": bool(gt) and not any(max(initial[0], a) < min(initial[1], b) for a, b in gt),
            "top1_iou": iou, "r1_at_05": int(iou >= .5), "r1_at_07": int(iou >= .7)}


def summarize(rows, cfg):
    groups = {}
    for key in sorted({(r["method"], r["condition"]) for r in rows}):
        group = [r for r in rows if (r["method"], r["condition"]) == key]
        recovery = [r for r in group if r["recovery_eligible"]]
        groups["/".join(key)] = {
            "n": len(group), **{k: sum(r[k] for r in group)/len(group)
                               for k in ["top1_iou", "r1_at_05", "r1_at_07"]},
            **{s: sum(r["status"] == s for r in group) for s in ["answered", "abstained", "error"]},
            "recovery_eligible": len(recovery),
            "recovery_successes": sum(r["r1_at_05"] for r in recovery),
            "recovery_rate": sum(r["r1_at_05"] for r in recovery)/len(recovery) if recovery else None,
            "recovery_actions": sum(t.get("decision", {}).get("action") in {"shift", "zoom_out"}
                                    for r in group for t in r["trace"]),
            "unique_frames": sum(r["unique_frames"] for r in group),
            "image_presentations": sum(r["logical_image_presentations"] for r in group),
            "actual_generations": sum(len(r["generation_stats"]) for r in group),
            "input_tokens": sum(c["input_tokens"] for r in group for c in r["generation_stats"]),
            "output_tokens": sum(c["output_tokens"] for r in group for c in r["generation_stats"]),
        }
    lookup = {(r["qid"], r["condition"], r["method"]): r for r in rows}
    diffs = {}
    for r in rows:
        if r["method"] == "backtracking":
            f = lookup[(r["qid"], r["condition"], "forward_only")]
            diffs.setdefault(r["source"], []).append([r[k]-f[k] for k in ["top1_iou", "r1_at_05", "r1_at_07"]])
    data = np.array([np.mean(diffs[s], axis=0) for s in sorted(diffs)])
    rng = np.random.default_rng(cfg["seed"])
    boot = data[rng.integers(0, len(data), (cfg["bootstrap_replicates"], len(data)))].mean(axis=1)
    paired = {k: {"mean": float(data[:, i].mean()), "ci95": np.quantile(boot[:, i], [.025, .975]).tolist()}
              for i, k in enumerate(["top1_iou", "r1_at_05", "r1_at_07"])}
    recoveries = sum(r["r1_at_05"] for r in rows if r["method"] == "backtracking" and r["recovery_eligible"])
    return {"groups": groups, "paired_backtracking_minus_forward_source_average": paired,
            "training_gate_pass": all(r["status"] != "error" for r in rows) and recoveries >= 2 and paired["r1_at_05"]["ci95"][0] > 0,
            "source_count": len(data), "records": len(rows), "qa_accuracy": None,
            "independent_validation": False, "bill": "TODO / unverified"}


def run(package, out, stage, mock=False):
    package = Path(package).resolve()
    out = within_outputs(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((package/"config.json").read_text("utf8"))
    manifest = json.loads((package/"manifest.json").read_text("utf8"))
    if manifest.get("mock", False) != mock:
        raise ValueError("Mock/real package mismatch")
    for name, expected in manifest["source_hashes"].items():
        if sha(ROOT/name) != expected:
            raise ValueError("Frozen source hash mismatch")
    samples = [s for s in manifest["samples"] if s["stage"] == stage]
    if not samples:
        raise ValueError("Empty stage")
    # Validate all source inputs before loading the model.
    for s in samples:
        for f in s["frames"]:
            path = (package/f["image_path"]).resolve()
            path.relative_to(package)
            if sha(path) != f["sha256"]:
                raise ValueError("Frozen image SHA mismatch")
    protocol = {"config": cfg, "stage": stage, "mock": mock, "anchor_version": ANCHOR_VERSION,
                "prompt": ANCHOR_PROMPT, "manifest_sha256": sha(package/"manifest.json"),
                "config_sha256": sha(package/"config.json"),
                "source_hashes": {p: sha(ROOT/p) for p in [
                    "scripts/gpu_agent_experiment.py", "queryseek/agent/anchored_policy.py",
                    "queryseek/agent/search_policy.py", "queryseek/eval/qvhighlights.py"]}}
    write(out/"protocol.json", protocol)
    write(out/"status.json", {"phase": "loading_model", "mock": mock})
    loaded_at = time.perf_counter()
    backend = MockBackend() if mock else TransformersBackend(cfg)
    started = time.perf_counter()
    write(out/"environment.json", {"python": sys.version,
        "model_load_sec": started-loaded_at,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pydantic"] + (
            [] if mock else ["torch", "torchvision", "transformers", "accelerate", "lm-format-enforcer"])}})
    rows = []
    for n, sample in enumerate(samples):
        frames = [VideoFrame(frame_id=f["frame_id"], timestamp_sec=f["timestamp_sec"],
                             image_path=str(package/f["image_path"])) for f in sample["frames"]]
        base_path = initial_branch(sample["video_id"], sample["query"])
        jobs = [("fixed_uniform", "global", ())]
        for condition, path in [("hash_half", base_path), ("mirrored_half", (1-base_path[0],))]:
            jobs.extend((m, condition, path) for m in ["forward_only", "backtracking"])
        jobs = jobs[n % len(jobs):] + jobs[:n % len(jobs)]
        for method, condition, path in jobs:
            before = len(backend.calls)
            policy = AnchoredPolicy(backend, frames, sample["duration_sec"])
            if time.perf_counter()-started > cfg["max_seconds_per_stage"]:
                def policy(*_):
                    raise RuntimeError("stage_deadline_exceeded")
            result = rollout(sample["query"], sample["duration_sec"], frames, method, path, cfg, policy)
            result["condition"] = condition
            result["generation_stats"] = backend.calls[before:]
            row = score(result, sample, path)
            rows.append(row)
            with (out/"records.jsonl").open("a", encoding="utf8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False)+"\n")
            write(out/"status.json", {"phase": "running", "completed": len(rows),
                  "expected": len(samples)*5, "generations": len(backend.calls),
                  "elapsed_sec": time.perf_counter()-started})
            print(json.dumps({"completed": len(rows), "qid": sample["qid"], "method": method,
                              "condition": condition, "status": row["status"]}), flush=True)
    write(out/"metrics.json", summarize(rows, cfg))
    write(out/"status.json", {"phase": "complete", "completed": len(rows),
        "expected": len(samples)*5, "generations": len(backend.calls),
        "elapsed_sec": time.perf_counter()-started, "mock": mock})
    return replay(out)


def replay(out):
    out = within_outputs(out)
    protocol = json.loads((out/"protocol.json").read_text("utf8"))
    rows = [json.loads(x) for x in (out/"records.jsonl").read_text("utf8").splitlines()]
    seen = set()
    for row in rows:
        key = (row["qid"], row["condition"], row["method"])
        if key in seen:
            raise ValueError("Duplicate episode")
        seen.add(key)
        pred = row["window"]
        expected = max((temporal_iou(*pred, *g) for g in row["gt_windows"]), default=0.) if pred else 0.
        if expected != row["top1_iou"] or int(expected >= .5) != row["r1_at_05"] or int(expected >= .7) != row["r1_at_07"]:
            raise ValueError("Per-query metric mismatch")
        for t in row["trace"]:
            if "decision" in t:
                p = t["state"]
                state = SearchState(p["query"], p["duration_sec"], row["method"],
                    tuple(p["tree_path"]), [VideoFrame(frame_id=f["id"], timestamp_sec=f["source_time_sec"],
                    image_path="not-loaded-during-replay") for f in p["images"]])
                validate_decision(t["decision"], state, p["allowed_actions"])
    metrics = summarize(rows, protocol["config"])
    if metrics != json.loads((out/"metrics.json").read_text("utf8")):
        raise ValueError("Aggregate replay mismatch")
    status = json.loads((out/"status.json").read_text("utf8"))
    if len(rows) != status["expected"]:
        raise ValueError("Incomplete denominator")
    report = {"exact_metrics_match": True, "valid_decisions_rechecked": True, "records": len(rows),
              "new_model_calls": 0, "records_sha256": sha(out/"records.jsonl"),
              "metrics_sha256": sha(out/"metrics.json")}
    write(out/"replay.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage", choices=["pilot", "confirmation"], default="pilot")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.replay) if args.replay else run(args.package, args.out, args.stage, args.mock)))
