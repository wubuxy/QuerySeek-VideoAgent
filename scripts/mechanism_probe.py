"""Offline oracle mechanism diagnosis; never an online retrieval strategy.

Reuse the completed correct baseline, with matched eight-frame oracle/background
clips. Unannotated backgrounds and text edits are not verified semantic negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.analyze_sampling_probe import clustered_delta
from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output, extra_features
from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
from queryseek.eval.qvhighlights import temporal_iou
from queryseek.eval.runner import dataset_from_labels, load_dataset
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.ingest.pipeline import VideoIndex
from queryseek.schemas import TimeWindow
from queryseek.tools.export import new_job_id

STRATA = ("appearance", "action", "state_change", "temporal_relation")


def midpoint_times(window, duration, count=8):
    start, end = window
    if (not all(math.isfinite(v) for v in (start, end, duration))
            or not 0 <= start < end <= duration
            or isinstance(count, bool) or not isinstance(count, int) or count <= 0):
        raise ValueError("Invalid half-open interval or frame budget")
    return [float(start + (i + .5) * (end - start) / count) for i in range(count)]


def longest_gt(windows, duration):
    if not windows:
        raise ValueError("Oracle diagnosis requires GT")
    for w in windows:
        midpoint_times((w.start_sec, w.end_sec), duration)
    w = min(windows, key=lambda w: (-w.duration_sec, w.start_sec))
    return (float(w.start_sec), float(w.end_sec))


def interval_mean(times, scores, window):
    if len(times) != len(scores):
        raise ValueError("Times/scores mismatch")
    selected = [s for t, s in zip(times, scores) if window[0] <= t < window[1]]
    return float(np.mean(selected)) if selected else None


def matched_background(gt, duration, length, times, scores, grid_sec=2.):
    """Hard unannotated control; exclusion of GT is explicitly oracle-only."""
    if (not math.isfinite(grid_sec) or grid_sec <= 0
            or not 0 < length <= duration or len(times) != len(scores)
            or not all(math.isfinite(float(v)) for v in [*times, *scores])):
        raise ValueError("Invalid background plan")
    latest = duration - length
    starts = sorted(set([*np.arange(0., latest, grid_sec).tolist(), float(latest)]))
    candidates = []
    for start in starts:
        end = min(duration, start + length)
        if any(start < w.end_sec and w.start_sec < end for w in gt):
            continue
        score = interval_mean(times, scores, (start, end))
        if score is not None:
            candidates.append((score, start, end))
    if not candidates:
        return None
    _, start, end = min(candidates, key=lambda x: (-x[0], x[1]))
    return (float(start), float(end))


def structural_diagnosis(group, gt):
    def best(windows):
        return max((temporal_iou(a, b, w.start_sec, w.end_sec)
                    for a, b in windows for w in gt), default=0.)
    pool = best([(w["start_sec"], w["end_sec"]) for w in group["all_windows"]])
    top5 = best([(w[0], w[1]) for w in group["top5"]])
    top1 = best([(w[0], w[1]) for w in group["top5"][:1]])
    kind = ("top1_hit" if top1 >= .5 else "candidate_missing" if pool < .5
            else "top5_pruned" if top5 < .5 else "top1_misranked")
    return {"pool_iou": pool, "top5_iou": top5, "top1_iou": top1, "failure_stage": kind}


def pooled_score(mat, text, timestamps=None):
    """Static-frame mean deliberately has no temporal encoding."""
    if mat.ndim != 2 or not len(mat) or mat.shape[1] != len(text):
        raise ValueError("Invalid clip features")
    if timestamps is not None and len(timestamps) != len(mat):
        raise ValueError("Timestamp metadata count mismatch")
    return float(np.mean(mat.astype(np.float64) @ text.astype(np.float64)))


def order_controls(mat, text, timestamps):
    rng = np.random.default_rng(12)
    forward = pooled_score(mat, text)
    permutation = rng.permutation(len(mat))
    return {"forward": forward, "reverse": pooled_score(mat[::-1], text),
            "shuffle": pooled_score(mat[permutation], text),
            "with_timestamps": pooled_score(mat, text, timestamps),
            "timestamp_consumed_by_model": False, "expected_invariance": True}


def validate_manifest(manifest, samples, records):
    items = manifest["samples"]
    qids = [x["qid"] for x in items]
    if not items or len(qids) != len(set(qids)):
        raise ValueError("Manifest qids must be nonempty and unique")
    for item in items:
        qid = item["qid"]
        if item["stratum"] not in STRATA or not item["object_text"].strip():
            raise ValueError("Invalid stratum or object description")
        if qid not in samples or qid not in records:
            raise ValueError("Manifest must remain inside completed development baseline")
        s, r = samples[qid], records[qid]
        if s.query != r["query"] or s.video_id != r["video_id"]:
            raise ValueError("Source record/dataset mismatch")
        if "both_fixes" not in r["groups"] or "both_fixes" not in r["observations"]:
            raise ValueError("Correct baseline arm missing")
        if "order_texts" in item and len(item["order_texts"]) != 2:
            raise ValueError("Order text control requires a pair")
    return items


def snapshot(out, paths):
    target = out / "source_snapshot"
    hashes = {}
    for path in paths:
        path = path.resolve()
        relative = path.relative_to(ROOT)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[relative.as_posix()] = sha256_file(path)
    (target / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return hashes


def summarize(records, intended, errors, mock):
    result = {"n_samples": len(intended), "completed": len(records),
              "coverage": len(records) / len(intended), "errors": errors,
              "mock": mock, "independent": False, "vlm_calls": 0,
              "new_encoded_frames": sum(r["new_encoded_frames"] for r in records),
              "extraction_sec": sum(r["extraction_sec"] for r in records),
              "encoding_sec": sum(r["encoding_sec"] for r in records), "strata": {}}
    for name in ("all", *STRATA):
        rows = records if name == "all" else [r for r in records if r["stratum"] == name]
        n = len(intended) if name == "all" else sum(i["stratum"] == name for i in intended)
        if not n:
            continue
        pairs = [r for r in rows if r["pair"] is not None]
        block = {"n_intended": n, "n_completed": len(rows), "pair_eligible": len(pairs),
                 "failure_stages": dict(Counter(r["structure"]["failure_stage"] for r in rows)),
                 "pool_hits": sum(r["structure"]["pool_iou"] >= .5 for r in rows),
                 "top5_hits": sum(r["structure"]["top5_iou"] >= .5 for r in rows),
                 "top1_hits": sum(r["structure"]["top1_iou"] >= .5 for r in rows)}
        if pairs:
            sources = [source_video_id(r["video_id"]) for r in pairs]
            for field in ("full_margin", "object_margin", "incremental_margin"):
                values = [r["pair"][field] for r in pairs]
                ci = clustered_delta(np.zeros(len(pairs)), values, sources)
                ci["unit"] = "cosine-similarity difference"
                block[field] = ci
            block["gt_beats_background"] = sum(r["pair"]["full_margin"] > 0 for r in pairs)
        controls = [r["text_control"] for r in rows if r["text_control"] is not None]
        block["text_control_count"] = len(controls)
        block["text_control_positive"] = sum(c["margin"] > 0 for c in controls)
        block["max_order_control_delta"] = max((abs(r["order_control"][key] - r["order_control"]["forward"])
                                                for r in rows for key in ("reverse", "shuffle", "with_timestamps")), default=0.)
        result["strata"][name] = block
    return result


def run_diagnostic(manifest, samples, baseline, cfg, out, mock=False, hints=None):
    by_q = {s.qid: s for s in samples}
    by_record = {r["qid"]: r for r in baseline}
    items = validate_manifest(manifest, by_q, by_record)
    count = manifest["frames_per_clip"]
    # Validate before allocating a model or producing an output directory.
    midpoint_times((0., 1.), 1., count)
    if not math.isfinite(manifest["background_grid_sec"]) or manifest["background_grid_sec"] <= 0:
        raise ValueError("Invalid background grid")
    out = checked_output(out)
    out.mkdir(parents=True, exist_ok=False)
    hashes = snapshot(out, [*ROOT.glob("queryseek/**/*.py"),
                            ROOT / "scripts/mechanism_probe.py", ROOT / "scripts/sampling_probe.py",
                            ROOT / "scripts/analyze_sampling_probe.py", ROOT / "scripts/make_qvhighlights_subset.py",
                            ROOT / "configs/mechanism_diagnostic_v1.json", ROOT / "configs/qvhighlights_v2.yaml",
                            ROOT / "docs/EVAL_MECHANISM_DIAGNOSTIC.md", ROOT / "pyproject.toml"])
    protocol = {"version": manifest["version"], "manifest": manifest, "mock": mock,
                "pipeline_version": PIPELINE_VERSION, "vlm_calls": 0, "independent": False,
                "source_sha256": hashes, "config": cfg.model_dump(mode="json"),
                "oracle_only": True, "background_label": "unannotated, not verified negative"}
    if not mock:
        source = ROOT / manifest["source_run"]
        protocol["baseline_hashes"] = {f: sha256_file(source / f) for f in ("protocol.json", "records.jsonl", "metrics.json")}
        protocol["dataset_sha256"] = sha256_file(ROOT / manifest["dataset"])
        protocol["fresh_selection_sha256"] = sha256_file(ROOT / "data/qvhighlights_fresh80/selection.json")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    started = time.perf_counter()
    emb = MockEmbedding() if mock else OpenCLIPEmbedding(max_side=cfg.video.max_frame_side_px)
    loading_sec = time.perf_counter() - started
    if not mock:
        actual = {tower: type(model.transformer.resblocks[0].mlp.gelu).__name__
                  for tower, model in (("vision", emb._model.visual), ("text", emb._model))}
        if actual != {"vision": "QuickGELU", "text": "QuickGELU"}:
            raise RuntimeError("Expected matched QuickGELU towers")
        (out / "activation_audit.json").write_text(json.dumps(actual), encoding="utf-8")
    records, errors = [], []
    for number, item in enumerate(items, 1):
        s = by_q[item["qid"]]
        try:
            asset = asset_from_probe(s.video_id, s.video_path)
            gt = longest_gt(s.gt_windows, asset.duration_sec)
            observation = by_record[s.qid]["observations"]["both_fixes"]
            times, scores = observation["times_sec"], observation["scores"]
            if any(not 0 <= t < asset.duration_sec for t in times):
                raise ValueError("Baseline time contract violation")
            background = matched_background(s.gt_windows, asset.duration_sec, gt[1] - gt[0],
                                            times, scores, manifest["background_grid_sec"])
            windows = [gt] + ([background] if background is not None else [])
            requested = [t for w in windows for t in midpoint_times(w, asset.duration_sec, count)]
            index = VideoIndex(asset, out, content_hints=hints or [])
            frames, mat, hit, extract_sec, encode_sec = extra_features(index, requested, emb, cfg, use_mock=mock)
            if len(frames) != count * len(windows):
                raise RuntimeError("Unequal frame budgets")
            for k, w in enumerate(windows):
                observed = [f.timestamp_sec for f in frames[k * count:(k + 1) * count]]
                if len(set(observed)) != count or any(not w[0] <= t < w[1] for t in observed):
                    raise RuntimeError("Clip frame identity/time contract violation")
            texts = [s.query, item["object_text"]]
            if "order_texts" in item:
                texts.extend(item["order_texts"])
            elif "contrast_text" in item:
                texts.extend([s.query, item["contrast_text"]])
            vectors = emb.encode_texts(texts)
            gt_scores = [pooled_score(mat[:count], v) for v in vectors]
            pair = None
            if background is not None:
                bg_scores = [pooled_score(mat[count:], v) for v in vectors]
                coarse_gt = interval_mean(times, scores, gt)
                coarse_bg = interval_mean(times, scores, background)
                full, obj = gt_scores[0] - bg_scores[0], gt_scores[1] - bg_scores[1]
                pair = {"full_margin": full, "object_margin": obj, "incremental_margin": full - obj,
                        "gt_full_score": gt_scores[0], "background_full_score": bg_scores[0],
                        "coarse_margin": coarse_gt - coarse_bg if coarse_gt is not None else None,
                        "coarse_gt_frames": int(sum(gt[0] <= t < gt[1] for t in times)),
                        "coarse_background_frames": int(sum(background[0] <= t < background[1] for t in times))}
            r = {"qid": s.qid, "video_id": s.video_id, "video_sha256": asset.sha256,
                 "query": s.query, "stratum": item["stratum"], "model_id": emb.model_id,
                 "structure": structural_diagnosis(by_record[s.qid]["groups"]["both_fixes"], s.gt_windows),
                 "gt_crop": list(gt), "all_gt": [[w.start_sec, w.end_sec] for w in s.gt_windows],
                 "background_crop": list(background) if background is not None else None,
                 "pair": pair, "texts": texts, "gt_text_scores": gt_scores,
                 "text_control": {"margin": gt_scores[2] - gt_scores[3]} if len(texts) == 4 else None,
                 "order_control": order_controls(mat[:count], vectors[0], [f.timestamp_sec for f in frames[:count]]),
                 "requested_times_sec": requested, "actual_times_sec": [f.timestamp_sec for f in frames],
                 "cache_hit": hit, "new_encoded_frames": 0 if hit else len(frames),
                 "extraction_sec": extract_sec, "encoding_sec": encode_sec}
            serialized = json.dumps(r, ensure_ascii=False)
            with (out / "records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
            records.append(r)
            print(f"{number}/{len(items)} qid={s.qid} {item['stratum']} pair={pair is not None} cache={hit}", flush=True)
        except Exception as exc:
            errors.append({"qid": s.qid, "error_type": type(exc).__name__, "message": str(exc)})
            print(f"FAILED qid={s.qid} {type(exc).__name__}: {exc}", flush=True)
        report = summarize(records, items, errors, mock)
        report["model_loading_sec"] = loading_sec
        (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not mock and sha256_file(ROOT / "data/qvhighlights_fresh80/selection.json") != protocol["fresh_selection_sha256"]:
        raise RuntimeError("Frozen independent selection changed")
    print("OUTPUT=" + str(out), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    cfg = Config.load(ROOT / "configs/qvhighlights_v2.yaml", apply_env=False).resolved()
    manifest = json.loads((ROOT / "configs/mechanism_diagnostic_v1.json").read_text(encoding="utf-8"))
    hints = None
    if args.mock:
        video, labels = ROOT / "assets/demo/sample.mp4", ROOT / "assets/demo/sample.labels.json"
        samples = [s for s in dataset_from_labels(labels, video, None) if s.gt_windows][:2]
        hints = json.loads(labels.read_text(encoding="utf-8"))["segments"]
        baseline = [{"qid": s.qid, "query": s.query, "video_id": s.video_id,
                     "groups": {"both_fixes": {"all_windows": [], "top5": []}},
                     "observations": {"both_fixes": {"times_sec": list(np.arange(0., 60., 4.)),
                                                      "scores": [0.] * 15}}} for s in samples]
        manifest["samples"] = [{"qid": s.qid, "stratum": "appearance", "object_text": s.query} for s in samples]
    else:
        source = ROOT / manifest["source_run"]
        metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
        if metrics["coverage"] != 1. or metrics["errors"] or protocol["pipeline_version"] != PIPELINE_VERSION:
            raise ValueError("A complete correct pipeline baseline is required")
        samples = load_dataset(ROOT / manifest["dataset"])
        baseline = [json.loads(line) for line in (source / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(baseline) != metrics["completed"] or len({r["qid"] for r in baseline}) != len(baseline):
            raise ValueError("Incomplete/duplicate baseline records")
    report = run_diagnostic(manifest, samples, baseline, cfg,
                            ROOT / "outputs/mechanism_probe" / new_job_id(), args.mock, hints)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
