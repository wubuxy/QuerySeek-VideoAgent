"""Frozen E4 / matched supplemental CLIP acquisition experiment (zero VLM).

Real default: old 80-query development set. --mock without --dataset uses the
repository's generated synthetic fixture. Outputs never overwrite earlier runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.embeddings.cache import load_embeddings, save_embeddings
from queryseek.embeddings.factory import build_embedder
from queryseek.eval.qvhighlights import eval_mr_brief, temporal_iou
from queryseek.eval.runner import dataset_from_labels, load_dataset
from queryseek.ingest.frames import extract_scene_frames, load_frames_index, save_frames_index
from queryseek.ingest.pipeline import ingest, load_index
from queryseek.retrieval.sampling_refinement import (
    plan_supplemental_times, supplemental_proposals,
)
from queryseek.schemas import Scene
from queryseek.tools.export import new_job_id
from queryseek.tools.paths import resolved_path_is_within

GROUPS = ("E4", "E4S", "uniform_curve", "query_curve", "uniform_short", "query_short")


def optional_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def checked_output(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved_path_is_within(resolved,(ROOT / "outputs").resolve()):
        raise ValueError("Experiment outputs must be within repository outputs/")
    return resolved


def feature_signature(index, model_id, times, max_side):
    signature = {"video_sha256": index.asset.sha256, "pipeline_version": PIPELINE_VERSION,
            "experiment_version": "sampling-probe-v2", "model_id": model_id,
            "max_side": max_side, "times_sec": times,
            "extractor": "extract_scene_frames-source-pts-v3",
            "jpeg_quality": 3, "open_clip_version": optional_version("open-clip-torch")}
    if model_id.startswith("mock"):
        signature["mock_hints_sha256"] = hashlib.sha256(json.dumps(
            getattr(index,"content_hints",[]),sort_keys=True).encode()).hexdigest()
    return signature


def extra_features(index, times, embedder, cfg, use_mock=False, workers=4):
    """Cache the union for physical work sharing; per-arm logical budgets stay separate."""
    if not times:
        return [], np.zeros((0, embedder.dim), dtype=np.float32), True, 0.0, 0.0
    signature = feature_signature(index, embedder.model_id, times, cfg.video.max_frame_side_px)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = checked_output(ROOT / "outputs/sampling_feature_cache" / key)
    meta_path = directory / "sampling_meta.json"
    loaded = load_embeddings(directory) if meta_path.is_file() else None
    if loaded is not None and json.loads(meta_path.read_text(encoding="utf-8")) == signature:
        frames = load_frames_index(directory / "frames.jsonl")
        mat, meta = loaded
        if (meta.get("model_id") == embedder.model_id
                and meta.get("frame_ids") == [f.frame_id for f in frames]
                and len(frames) == len(times)
                and all(0 <= f.timestamp_sec < index.asset.duration_sec for f in frames)):
            return frames, mat, True, 0.0, 0.0
        raise RuntimeError("Supplemental feature cache mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    def extract(pair):
        i, timestamp = pair
        # A tiny structured Scene makes the existing extractor seek to the exact midpoint.
        half = min(.001, timestamp / 2, (index.asset.duration_sec - timestamp) / 2)
        scene = Scene(scene_id=f"probe_{i}", start_sec=timestamp-half, end_sec=timestamp+half)
        frames = extract_scene_frames(index.asset.path, directory / f"frame_{i:04d}",
                                      [scene], cfg.video.max_frame_side_px)
        if len(frames) != 1:
            raise RuntimeError("Supplemental frame extraction failed")
        frame = frames[0]
        frame.frame_id = f"supplement_{i:04d}"
        return frame
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = list(executor.map(extract, enumerate(times)))
    extraction_sec = time.perf_counter() - start
    encoder = MockEmbedding(index.content_hints, frames) if use_mock else embedder
    start = time.perf_counter()
    mat = encoder.encode_images([f.image_path for f in frames])
    encoding_sec = time.perf_counter() - start
    save_frames_index(frames, directory / "frames.jsonl")
    save_embeddings(directory, mat, [f.frame_id for f in frames], {"model_id": embedder.model_id,
                    "backend": "mock" if use_mock else "openclip"})
    meta_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return frames, mat, False, extraction_sec, encoding_sec


def supplemental_observations(requested_times, frames, scores):
    """Associate seek requests with ordered frames without relabeling their PTS."""
    if not len(requested_times) == len(frames) == len(scores):
        raise RuntimeError("Supplemental observation counts differ")
    return {t: (frame.timestamp_sec, float(score))
            for t, frame, score in zip(requested_times, frames, scores)}


def oracle_details(windows, gt):
    return [max((temporal_iou(w.start_sec, w.end_sec, g.start_sec, g.end_sec)
                 for w in windows), default=0.0) for g in gt]


def run_probe(samples, cfg, output, use_mock=False, budget=32, grid_sec=.5, workers=4):
    plan_supplemental_times([], [], 1.0, budget, grid_sec)
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be 1..8")
    output = checked_output(output)
    output.mkdir(parents=True, exist_ok=False)
    if len({s.qid for s in samples}) != len(samples) or not samples:
        raise ValueError("Nonempty samples with unique qids required")
    protocol = {"version": "sampling-probe-v2", "mock": use_mock,
                "qids": [s.qid for s in samples], "queries": [s.query for s in samples],
                "video_ids": [s.video_id for s in samples], "supplemental_budget_per_arm": budget,
                "grid_sec": grid_sec, "exclude_existing_within_sec": .2,
                "query_policy": "1/4 global; 3 coarse-score anchors >=8s apart; +/-4s local",
                "groups": GROUPS, "nms_iou": .9, "top_k": 5, "short_lengths_sec": [2,4,8],
                "vlm_calls": 0, "config": cfg.model_dump(mode="json"),
                "paired_failures": "Any arm failure makes all groups empty for that query",
                "cost": "Union physically encoded once; logical additional frames reported per arm",
                "cache": "video SHA + pipeline + model + versions + image config + exact timestamps",
                "selection": "Frozen before evaluation; sampling and candidates do not consume GT"}
    threads = None
    if not use_mock:
        import torch
        threads = torch.get_num_threads()
    protocol["runtime"] = {"python":sys.version,"torch":optional_version("torch"),
                           "open_clip":optional_version("open-clip-torch"),"numpy":np.__version__,
                           "torch_threads":threads,"extraction_workers":workers}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    preds = {group: {} for group in GROUPS}
    rows, errors, records = [], [], []
    embedder = None if use_mock else build_embedder(cfg, use_mock=False)
    if not use_mock and embedder is None:
        raise RuntimeError("Real OpenCLIP unavailable; refusing silent Mock")
    for number, sample in enumerate(samples, 1):
        start = time.perf_counter()
        try:
            index = (ingest(sample.video_path, video_id=sample.video_id, srt_path=sample.srt_path,
                            config=cfg) if use_mock else load_index(sample.video_id, cfg))
            if use_mock:
                current = MockEmbedding(index.content_hints, index.frames)
                base_mat = current.encode_images([f.image_path for f in index.frames])
            else:
                current = embedder
                loaded = load_embeddings(index.cache_dir)
                if loaded is None:
                    raise RuntimeError("Existing coarse embeddings required")
                base_mat, meta = loaded
                if (meta.get("model_id") != current.model_id
                        or meta.get("backend") != "openclip"
                        or meta.get("frame_ids") != [f.frame_id for f in index.frames]):
                    raise RuntimeError("Coarse feature cache mismatch")
            times = [f.timestamp_sec for f in index.frames]
            text = current.encode_texts([sample.query])[0]
            scores = base_mat @ text
            plans = plan_supplemental_times(times, scores.tolist(), index.asset.duration_sec,
                                            budget, grid_sec)
            union = sorted(set(plans["uniform"] + plans["query"]))
            fs, mat, cache_hit, extract_sec, encode_sec = extra_features(
                index, union, current, cfg, use_mock, workers)
            # Plans are seek requests; observations use the actual source PTS.
            observed = supplemental_observations(union, fs, (mat @ text).tolist())
            groups = {}
            for group in GROUPS:
                arm = group.split("_")[0] if "_" in group else None
                points = sorted(zip(times, scores.tolist()))
                if arm:
                    points += [observed[t] for t in plans[arm]]
                    points.sort()
                pool, selected = supplemental_proposals(
                    [p[0] for p in points], [p[1] for p in points], index.asset.duration_sec,
                    include_short=group == "E4S" or group.endswith("_short"))
                groups[group] = (pool, selected)
            # GT is used only after all acquisition plans and candidate windows exist.
            query_rows, query_predictions = [], {}
            for group, (pool, selected) in groups.items():
                query_predictions[group] = [[w.start_sec,w.end_sec,w.score] for w in selected]
                all_ious = oracle_details(pool, sample.gt_windows)
                top5_ious = oracle_details(selected, sample.gt_windows)
                top1_ious = oracle_details(selected[:1], sample.gt_windows)
                short = [i for i,g in enumerate(sample.gt_windows) if g.duration_sec <= 10]
                arm = group.split("_")[0] if "_" in group else None
                query_rows.append({"qid":sample.qid,"video_id":sample.video_id,"group":group,
                    "base_frames":len(times),"additional_frames":len(plans[arm]) if arm else 0,
                    "top1_iou":max(top1_ious,default=0),"top5_oracle_iou":max(top5_ious,default=0),
                    "all_oracle_iou":max(all_ious,default=0),"n_candidates":len(pool),
                    "short_gt":len(short),"short_all_hits":sum(all_ious[i]>=.5 for i in short),
                    "short_top5_hits":sum(top5_ious[i]>=.5 for i in short),
                    "short_top1_hits":sum(top1_ious[i]>=.5 for i in short)})
            record = {"qid":sample.qid,"query":sample.query,"video_id":sample.video_id,
                      "pipeline_version":PIPELINE_VERSION,"model_id":current.model_id,
                      "supplemental_source_times_sec":[f.timestamp_sec for f in fs],
                      "base_times_sec":times,"base_scores":scores.tolist(),"plans":plans,
                      "union_frames":len(union),"physically_encoded_frames":0 if cache_hit else len(union),
                      "feature_cache_hit":cache_hit,"union_extraction_sec":extract_sec,
                      "union_encoding_sec":encode_sec,"elapsed_sec":time.perf_counter()-start,
                      "groups":{g:{"all_windows":[w.model_dump() for w in ws[0]],
                                    "top5":query_predictions[g]} for g,ws in groups.items()}}
            with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False)+"\n")
            records.append(record)
            rows.extend(query_rows)
            for group in GROUPS:
                preds[group][sample.qid] = query_predictions[group]
            print(f"{number}/{len(samples)} qid={sample.qid} added={len(plans['query'])} "
                  f"union={len(union)} cache={cache_hit} elapsed={record['elapsed_sec']:.1f}s", flush=True)
        except Exception as exc:
            errors.append({"qid":sample.qid,"error_type":type(exc).__name__,"message":str(exc),
                           "elapsed_sec":time.perf_counter()-start})
            for group in GROUPS:
                preds[group][sample.qid] = []
            print(f"{number}/{len(samples)} qid={sample.qid} FAILED {type(exc).__name__}: {exc}",flush=True)
    gt = {s.qid:{"relevant_windows":[[w.start_sec,w.end_sec] for w in s.gt_windows]} for s in samples}
    metrics = {g:dict(eval_mr_brief(p,gt)) for g,p in preds.items()}
    diagnostics = {}
    for group in GROUPS:
        selected = [r for r in rows if r["group"] == group]
        total = sum(r["short_gt"] for r in selected)
        diagnostics[group] = {"completed":len(selected),"short_gt_intervals_completed":total,
            "mean_additional_frames_completed":float(np.mean([r["additional_frames"] for r in selected])) if selected else None,
            "all_query_oracle_r1_05_completed":sum(r["all_oracle_iou"]>=.5 for r in selected)/len(selected) if selected else None,
            "top5_query_oracle_r1_05_completed":sum(r["top5_oracle_iou"]>=.5 for r in selected)/len(selected) if selected else None,
            **{kind: {"hits":sum(r[kind] for r in selected),"total":total,
                       "rate":sum(r[kind] for r in selected)/total if total else None}
               for kind in ("short_all_hits","short_top5_hits","short_top1_hits")}}
    paired = {}
    by_row = {(r["qid"],r["group"]):r for r in rows}
    for left, right in (("E4","uniform_short"),("E4","query_short"),
                        ("uniform_curve","query_curve"),("uniform_short","query_short")):
        pairs = [(by_row[s.qid,left],by_row[s.qid,right]) for s in samples if (s.qid,left) in by_row]
        paired[f"{left}_vs_{right}"] = {"matched_completed":len(pairs),
            "right_new_hits":sum(a["top1_iou"]<.5<=b["top1_iou"] for a,b in pairs),
            "right_lost_hits":sum(b["top1_iou"]<.5<=a["top1_iou"] for a,b in pairs)}
    summary = {"n_samples":len(samples),"completed":len(records),
               "coverage":len(records)/len(samples),"incomplete":bool(errors),"mock":use_mock,
               "metrics":metrics,"diagnostics_completed_only":diagnostics,"paired":paired,
               "physical_work":{"extracted_frames":sum(r["physically_encoded_frames"] for r in records),
                    "encoded_frames":sum(r["physically_encoded_frames"] for r in records),
                    "union_extraction_sec":sum(r["union_extraction_sec"] for r in records),
                    "union_encoding_sec":sum(r["union_encoding_sec"] for r in records)},
               "errors":errors,"vlm_calls":0}
    summary["all_query_elapsed_sec"] = sum(r["elapsed_sec"] for r in records + errors)
    summary["physical_work"]["scope"] = "completed queries only; failed-query elapsed is retained separately"
    (output/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (output/"errors.json").write_text(json.dumps(errors,indent=2),encoding="utf-8")
    if rows:
        with (output/"per_sample.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(json.dumps({"output":str(output),"coverage":summary["coverage"],"metrics":metrics},indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--limit",type=int,default=80)
    parser.add_argument("--mock",action="store_true")
    parser.add_argument("--budget",type=int,default=32)
    parser.add_argument("--grid-sec",type=float,default=.5)
    args=parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    cfg=Config.load(ROOT/"configs/qvhighlights_v2.yaml",apply_env=False).resolved()
    if args.mock:
        cfg.embedding.backend="mock"
    if args.mock and args.dataset is None:
        samples=dataset_from_labels(ROOT/"assets/demo/sample.labels.json",ROOT/"assets/demo/sample.mp4",
                                   ROOT/"assets/demo/sample.srt")
    else:
        samples=load_dataset(args.dataset or ROOT/"data/qvhighlights_test/dataset.json")
    if not args.mock:
        import torch
        torch.set_num_threads(4)
    summary=run_probe(samples[:args.limit],cfg,ROOT/"outputs/sampling_probe"/new_job_id(),
                      args.mock,args.budget,args.grid_sec)
    if summary["incomplete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
