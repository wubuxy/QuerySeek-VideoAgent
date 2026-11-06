"""Preserve E4 pools and compare shared-scale/local-context ranking on frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np

from scripts.analyze_sampling_probe import audit_rows,clustered_delta
from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output,feature_signature,supplemental_observations
from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.embeddings.cache import load_embeddings
from queryseek.embeddings.factory import build_embedder
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.eval.runner import dataset_from_labels,load_dataset
from queryseek.ingest.pipeline import load_index
from queryseek.ingest.frames import load_frames_index
from queryseek.retrieval.curve_spans import curve_spans,select_curve_windows,short_peak_spans
from queryseek.retrieval.segment_calibration import calibrated_windows
from queryseek.schemas import TimeWindow
from queryseek.tools.export import new_job_id

SPECS={"shared_base":("base","shared",False),"contrast_base":("base","contrast",False)}
for arm in ("uniform","query"):
    for label,mode,locked in (("raw","legacy",False),("shared","shared",False),
                              ("contrast","contrast",False),("locked_contrast","contrast",True)):
        SPECS[f"{arm}_{label}"]=(arm,mode,locked)


def load_observations(sample,cfg,embedder,source_record=None,use_mock=False):
    index=load_index(sample.video_id,cfg)
    base_times=[f.timestamp_sec for f in index.frames]
    if use_mock:
        encoder=MockEmbedding(index.content_hints,index.frames)
        base_mat=encoder.encode_images([f.image_path for f in index.frames])
    else:
        encoder=embedder
        loaded=load_embeddings(index.cache_dir)
        if loaded is None:raise RuntimeError("Coarse features required")
        base_mat,meta=loaded
        if (meta.get("model_id")!=encoder.model_id or meta.get("backend")!="openclip"
                or meta.get("frame_ids")!=[f.frame_id for f in index.frames]):
            raise RuntimeError("Coarse cache mismatch")
    text=encoder.encode_texts([sample.query])[0]
    base_scores=(base_mat@text).tolist()
    observations={"base":(base_times,base_scores)}
    if source_record is not None:
        if source_record.get("pipeline_version")!=PIPELINE_VERSION or source_record.get("model_id")!=encoder.model_id:
            raise RuntimeError("Source run uses a different pipeline/model; historical runs require their frozen implementation")
        if (source_record["video_id"]!=sample.video_id or source_record["query"]!=sample.query
                or source_record["base_times_sec"]!=base_times
                or not np.allclose(base_scores,source_record["base_scores"],atol=1e-6,rtol=0)):
            raise RuntimeError("Frozen source observations differ from current coarse cache")
        union=sorted(set(source_record["plans"]["uniform"]+source_record["plans"]["query"]))
        signature=feature_signature(index,encoder.model_id,union,cfg.video.max_frame_side_px)
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        directory=checked_output(ROOT/"outputs/sampling_feature_cache"/key)
        if json.loads((directory/"sampling_meta.json").read_text(encoding="utf-8"))!=signature:
            raise RuntimeError("Supplemental signature mismatch")
        loaded=load_embeddings(directory)
        if loaded is None:raise RuntimeError("Supplemental features required; no image recomputation")
        mat,meta=loaded
        if (meta.get("model_id")!=encoder.model_id
                or meta.get("backend")!=("mock" if use_mock else "openclip")
                or meta.get("frame_ids")!=[f"supplement_{i:04d}" for i in range(len(union))]):
            raise RuntimeError("Supplemental cache mismatch")
        frames=load_frames_index(directory/"frames.jsonl")
        observed=supplemental_observations(union,frames,(mat@text).tolist())
        for arm in ("uniform","query"):
            points=sorted(list(zip(base_times,base_scores))+[observed[t] for t in source_record["plans"][arm]])
            observations[arm]=([t for t,s in points],[s for t,s in points])
    return observations,index.asset.duration_sec


def build_groups(observations,duration,spec_names):
    """No GT argument; E4 is recomputed and original geometry retained in every pool."""
    bt,bs=observations["base"]
    original=curve_spans(bt,bs,duration)
    baseline=select_curve_windows(original)
    groups={"E4":{"all_windows":[w.model_dump() for w in original],
                  "top5":[[w.start_sec,w.end_sec,w.score] for w in baseline]}}
    features={}
    for name in spec_names:
        arm,mode,locked=SPECS[name]
        times,scores=observations[arm]
        additions=short_peak_spans(times,scores,duration) if arm!="base" else []
        pool,selected,details=calibrated_windows(original,additions,bt,bs,times,scores,duration,
                         mode,baseline[0] if locked and baseline else None)
        if not {(w.start_sec,w.end_sec) for w in original}.issubset(
                {(w.start_sec,w.end_sec) for w in pool}):
            raise AssertionError("Original candidate geometry lost")
        groups[name]={"all_windows":[w.model_dump() for w in pool],
                      "top5":[[w.start_sec,w.end_sec,w.score] for w in selected]}
        features[name]=details
    return groups,features


def run_experiment(samples,cfg,out,source_records=None,use_mock=False,spec_names=None,stage="development"):
    spec_names=list(SPECS) if spec_names is None else list(spec_names)
    if not samples or len({s.qid for s in samples})!=len(samples) or any(n not in SPECS for n in spec_names):
        raise ValueError("Valid nonempty unique samples/specs required")
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    protocol={"version":"preserved-calibration-v1","stage":stage,"mock":use_mock,
              "qids":[s.qid for s in samples],"queries":[s.query for s in samples],
              "video_ids":[s.video_id for s in samples],"specs":{n:SPECS[n] for n in spec_names},
              "reference":"raw coarse median/std; same reference for original and supplemental",
              "support":"time-weighted positive z * sqrt(window seconds)",
              "contrast_weight":2,"context_margin_sec":8,"top_k":5,"nms_iou":.9,
              "short_lengths_sec":[2,4,8],"image_encoding_calls":0,"vlm_calls":0,
              "locked_policy":"original E4 top1 retained by construction; do not claim R1 improvement",
              "acceptance":"R1@0.5 >= E4 and mAP >= E4 and short mAP > E4; prefer highest mAP",
              "config":cfg.model_dump(mode="json")}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    emb=None if use_mock else build_embedder(cfg,use_mock=False)
    if not use_mock and emb is None:raise RuntimeError("Real OpenCLIP required; no silent Mock")
    records=[];errors=[];failed_elapsed=0.
    for i,sample in enumerate(samples,1):
        started=time.perf_counter()
        try:
            source=(source_records or {}).get(sample.qid)
            if any(SPECS[n][0]!="base" for n in spec_names) and source is None:
                raise ValueError("Supplemental arm requires frozen source record")
            observations,duration=load_observations(sample,cfg,emb,source,use_mock)
            groups,features=build_groups(observations,duration,spec_names)
            if source and groups["E4"]["top5"]!=source["groups"]["E4"]["top5"]:
                if not np.allclose(groups["E4"]["top5"],source["groups"]["E4"]["top5"],atol=1e-5,rtol=0):
                    raise RuntimeError("Original E4 baseline not reproduced")
            record={"qid":sample.qid,"query":sample.query,"video_id":sample.video_id,
                    "observations":observations,"duration_sec":duration,"groups":groups,
                    "features":features,"elapsed_sec":time.perf_counter()-started}
            with (out/"records.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(record,ensure_ascii=False)+"\n")
            records.append(record)
        except Exception as exc:
            elapsed=time.perf_counter()-started;failed_elapsed+=elapsed
            errors.append({"qid":sample.qid,"error_type":type(exc).__name__,"message":str(exc),"elapsed_sec":elapsed})
            print(f"FAILED qid={sample.qid}: {type(exc).__name__}: {exc}",flush=True)
        if i%10==0 or i==len(samples):print(f"completed {i}/{len(samples)} errors={len(errors)}",flush=True)
    groups=["E4",*spec_names]
    gt={s.qid:{"relevant_windows":[[w.start_sec,w.end_sec] for w in s.gt_windows]} for s in samples}
    by_q={r["qid"]:r for r in records}
    predictions={g:{s.qid:by_q[s.qid]["groups"][g]["top5"] if s.qid in by_q else [] for s in samples} for g in groups}
    metrics={g:dict(eval_mr_brief(p,gt)) for g,p in predictions.items()}
    diagnostics={};paired={}
    if records:
        diagnostics,top1,sources,aps=audit_rows(records,samples,groups)
        for group in spec_names:
            a=np.asarray(top1["E4"]);b=np.asarray(top1[group])
            paired[group]={"new_hits":int(np.sum((a<.5)&(b>=.5))),"lost_hits":int(np.sum((a>=.5)&(b<.5))),
                           "r1_05":clustered_delta((a>=.5).astype(float),(b>=.5).astype(float),sources),
                           "average_map":clustered_delta(aps["E4"],aps[group],sources)}
    baseline=metrics["E4"]
    eligible=[g for g in spec_names if metrics[g]["MR-full-R1@0.5"]>=baseline["MR-full-R1@0.5"]
              and metrics[g]["MR-full-mAP"]>=baseline["MR-full-mAP"]
              and metrics[g]["MR-short-mAP"]>baseline["MR-short-mAP"]]
    winner=max(eligible,key=lambda g:metrics[g]["MR-full-mAP"]) if eligible and not errors else None
    report={"n_samples":len(samples),"completed":len(records),"coverage":len(records)/len(samples),
            "incomplete":bool(errors),"stage":stage,"mock":use_mock,"metrics":metrics,
            "diagnostics_completed_only":diagnostics,"paired_completed_only":paired,
            "eligible":eligible,"development_winner":winner if stage=="development" else None,
            "errors":errors,"image_encoding_calls":0,"vlm_calls":0,
            "all_query_elapsed_sec":sum(r["elapsed_sec"] for r in records)+failed_elapsed,
            "independent":False}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if stage=="development":
        (out/"frozen_selection.json").write_text(json.dumps({"winner":winner,
            "spec":SPECS[winner] if winner else None,"contrast_weight":2,"context_margin_sec":8,
            "warning":"development selection; not independent or significance-confirmed"},indent=2),encoding="utf-8")
    lines=["# Preserved-candidate calibration", "",f"Stage: {stage}; coverage {len(records)}/{len(samples)}; zero new image encoding/VLM calls.",
           "", "| Method | R1@0.5 | R1@0.7 | Average mAP | Short mAP |", "|---|---:|---:|---:|---:|"]
    for group,m in metrics.items():lines.append(f"| {group} | {m['MR-full-R1@0.5']:.2f} | {m['MR-full-R1@0.7']:.2f} | {m['MR-full-mAP']:.2f} | {m['MR-short-mAP']:.2f} |")
    lines+=["",f"Development winner: {winner if stage=='development' else 'not selected in regression'}.",
            "Locked policies preserve E4 top1 by construction. Cached supplemental observations still have acquisition cost.",
            "Development/bootstrap comparisons are exploratory; no multiple-comparison correction, no independent claims."]
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"coverage":report["coverage"],"metrics":metrics,"winner":winner},indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path)
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--mock",action="store_true")
    parser.add_argument("--spec",choices=list(SPECS))
    parser.add_argument("--stage",choices=("development","regression"),default="development")
    args=parser.parse_args()
    records=None
    if args.source_run:
        source=checked_output(args.source_run)
        protocol=json.loads((source/"protocol.json").read_text(encoding="utf-8"))
        summary=json.loads((source/"metrics.json").read_text(encoding="utf-8"))
        if summary["coverage"]!=1 or summary["mock"]!=args.mock:raise ValueError("Require complete matched real/mock source")
        cfg=Config.model_validate(protocol["config"])
        raw=[json.loads(line) for line in (source/"records.jsonl").read_text(encoding="utf-8").splitlines()]
        if [r["qid"] for r in raw]!=protocol["qids"]:raise ValueError("Source record coverage/order mismatch")
        records={r["qid"]:r for r in raw}
    else:
        cfg=Config.load(ROOT/"configs/qvhighlights_v2.yaml",apply_env=False).resolved()
        if args.mock:cfg.embedding.backend="mock"
    if args.mock and not args.dataset:
        samples=dataset_from_labels(ROOT/"assets/demo/sample.labels.json",ROOT/"assets/demo/sample.mp4",ROOT/"assets/demo/sample.srt")
    else:samples=load_dataset(args.dataset or ROOT/"data/qvhighlights_test/dataset.json")
    if records:
        by_q={s.qid:s for s in samples};samples=[by_q[q] for q in records]
        if any(s.query!=records[s.qid]["query"] or s.video_id!=records[s.qid]["video_id"] for s in samples):
            raise ValueError("Source and dataset identities differ")
    spec_names=[args.spec] if args.spec else (list(SPECS) if records else ["shared_base","contrast_base"])
    if not args.mock:
        import torch
        torch.set_num_threads(4)
    report=run_experiment(samples,cfg,ROOT/"outputs/calibration_probe"/new_job_id(),records,args.mock,spec_names,args.stage)
    if report["incomplete"]:raise SystemExit(1)


if __name__=="__main__":main()
