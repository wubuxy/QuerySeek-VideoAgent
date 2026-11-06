"""Read-only source-run audit, clustered paired intervals, optional time-axis control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np

from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output,feature_signature,oracle_details,supplemental_observations
from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.embeddings.cache import load_embeddings
from queryseek.embeddings.factory import build_embedder
from queryseek.eval.qvhighlights import compute_ap, eval_mr_brief
from queryseek.eval.runner import load_dataset
from queryseek.ingest.pipeline import load_index
from queryseek.ingest.frames import load_frames_index
from queryseek.retrieval.sampling_refinement import time_aligned_proposals
from queryseek.tools.export import new_job_id


def clustered_delta(left, right, sources, seed=12, repeats=5000):
    """Bootstrap whole source-video clusters, retaining paired query outcomes."""
    a,b=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if len(a)!=len(b) or len(a)!=len(sources) or not len(a):
        raise ValueError("Nonempty matched arrays required")
    names=sorted(set(sources)); sums=[]; counts=[]
    for name in names:
        mask=np.array([s==name for s in sources])
        sums.append(float((b-a)[mask].sum()));counts.append(int(mask.sum()))
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(names),size=(repeats,len(names)))
    deltas=np.asarray(sums)[draws].sum(1)/np.asarray(counts)[draws].sum(1)
    return {"delta":float(np.mean(b-a)),"ci95":np.quantile(deltas,[.025,.975]).tolist(),
            "n_queries":len(a),"n_source_videos":len(names),"seed":seed,"resamples":repeats,
            "unit":"fraction, not percentage points"}


def audit_rows(records,samples,groups):
    by_q={s.qid:s for s in samples}
    diagnostics={};top1={g:[] for g in groups};aps={g:[] for g in groups};sources=[]
    for record in records:
        sources.append(source_video_id(record["video_id"]))
    for group in groups:
        total=all_hits=top5_hits=0;full_oracle=[]
        for record in records:
            sample=by_q[record["qid"]]
            data=record["groups"][group]
            from queryseek.schemas import TimeWindow
            pool=[TimeWindow.model_validate(w) for w in data["all_windows"]]
            selected=[TimeWindow(start_sec=w[0],end_sec=w[1],score=w[2]) for w in data["top5"]]
            all_iou=oracle_details(pool,sample.gt_windows)
            top5_iou=oracle_details(selected,sample.gt_windows)
            first=oracle_details(selected[:1],sample.gt_windows)
            top1[group].append(max(first,default=0))
            full_gt=[(g.start_sec,g.end_sec) for g in sample.gt_windows if 0<g.duration_sec<=150]
            aps[group].append(float(np.mean(compute_ap(full_gt,
                              [(w.start_sec,w.end_sec,w.score) for w in selected]))) if full_gt else 0.)
            full_oracle.append(max(all_iou,default=0)>=.5)
            for i,gt in enumerate(sample.gt_windows):
                if gt.duration_sec<=10:
                    total+=1;all_hits+=all_iou[i]>=.5;top5_hits+=top5_iou[i]>=.5
        diagnostics[group]={"short_gt":total,"short_all_hits":int(all_hits),
                            "short_top5_hits":int(top5_hits),
                            "all_query_oracle_r1_05":float(np.mean(full_oracle))}
    return diagnostics,top1,sources,aps


def aligned_records(records,samples,cfg):
    """Reuse the exact supplemental cache; no extraction, image encoding, or GT in proposals."""
    emb=build_embedder(cfg,use_mock=False)
    if emb is None:raise RuntimeError("Real OpenCLIP required")
    by_q={s.qid:s for s in samples};out=[]
    for record in records:
        sample=by_q[record["qid"]]
        if record.get("pipeline_version")!=PIPELINE_VERSION or record.get("model_id")!=emb.model_id:
            raise RuntimeError("Source run uses a different pipeline/model; historical runs require their frozen implementation")
        index=load_index(sample.video_id,cfg)
        union=sorted(set(record["plans"]["uniform"]+record["plans"]["query"]))
        signature=feature_signature(index,emb.model_id,union,cfg.video.max_frame_side_px)
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        directory=checked_output(ROOT/"outputs/sampling_feature_cache"/key)
        if json.loads((directory/"sampling_meta.json").read_text(encoding="utf-8"))!=signature:
            raise RuntimeError("Feature signature mismatch")
        loaded=load_embeddings(directory)
        if loaded is None:raise RuntimeError("Frozen supplemental features unavailable")
        mat,meta=loaded
        if (meta.get("model_id")!=emb.model_id or meta.get("backend")!="openclip"
                or meta.get("frame_ids")!=[f"supplement_{i:04d}" for i in range(len(union))]):
            raise RuntimeError("Feature metadata mismatch")
        frames=load_frames_index(directory/"frames.jsonl")
        observed=supplemental_observations(union,frames,(mat@emb.encode_texts([sample.query])[0]).tolist())
        groups={}
        for arm in ("base","uniform","query"):
            points=list(zip(record["base_times_sec"],record["base_scores"]))
            if arm!="base":points.extend(observed[t] for t in record["plans"][arm])
            points.sort()
            for short in (False,True):
                pool,selected=time_aligned_proposals([t for t,s in points],[s for t,s in points],
                                                     index.asset.duration_sec,short)
                group=f"aligned_{arm}_{'short' if short else 'curve'}"
                groups[group]={"all_windows":[w.model_dump() for w in pool],
                               "top5":[[w.start_sec,w.end_sec,w.score] for w in selected]}
        out.append({"qid":sample.qid,"video_id":sample.video_id,"groups":groups})
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--dataset",type=Path,default=ROOT/"data/qvhighlights_test/dataset.json")
    parser.add_argument("--time-aligned",action="store_true")
    args=parser.parse_args()
    run=checked_output(args.run)
    protocol=json.loads((run/"protocol.json").read_text(encoding="utf-8"))
    original=json.loads((run/"metrics.json").read_text(encoding="utf-8"))
    if original["coverage"]!=1 or original["mock"]:
        raise ValueError("Audit requires complete real experiment")
    records=[json.loads(line) for line in (run/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    dataset=load_dataset(args.dataset)
    by_q={s.qid:s for s in dataset}
    samples=[by_q[q] for q in protocol["qids"]]
    if ([r["qid"] for r in records]!=protocol["qids"]
            or [s.query for s in samples]!=protocol["queries"]
            or [s.video_id for s in samples]!=protocol["video_ids"]):
        raise ValueError("Source run and dataset are not matched")
    out=checked_output(ROOT/"outputs/sampling_analysis"/new_job_id());out.mkdir(parents=True)
    control={"source_run":str(run),"dataset":str(args.dataset.resolve()),
             "dataset_sha256":hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
             "time_aligned":args.time_aligned,"step_sec":1,"interpolation":"linear score imputation",
             "image_encoding_calls":0,"gt_used_by_candidate_builder":False,
             "status":"development mechanism diagnosis, not independent validation"}
    (out/"protocol.json").write_text(json.dumps(control,indent=2),encoding="utf-8")
    if args.time_aligned:
        import torch
        torch.set_num_threads(4)
        cfg=Config.model_validate(protocol["config"])
        original_records=records
        records=aligned_records(records,samples,cfg)
        for aligned,source in zip(records,original_records):
            aligned["groups"]["E4"]=source["groups"]["E4"]
            aligned["groups"]["E4S"]=source["groups"]["E4S"]
        pairs=(("aligned_base_curve","aligned_uniform_curve"),
               ("aligned_base_curve","aligned_query_curve"),
               ("aligned_uniform_curve","aligned_query_curve"),
               ("aligned_uniform_short","aligned_query_short"),
               ("E4","aligned_query_curve"),("E4","aligned_uniform_curve"))
    else:
        pairs=(("E4","uniform_curve"),("E4","query_curve"),
               ("uniform_curve","query_curve"),("uniform_short","query_short"))
    groups=list(records[0]["groups"])
    gt={s.qid:{"relevant_windows":[[w.start_sec,w.end_sec] for w in s.gt_windows]} for s in samples}
    metrics={g:dict(eval_mr_brief({r["qid"]:r["groups"][g]["top5"] for r in records},gt)) for g in groups}
    diagnostics,top1,sources,aps=audit_rows(records,samples,groups)
    paired={}
    for left,right in pairs:
        a=np.asarray(top1[left]);b=np.asarray(top1[right])
        paired[f"{left}_vs_{right}"]={
            "r1_05":clustered_delta((a>=.5).astype(float),(b>=.5).astype(float),sources),
            "custom_miou":clustered_delta(a,b,sources),
            "average_map":clustered_delta(aps[left],aps[right],sources),
            "new_hits":int(np.sum((a<.5)&(b>=.5))),"lost_hits":int(np.sum((b<.5)&(a>=.5)))}
    report={"n_samples":len(samples),"coverage":1.,"metrics":metrics,"diagnostics":diagnostics,
            "paired":paired,"independent":False,"image_encoding_calls":0}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
    print(json.dumps({"output":str(out),"metrics":metrics,"paired":paired},indent=2))


if __name__=="__main__":main()
