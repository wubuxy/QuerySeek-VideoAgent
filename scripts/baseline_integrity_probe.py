"""Paired 2x2 audit: historical/corrected source sampling x GELU/QuickGELU.

Reads v1 indices explicitly; corrected features live in outputs/ and do not
overwrite any historical index. This evaluates fixes, not algorithm innovations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np

from scripts.analyze_sampling_probe import audit_rows,clustered_delta
from scripts.sampling_probe import checked_output,optional_version
from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.embeddings.cache import load_embeddings,save_embeddings
from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.eval.runner import load_dataset
from queryseek.ingest.cache import compute_cache_key
from queryseek.ingest.ffprobe import asset_from_probe
from queryseek.ingest.frames import select_frames,load_frames_index,save_frames_index
from queryseek.ingest.pipeline import VideoIndex
from queryseek.retrieval.curve_spans import curve_spans,select_curve_windows
from queryseek.schemas import Scene
from queryseek.tools.export import new_job_id

GROUPS=("historical","activation_fix","sampling_fix","both_fixes")


def load_v1_index(sample,cfg):
    """Explicit forensic reader: never silently accept v1 as the current cache."""
    asset=asset_from_probe(sample.video_id,sample.video_path)
    directory=cfg.cache_root/asset.sha256
    meta=json.loads((directory/"index_meta.json").read_text(encoding="utf-8"))
    snapshot=cfg.snapshot_for_cache();snapshot["pipeline"]="1"
    if (meta.get("pipeline_version")!="1"
            or meta.get("cache_key")!=compute_cache_key(asset.sha256,snapshot)):
        raise RuntimeError("Historical v1 index/config mismatch")
    scenes=[Scene.model_validate(s) for s in json.loads((directory/"scenes.json").read_text(encoding="utf-8"))]
    return VideoIndex(asset=asset,cache_dir=directory,frames=load_frames_index(directory/"frames.jsonl"),
                      scenes=scenes,cache_key=meta["cache_key"],cache_hit=True)


def corrected_index(old,cfg):
    signature={"video_sha256":old.asset.sha256,"pipeline_version":PIPELINE_VERSION,
               "sampling_version":"source-pts-select-v3","steps":cfg.snapshot_for_cache(),
               "scene_sha256":hashlib.sha256((old.cache_dir/"scenes.json").read_bytes()).hexdigest()}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    directory=checked_output(ROOT/"outputs/baseline_integrity_cache"/key)
    meta_path=directory/"sampling_meta.json"
    if meta_path.is_file():
        if json.loads(meta_path.read_text(encoding="utf-8"))!=signature:raise RuntimeError("Sampling cache mismatch")
        frames=load_frames_index(directory/"frames.jsonl")
        extraction_sec=0.;hit=True
    else:
        directory.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        frames=select_frames(old.asset.path,directory/"frames",old.scenes,old.asset.duration_sec,
                            cfg.video.coarse_fps,cfg.video.frame_sampling,cfg.video.max_frame_side_px)
        extraction_sec=time.perf_counter()-start;hit=False
        save_frames_index(frames,directory/"frames.jsonl")
        meta_path.write_text(json.dumps(signature,indent=2),encoding="utf-8")
    if not frames or any(not 0<=f.timestamp_sec<old.asset.duration_sec for f in frames):
        raise RuntimeError("Corrected index violates source-time contract")
    return VideoIndex(asset=old.asset,cache_dir=directory,frames=frames,scenes=old.scenes),hit,extraction_sec


def cached_mat(index,embedder,label):
    signature={"video_sha256":index.asset.sha256,"pipeline_version":PIPELINE_VERSION,
               "model_id":embedder.model_id,"frame_ids":[f.frame_id for f in index.frames],
               "frame_paths":[f.image_path for f in index.frames],
               "frame_times_sec":[f.timestamp_sec for f in index.frames],
               "max_side":embedder.max_side,"open_clip":optional_version("open-clip-torch"),
               "preprocess":"adapter-v2", "label":label}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    directory=checked_output(ROOT/"outputs/baseline_integrity_features"/key)
    metadata=directory/"audit_meta.json"
    if metadata.is_file():
        if json.loads(metadata.read_text(encoding="utf-8"))!=signature:raise RuntimeError("Feature signature mismatch")
        loaded=load_embeddings(directory)
        if loaded is None or loaded[1].get("frame_ids")!=signature["frame_ids"]:
            raise RuntimeError("Feature cache incomplete")
        return loaded[0],True,0.
    directory.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();mat=embedder.encode_images(signature["frame_paths"]);elapsed=time.perf_counter()-start
    save_embeddings(directory,mat,signature["frame_ids"],{"backend":"openclip","model_id":embedder.model_id})
    metadata.write_text(json.dumps(signature,indent=2),encoding="utf-8")
    return mat,False,elapsed


def run_probe(samples,cfg,out):
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    protocol={"version":"baseline-integrity-v1","qids":[s.qid for s in samples],
              "queries":[s.query for s in samples],"video_ids":[s.video_id for s in samples],
              "groups":GROUPS,"pipeline_version":PIPELINE_VERSION,"config":cfg.model_dump(mode="json"),
              "historical":"v1 images + cached GELU image embeddings, matched legacy GELU text",
              "activation_fix":"same v1 images/times, QuickGELU in both image/text towers",
              "sampling_fix":"source-PTS grid selection and corrected scene timestamps, GELU towers",
              "both_fixes":"source-PTS observations, QuickGELU towers",
              "retrieval":"unchanged E4 quantiles/smoothing/support/NMS=.9/top5",
              "coupling":"sampling fix changes image selection, PTS and potentially frame counts; not a relabel-only intervention",
              "cost":"historical visual cache reused; all other image/text pairs re-encoded together",
              "vlm_calls":0,"independent":False}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    started=time.perf_counter()
    # Mismatch is intentional ONLY in the historical counterfactual.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="QuickGELU mismatch.*")
        legacy=OpenCLIPEmbedding(force_quick_gelu=False,max_side=cfg.video.max_frame_side_px)
    correct=OpenCLIPEmbedding(force_quick_gelu=True,max_side=cfg.video.max_frame_side_px)
    activation={label:{"vision":type(emb._model.visual.transformer.resblocks[0].mlp.gelu).__name__,
                       "text":type(emb._model.transformer.resblocks[0].mlp.gelu).__name__}
                for label,emb in (("legacy",legacy),("correct",correct))}
    if activation["legacy"]!={"vision":"GELU","text":"GELU"} or activation["correct"]!={"vision":"QuickGELU","text":"QuickGELU"}:
        raise RuntimeError("Actual activation does not match frozen protocol")
    (out/"activation_audit.json").write_text(json.dumps(activation,indent=2),encoding="utf-8")
    loading_sec=time.perf_counter()-started
    records=[];errors=[]
    for number,sample in enumerate(samples,1):
        start=time.perf_counter()
        try:
            old=load_v1_index(sample,cfg)
            loaded=load_embeddings(old.cache_dir)
            expected=f"openclip:{cfg.embedding.model_name}:{cfg.embedding.pretrained}"
            if loaded is None or loaded[1].get("model_id")!=expected or loaded[1].get("frame_ids")!=[f.frame_id for f in old.frames]:
                raise RuntimeError("Legacy features incompatible")
            new,sampling_hit,extract_sec=corrected_index(old,cfg)
            matrices={"historical":loaded[0]};cache_hits={"historical":True};encoding_sec={"historical":0.}
            for label,index,emb in (("activation_fix",old,correct),("sampling_fix",new,legacy),("both_fixes",new,correct)):
                matrices[label],cache_hits[label],encoding_sec[label]=cached_mat(index,emb,label)
            texts={"legacy":legacy.encode_texts([sample.query])[0],"correct":correct.encode_texts([sample.query])[0]}
            groups={};observations={}
            for label in GROUPS:
                index=old if label in ("historical","activation_fix") else new
                text=texts["legacy" if label in ("historical","sampling_fix") else "correct"]
                times=[f.timestamp_sec for f in index.frames];scores=matrices[label]@text
                pool=curve_spans(times,scores,index.asset.duration_sec);selected=select_curve_windows(pool)
                groups[label]={"all_windows":[w.model_dump() for w in pool],
                               "top5":[[w.start_sec,w.end_sec,w.score] for w in selected]}
                observations[label]={"times_sec":times,"scores":scores.tolist()}
            record={"qid":sample.qid,"video_id":sample.video_id,"query":sample.query,"groups":groups,
                    "observations":observations,"old_frames":len(old.frames),"new_frames":len(new.frames),
                    "cache_hits":cache_hits,"sampling_cache_hit":sampling_hit,"extraction_sec":extract_sec,
                    "encoding_sec":encoding_sec,"new_encoded_frames":sum(matrices[g].shape[0] for g in GROUPS if not cache_hits[g]),
                    "elapsed_sec":time.perf_counter()-start}
            with (out/"records.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(record)+"\n")
            records.append(record)
            print(f"{number}/{len(samples)} qid={sample.qid} frames={len(old.frames)}->{len(new.frames)} elapsed={record['elapsed_sec']:.1f}s",flush=True)
        except Exception as exc:
            errors.append({"qid":sample.qid,"error_type":type(exc).__name__,"message":str(exc),"elapsed_sec":time.perf_counter()-start})
            print(f"FAILED {sample.qid}: {type(exc).__name__}: {exc}",flush=True)
    by_q={r["qid"]:r for r in records}
    gt={s.qid:{"relevant_windows":[[w.start_sec,w.end_sec] for w in s.gt_windows]} for s in samples}
    metrics={g:dict(eval_mr_brief({s.qid:by_q[s.qid]["groups"][g]["top5"] if s.qid in by_q else [] for s in samples},gt)) for g in GROUPS}
    diagnostics={};paired={}
    if records:
        diagnostics,top1,sources,aps=audit_rows(records,samples,GROUPS)
        for a,b in (("historical","activation_fix"),("historical","sampling_fix"),("historical","both_fixes"),("sampling_fix","both_fixes")):
            ai=np.asarray(top1[a]);bi=np.asarray(top1[b])
            paired[f"{a}_vs_{b}"]={"new_hits":int(np.sum((ai<.5)&(bi>=.5))),"lost_hits":int(np.sum((ai>=.5)&(bi<.5))),
                  "r1_05":clustered_delta((ai>=.5).astype(float),(bi>=.5).astype(float),sources),
                  "average_map":clustered_delta(aps[a],aps[b],sources)}
    report={"n_samples":len(samples),"completed":len(records),"coverage":len(records)/len(samples),
            "incomplete":bool(errors),"metrics":metrics,"paired_completed_only":paired,"diagnostics_completed_only":diagnostics,
            "errors":errors,"vlm_calls":0,"model_loading_sec":loading_sec,
            "query_elapsed_sec":sum(r["elapsed_sec"] for r in records+errors),
            "new_encoded_frames":sum(r["new_encoded_frames"] for r in records),"independent":False}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out),"coverage":report["coverage"],"metrics":metrics},indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=ROOT/"data/qvhighlights_test/dataset.json")
    parser.add_argument("--limit",type=int,default=80)
    args=parser.parse_args()
    if args.limit<=0:raise ValueError("Positive limit required")
    import torch
    torch.set_num_threads(4)
    cfg=Config.load(ROOT/"configs/qvhighlights_v2.yaml",apply_env=False).resolved()
    report=run_probe(load_dataset(args.dataset)[:args.limit],cfg,ROOT/"outputs/baseline_integrity_probe"/new_job_id())
    if report["incomplete"]:raise SystemExit(1)


if __name__=="__main__":main()
