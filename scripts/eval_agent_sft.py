"""Identical frozen evaluation for the NF4 base and selected policy adapters."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from queryseek.agent.search_policy import initial_branch,rollout
from queryseek.agent.sft_policy import SearchPolicy,SFT_PROMPT
from queryseek.schemas import VideoFrame
from scripts.gpu_agent_experiment import sha,write,within_outputs,score,summarize
from scripts.audit_gpu_agent import audit_episode
from scripts.train_agent_sft import QuantizedBackend


def metrics(rows,cfg):
    result=summarize(rows,{**cfg,"seed":cfg["bootstrap_seed"]})
    result["positive_recovery_effect_within_this_model"]=result.pop("training_gate_pass")
    return result


def replay(package,out):
    package=Path(package);out=within_outputs(out)
    manifest=json.loads((package/"manifest.json").read_text("utf8"))
    protocol=json.loads((out/"protocol.json").read_text("utf8"))
    if sha(package/"manifest.json")!=protocol["manifest_sha256"]:
        raise ValueError("Frozen input manifest changed")
    samples={s["qid"]:s for s in manifest["samples"] if s["stage"]==protocol["stage"]}
    rows=[json.loads(x) for x in (out/"records.jsonl").read_text("utf8").splitlines()]
    required={(qid,method,condition) for qid in samples for method,condition in [
        ("fixed_uniform","global"),("forward_only","hash_half"),("backtracking","hash_half"),
        ("forward_only","mirrored_half"),("backtracking","mirrored_half")]}
    seen=set()
    for r in rows:
        key=(r["qid"],r["method"],r["condition"])
        if key in seen:raise ValueError("Duplicate episode")
        seen.add(key)
        s=samples[r["qid"]]
        if r["gt_windows"]!=s["gt_windows"] or r["source"]!=s["source"]:
            raise ValueError("Labels or source mismatch")
        audit_episode(r,s,protocol["config"])
        path=initial_branch(s["video_id"],s["query"])
        if r["condition"]=="mirrored_half":path=(1-path[0],)
        if r["method"]=="fixed_uniform":path=()
        rescored=score(r,s,path)
        for k in ["top1_iou","r1_at_05","r1_at_07","recovery_eligible","initial_window"]:
            if r[k]!=rescored[k]:raise ValueError("Saved score mismatch")
    if seen!=required:raise ValueError("Incomplete fixed denominator")
    if metrics(rows,protocol["config"])!=json.loads((out/"metrics.json").read_text("utf8")):
        raise ValueError("Aggregate replay mismatch")
    result={"all_episode_audits_pass":True,"exact_metrics_match":True,"records":len(rows),
            "records_sha256":sha(out/"records.jsonl"),"metrics_sha256":sha(out/"metrics.json"),
            "new_model_calls":0}
    write(out/"replay.json",result);return result


def evaluate(package,out,stage,adapter=None):
    package=Path(package).resolve();out=within_outputs(out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((package/"config.json").read_text("utf8"))
    manifest=json.loads((package/"manifest.json").read_text("utf8"))
    samples=[s for s in manifest["samples"] if s["stage"]==stage]
    if not samples:raise ValueError("Empty evaluation")
    for name,digest in manifest["source_hashes"].items():
        if sha(ROOT/name)!=digest:raise ValueError("Frozen source mismatch")
    for s in samples:
        for f in s["frames"]:
            image=(package/f["image_path"]).resolve();image.relative_to(package)
            if sha(image)!=f["sha256"]:raise ValueError("Image SHA mismatch")
    adapter=Path(adapter).resolve() if adapter else None
    write(out/"protocol.json",{"config":cfg,"stage":stage,"prompt":SFT_PROMPT,
        "manifest_sha256":sha(package/"manifest.json"),"adapter":str(adapter) if adapter else None,
        "adapter_sha256":sha(adapter/"adapter_model.safetensors") if adapter else None,
        "source_sha256":{p:sha(ROOT/p) for p in ["scripts/eval_agent_sft.py","scripts/train_agent_sft.py",
                                               "queryseek/agent/sft_policy.py"]}})
    write(out/"status.json",{"phase":"loading_model"})
    before=time.perf_counter();backend=QuantizedBackend(cfg,adapter)
    started=time.perf_counter()
    write(out/"environment.json",{"python":sys.version,"load_sec":started-before,
                                 "quantization":cfg["quantization"]})
    rows=[]
    for n,s in enumerate(samples):
        ff=[VideoFrame(frame_id=f["frame_id"],timestamp_sec=f["timestamp_sec"],
                       image_path=str(package/f["image_path"])) for f in s["frames"]]
        path=initial_branch(s["video_id"],s["query"])
        jobs=[("fixed_uniform","global",()),("forward_only","hash_half",path),
              ("backtracking","hash_half",path),("forward_only","mirrored_half",(1-path[0],)),
              ("backtracking","mirrored_half",(1-path[0],))]
        jobs=jobs[n%5:]+jobs[:n%5]
        for method,condition,initial in jobs:
            before=len(backend.calls)
            policy=SearchPolicy(backend,ff,s["duration_sec"])
            if time.perf_counter()-started>14400:
                def policy(*args):raise RuntimeError("Stage deadline")
            row=rollout(s["query"],s["duration_sec"],ff,method,initial,cfg,policy)
            row.update(condition=condition,generation_stats=backend.calls[before:])
            row=score(row,s,initial);rows.append(row)
            with (out/"records.jsonl").open("a",encoding="utf8") as handle:
                handle.write(json.dumps(row,ensure_ascii=False)+"\n")
            write(out/"status.json",{"phase":"running","completed":len(rows),"expected":len(samples)*5,
                "generations":len(backend.calls),"elapsed_sec":time.perf_counter()-started})
            print(json.dumps({"completed":len(rows),"status":row["status"]}),flush=True)
    write(out/"metrics.json",metrics(rows,cfg))
    write(out/"status.json",{"phase":"complete","completed":len(rows),"expected":len(samples)*5,
        "generations":len(backend.calls),"elapsed_sec":time.perf_counter()-started})
    return replay(package,out)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--package",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--adapter",type=Path)
    ap.add_argument("--stage",choices=["dev","confirmation"],default="dev");ap.add_argument("--replay",action="store_true")
    a=ap.parse_args();print(json.dumps(replay(a.package,a.out) if a.replay else evaluate(
        a.package,a.out,a.stage,a.adapter)))
