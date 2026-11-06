"""Source-clustered paired base/adapter contrasts; no model calls."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.gpu_agent_experiment import sha,write,within_outputs


def compare_rows(base,trained,cfg):
    keys=lambda rows:{(r["qid"],r["method"],r["condition"]):r for r in rows}
    left,right=keys(base),keys(trained)
    if len(left)!=len(base) or len(right)!=len(trained) or left.keys()!=right.keys():
        raise ValueError("Duplicate or unmatched evaluation episodes")
    metrics=["top1_iou","r1_at_05","r1_at_07","unique_frames","logical_image_presentations"]
    diffs={};costs={};diagnosis={}
    for label,rows in [("base",base),("sft",trained)]:
        costs[label]={}
        for method in sorted({r["method"] for r in rows}):
            subset=[r for r in rows if r["method"]==method]
            calls=[c for r in subset for c in r["generation_stats"]]
            recover=[r for r in subset if r["recovery_eligible"]]
            costs[label][method]={"episodes":len(subset),
                **{k:sum(r[k] for r in subset)/len(subset) for k in metrics},
                "statuses":dict(Counter(r["status"] for r in subset)),
                "actions":dict(Counter(t.get("decision",{}).get("action","error") for r in subset for t in r["trace"])),
                "generations":len(calls),
                "input_tokens":sum(c["input_tokens"] for c in calls),
                "output_tokens":sum(c["output_tokens"] for c in calls),
                "recovery_eligible":len(recover),"recovery_successes":sum(r["r1_at_05"] for r in recover)}
    paired=[]
    for key,b in sorted(left.items()):
        s=right[key]
        if any(b[k]!=s[k] for k in ["source","gt_windows","initial_window","recovery_eligible"]):
            raise ValueError("Evidence or denominator mismatch")
        delta=[s[k]-b[k] for k in metrics]
        diffs.setdefault(b["method"],{}).setdefault(b["source"],[]).append(delta)
        if b["method"]=="backtracking":
            paired.append({"qid":b["qid"],"condition":b["condition"],"source":b["source"],
                "base_r1":b["r1_at_05"],"sft_r1":s["r1_at_05"],"delta_iou":s["top1_iou"]-b["top1_iou"],
                "recovery_eligible":b["recovery_eligible"]})
    contrasts={}
    for method,source_rows in sorted(diffs.items()):
        data=np.array([np.mean(source_rows[s],axis=0) for s in sorted(source_rows)])
        rng=np.random.default_rng(cfg["bootstrap_seed"])
        boot=data[rng.integers(0,len(data),(cfg["bootstrap_replicates"],len(data)))].mean(axis=1)
        contrasts[method]={"sources":len(data),**{k:{"mean":float(data[:,i].mean()),
            "ci95":np.quantile(boot[:,i],[.025,.975]).tolist()} for i,k in enumerate(metrics)}}
    return {"groups":costs,"sft_minus_base_source_average":contrasts,
            "primary_contrast":"backtracking/r1_at_05","paired_backtracking":paired,
            "bootstrap_seed":cfg["bootstrap_seed"],"replicates":cfg["bootstrap_replicates"]}


def compare(base,trained,out):
    base,trained=Path(base),Path(trained);out=within_outputs(out)
    protocols=[json.loads((p/"protocol.json").read_text("utf8")) for p in [base,trained]]
    for k in ["config","stage","prompt","manifest_sha256"]:
        if protocols[0][k]!=protocols[1][k]:raise ValueError("Protocol mismatch")
    if protocols[0]["adapter"] is not None or not protocols[1]["adapter"]:
        raise ValueError("Expected base then adapter")
    rows=[[json.loads(x) for x in (p/"records.jsonl").read_text("utf8").splitlines()] for p in [base,trained]]
    result=compare_rows(*rows,protocols[0]["config"])
    result["input_sha256"]={str(p):sha(p/"records.jsonl") for p in [base,trained]}
    write(out,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base",required=True,type=Path)
    p.add_argument("--trained",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();compare(a.base,a.trained,a.out)
