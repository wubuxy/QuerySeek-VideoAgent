"""Freeze source-isolated images and executable, annotation-derived SFT states."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from queryseek.agent.sft_policy import AnnotationTeacher, SearchPolicy
from queryseek.agent.search_policy import initial_branch, node_window, rollout
from queryseek.ingest.frames import load_frames_index
from scripts.gpu_agent_experiment import sha,within_outputs,write


def source(vid):
    return vid.rsplit("_",2)[0]


def select(rows,cfg,excluded):
    groups={}
    for r in rows:
        sid=source(r["entry"]["vid"])
        if sid in excluded:
            continue
        key=(r["split"],sid)
        if key not in groups or r["entry"]["qid"]<groups[key]["entry"]["qid"]:
            groups[key]=r
    order=lambda sid:hashlib.sha256((cfg["selection_seed"]+":"+sid).encode()).hexdigest()
    train=sorted([s for split,s in groups if split=="train"],key=order)
    val=sorted([s for split,s in groups if split=="validation"],key=order)
    if len(train)<cfg["train_sources"] or len(val)<cfg["dev_sources"]+cfg["confirmation_sources"]:
        raise ValueError("Insufficient disjoint source inventory")
    result={}
    result["train"]=[groups[("train",s)] for s in train[:cfg["train_sources"]]]
    result["dev"]=[groups[("validation",s)] for s in val[:cfg["dev_sources"]]]
    result["confirmation"]=[groups[("validation",s)] for s in val[cfg["dev_sources"]:cfg["dev_sources"]+cfg["confirmation_sources"]]]
    sets={k:{source(r["entry"]["vid"]) for r in v} for k,v in result.items()}
    if any(sets[a]&sets[b] for a in sets for b in sets if a<b):
        raise ValueError("Source leakage across splits")
    return result


def make_trajectories(sample,frames,cfg):
    base=initial_branch(sample["video_id"],sample["query"])
    quarters=[(0,0),(0,1),(1,0),(1,1)]
    def overlap(path):
        a,b=node_window(path,sample["duration_sec"])
        return sum(max(0,min(b,y)-max(a,x)) for x,y in sample["gt_windows"])
    wrong=min(quarters,key=lambda p:(overlap(p),p))
    jobs=[("fixed_uniform",(), "global"),("backtracking",base,"hash_half"),
          ("backtracking",(1-base[0],),"mirrored_half"),("backtracking",wrong,"annotation_detour")]
    examples,episodes=[],[]
    for method,path,condition in jobs:
        teacher=AnnotationTeacher(frames,sample["duration_sec"],sample["gt_windows"],cfg,method)
        row=rollout(sample["query"],sample["duration_sec"],frames,method,path,cfg,
                    SearchPolicy(teacher,frames,sample["duration_sec"]))
        if row["status"]=="error":
            raise ValueError("Invalid teacher rollout")
        episode_id=sample["qid"]+"/"+condition
        episodes.append({**row,"qid":sample["qid"],"source":sample["source"],
                         "episode_id":episode_id,"condition":condition})
        for i,ex in enumerate(teacher.examples):
            examples.append({**ex,"qid":sample["qid"],"source":sample["source"],
                             "episode_id":episode_id,"example_id":episode_id+f"/{i}"})
    return examples,episodes


def prepare(out):
    out=within_outputs(out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/"configs/agent_sft_v1.json").read_text("utf8"))
    records=ROOT/cfg["data_records"]
    rows=[json.loads(x) for x in records.read_text("utf8").splitlines()]
    old=json.loads((ROOT/"outputs/gpu_agent_20261021/input_cached/manifest.json").read_text("utf8"))
    selected=select(rows,cfg,{s["source"] for s in old["samples"]})
    selection=[{"stage":stage,"qid":f'{stage}_{r["entry"]["qid"]}',"video_id":r["entry"]["vid"],
                "source":source(r["entry"]["vid"])} for stage,rr in selected.items() for r in rr]
    write(out/"selection.json",selection)
    write(out/"config.json",cfg)
    fresh=ROOT/"data/qvhighlights_fresh80/selection.json"
    manifest={"samples":[],"selection_sha256":sha(out/"selection.json"),
              "records_sha256":sha(records),"fresh_selection_sha256":sha(fresh),
              "source_hashes":{n:sha(ROOT/n) for n in [
                  "queryseek/agent/sft_policy.py","queryseek/agent/anchored_policy.py",
                  "queryseek/agent/search_policy.py","queryseek/ingest/frames.py"]}}
    began=time.perf_counter();examples=[];episodes=[]
    for stage,rr in selected.items():
        for r in rr:
            entry=r["entry"];cache=Path(r["images_path"]).parent
            meta=json.loads((cache/"feature_meta.json").read_text("utf8"))["signature"]
            if (meta["pipeline_version"]!="2" or meta["extractor_sha256"]!=sha(ROOT/"queryseek/ingest/frames.py")
                    or meta["fps"]!=1. or meta["video_sha256"]!=r["video_sha256"]
                    or sha(r["video"])!=r["video_sha256"]):
                raise ValueError("Source image cache or video mismatch")
            ff=load_frames_index(cache/"frames.jsonl")[::cfg["pool_stride"]]
            duration=r["duration_sec"];gt=entry["relevant_windows"]
            if any(not 0<=a<b<=duration for a,b in gt):
                raise ValueError("Invalid frozen labels")
            portable=[]
            for f in ff:
                if not 0<=f.timestamp_sec<duration:
                    raise ValueError("Invalid source PTS")
                image=Path(f.image_path)
                target=out/"frames"/r["video_sha256"]/(f.frame_id+".jpg")
                target.parent.mkdir(parents=True,exist_ok=True)
                try:os.link(image,target)
                except OSError:shutil.copyfile(image,target)
                portable.append({"frame_id":f.frame_id,"timestamp_sec":f.timestamp_sec,
                                 "image_path":target.relative_to(out).as_posix(),"sha256":sha(target)})
            sample={"stage":stage,"qid":f'{stage}_{entry["qid"]}',"source":source(entry["vid"]),
                    "video_id":entry["vid"],"query":entry["query"],"duration_sec":duration,
                    "gt_windows":gt,"frames":portable,"video_sha256":r["video_sha256"],
                    "cache_signature":meta}
            manifest["samples"].append(sample)
            if stage=="train":
                new,rolls=make_trajectories(sample,ff,cfg)
                examples.extend(new);episodes.extend(rolls)
                with (out/"train.jsonl").open("a",encoding="utf8") as handle:
                    for x in new:handle.write(json.dumps(x,ensure_ascii=False)+"\n")
                with (out/"teacher_episodes.jsonl").open("a",encoding="utf8") as handle:
                    for x in rolls:handle.write(json.dumps(x,ensure_ascii=False)+"\n")
            write(out/"manifest.json",manifest)
            write(out/"status.json",{"phase":"preparing","samples":len(manifest["samples"]),
                                    "training_states":len(examples),"elapsed_sec":time.perf_counter()-began})
            print(json.dumps({"prepared":len(manifest["samples"]),"stage":stage,
                              "training_states":len(examples)}),flush=True)
    write(out/"data_audit.json",{"source_counts":{k:len(v) for k,v in selected.items()},
        "source_disjoint":True,"old_agent_sources_disjoint":True,"fresh_selection_unchanged":sha(fresh)==manifest["fresh_selection_sha256"],
        "training_states":len(examples),"training_episodes":len(episodes),
        "target_actions":dict(Counter(x["target"]["choice"].split(":")[0] for x in examples)),
        "teacher_outcomes":dict(Counter(x["status"] for x in episodes)),
        "image_entries":sum(len(s["frames"]) for s in manifest["samples"]),
        "sampling":"every second existing 1fps source-PTS frame; no new CLIP encoding",
        "weak_supervision":"annotation geometry, not expert semantic trajectories",
        "elapsed_sec":time.perf_counter()-began})
    write(out/"status.json",{"phase":"complete","samples":len(manifest["samples"]),
                            "training_states":len(examples),"elapsed_sec":time.perf_counter()-began})
    archive=out.parent/(out.name+".tar.gz")
    with tarfile.open(archive,"w:gz") as tar:
        for p in out.rglob("*"):
            if p.is_file():tar.add(p,arcname="outputs/sft_agent_input/"+p.relative_to(out).as_posix())
    write(out/"archive.json",{"sha256":sha(archive),"bytes":archive.stat().st_size})
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    print(prepare(ap.parse_args().out))
