"""Freeze source-separated samples and portable, SHA-checked source-PTS images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.agent_policy_pilot import load_pool
from scripts.gpu_agent_experiment import sha, within_outputs, write
from queryseek.eval.runner import load_dataset, dataset_from_labels

SOURCES = ["scripts/gpu_agent_experiment.py", "queryseek/agent/anchored_policy.py",
           "queryseek/agent/search_policy.py", "queryseek/eval/qvhighlights.py"]


def source_id(video_id):
    return video_id.rsplit("_", 2)[0]


def select_confirmation(pilot, candidates, count):
    excluded = {source_id(s.video_id) for s in pilot}
    first = {}
    for s in candidates:
        source = source_id(s.video_id)
        if source not in excluded:
            first.setdefault(source, s)
    ordered = sorted(first, key=lambda s: hashlib.sha256(("gpu-agent-confirm-20261021:"+s).encode()).hexdigest())
    if len(ordered) < count:
        raise ValueError("Not enough disjoint sources")
    return [first[s] for s in ordered[:count]]


def prepare(out, mock=False, fixture=None):
    out = within_outputs(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg_path = ROOT/"configs/gpu_agent_v2.json"
    cfg = json.loads(cfg_path.read_text("utf8"))
    if mock:
        fixture = fixture or out/"fixture"
        if not (fixture/"sample.mp4").exists():
            subprocess.run([sys.executable, str(ROOT/"scripts/make_synthetic_fixture.py"),
                            "--out-dir", str(fixture)], shell=False, timeout=600, check=True, capture_output=True)
        pilot = dataset_from_labels(fixture/"sample.labels.json", fixture/"sample.mp4", None)
        confirmation = []
    else:
        pilot = load_dataset(ROOT/"data/qvhighlights/dataset.json")
        if len(pilot) != cfg["pilot_count"]:
            raise ValueError("Pilot denominator changed")
        confirmation = select_confirmation(pilot, load_dataset(ROOT/"data/qvhighlights_test200/dataset.json"),
                                           cfg["confirmation_count"])
    selection = [{"stage": stage, "qid": f"{stage}_{s.qid}", "source": source_id(s.video_id),
                  "video_id": s.video_id, "query": s.query}
                 for stage, samples in [("pilot", pilot), ("confirmation", confirmation)] for s in samples]
    write(out/"selection.json", selection)
    fresh = ROOT/"data/qvhighlights_fresh80/selection.json"
    manifest = {"mock": mock, "selection_sha256": sha(out/"selection.json"),
                "source_hashes": {p: sha(ROOT/p) for p in SOURCES},
                "fresh_selection_sha256": sha(fresh) if fresh.exists() else None, "samples": []}
    shutil.copyfile(cfg_path, out/"config.json")
    started = time.perf_counter()
    for stage, samples in [("pilot", pilot), ("confirmation", confirmation)]:
        for s in samples:
            asset, frames, audit = load_pool(s, out, mock)
            image_dir = out/"frames"/asset.sha256
            image_dir.mkdir(parents=True, exist_ok=True)
            portable = []
            for f in frames:
                target = image_dir/(f.frame_id+".jpg")
                if not target.exists():
                    shutil.copyfile(f.image_path, target)
                portable.append({"frame_id": f.frame_id, "timestamp_sec": f.timestamp_sec,
                                 "image_path": target.relative_to(out).as_posix(), "sha256": sha(target)})
            manifest["samples"].append({"stage": stage, "qid": f"{stage}_{s.qid}", "source": source_id(s.video_id),
                "video_id": s.video_id, "query": s.query, "duration_sec": asset.duration_sec,
                "gt_windows": [[w.start_sec, w.end_sec] for w in s.gt_windows], "frames": portable,
                "input_audit": audit})
            write(out/"manifest.json", manifest)
            print(json.dumps({"prepared": len(manifest["samples"]), "qid": f"{stage}_{s.qid}",
                              "frames": len(frames), "cache_hit": audit["frame_cache_hit"]}), flush=True)
    write(out/"preparation.json", {"phase": "complete", "elapsed_sec": time.perf_counter()-started,
          "samples": len(manifest["samples"]), "new_image_encodings": 0,
          "source_disjoint": not ({source_id(s.video_id) for s in pilot} & {source_id(s.video_id) for s in confirmation}),
          "fresh_selection_unchanged": not fresh.exists() or sha(fresh) == manifest["fresh_selection_sha256"]})
    archive = out.parent/(out.name+".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for folder in ["queryseek"]:
            for p in sorted((ROOT/folder).rglob("*.py")):
                tar.add(p, arcname=p.relative_to(ROOT).as_posix())
        for name in ["scripts/gpu_agent_experiment.py", "configs/gpu_agent_v2.json", "docs/EVAL_GPU_AGENT_V2.md"]:
            tar.add(ROOT/name, arcname=name)
        for p in sorted(out.rglob("*")):
            if p.is_file():
                tar.add(p, arcname="outputs/input/"+p.relative_to(out).as_posix())
    write(out/"archive.json", {"path": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    prepare(args.out, args.mock, args.fixture)
