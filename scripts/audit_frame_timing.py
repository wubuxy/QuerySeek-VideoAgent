"""Independent image-content identification of source frames on a generated fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np

from queryseek.ingest.frames import _require_ffmpeg,extract_uniform_frames,extract_local_frames,extract_scene_frames
from queryseek.schemas import Scene
from queryseek.tools.export import new_job_id
from scripts.make_synthetic_fixture import build_timing_video
from scripts.sampling_probe import checked_output


def legacy_frames(video,out,fps,start=0.,end=8.):
    out.mkdir(parents=True,exist_ok=True)
    cmd=[_require_ffmpeg(),"-y","-i",str(video)]
    if start:cmd.extend(["-ss",str(start),"-to",str(end)])
    cmd.extend(["-vf",f"fps={fps},scale=64:64","-q:v","3",str(out/"frame_%06d.jpg")])
    proc=subprocess.run(cmd,shell=False,capture_output=True,timeout=120)
    if proc.returncode:raise RuntimeError(proc.stderr.decode(errors="replace")[-500:])
    from queryseek.schemas import VideoFrame
    return [VideoFrame(frame_id=str(i),timestamp_sec=start+i/fps,image_path=str(out/f"frame_{i+1:06d}.jpg"))
            for i in range(int((end-start)*fps)) if (out/f"frame_{i+1:06d}.jpg").is_file()]


def audit(out):
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    video=build_timing_video(out/"fixture")
    # At native 12fps the historical filter keeps every source frame. Frame identity
    # comes from independent luminance reference, not the tested metadata parser.
    references=legacy_frames(video,out/"reference",12)
    means=np.array([float(cv2.imread(f.image_path).mean()) for f in references])
    if len(means)!=96 or len(np.unique(means))!=96:raise AssertionError("Fixture identity is ambiguous")
    groups={"legacy_uniform":legacy_frames(video,out/"legacy_uniform",.25),
            "current_uniform":extract_uniform_frames(video,out/"current_uniform",.25,8,64),
            "legacy_local":legacy_frames(video,out/"legacy_local",1,2.3,6.3),
            "current_local":extract_local_frames(video,out/"current_local",2.3,6.3,1,64),
            "scene_seek":extract_scene_frames(video,out/"scene",[Scene(scene_id="s",start_sec=2.2,end_sec=2.4)],64)}
    report={"fixture":"8 seconds / 12fps, lossless frame-index luminance", "groups":{}}
    for name,frames in groups.items():
        rows=[]
        for f in frames:
            index=int(np.argmin(np.abs(means-float(cv2.imread(f.image_path).mean()))))
            source_sec=index/12
            rows.append({"reported_sec":f.timestamp_sec,"source_frame_index":index,"source_sec":source_sec,
                         "error_sec":f.timestamp_sec-source_sec})
        report["groups"][name]={"frames":rows,"max_abs_error_sec":max((abs(r["error_sec"]) for r in rows),default=None)}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out),**report},indent=2))
    return report


if __name__=="__main__":audit(ROOT/"outputs/timing_audit"/new_job_id())
