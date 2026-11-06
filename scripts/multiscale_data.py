"""Extend the existing frozen source prefix; reuse acquisition and CLIP pipeline."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import localizer_study as study
from scripts.sampling_probe import checked_output
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.tools.export import new_job_id


def freeze(cfg,out):
    old=ROOT/cfg['study_run'];parent=study.verify(old)
    c={**parent['config'],'version':cfg['version'],'train_scales':[256,cfg['planned_train_sources']],
       'scope':cfg['scope'],'variants':[]}
    study.freeze(c,out)
    # New-run bookkeeping only: the historical helper's pilot cap was 320.
    p=study.read(out/'protocol.json');p['max_crops']=cfg['planned_train_sources']+c['validation_sources']
    p['max_video_bytes_total']=p['max_crops']*c['max_video_bytes']
    for name in ('train_256.jsonl','validation.jsonl'):
        if sha256_file(old/name)!=sha256_file(out/name):raise ValueError('Original prefix/validation changed')
    p['extension']={'config':cfg,'parent_run':cfg['study_run'],'parent_feature_records_sha256':sha256_file(old/'feature_records.jsonl'),
                    'unchanged_prefix_and_validation':True,'fresh_query_label_access':False}
    for name in ('scripts/multiscale_data.py','configs/multiscale_ranker_v1.json'):
        dst=out/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/name).read_bytes())
        p['source_sha256'][name]=sha256_file(ROOT/name)
    write_json_atomic(out/'protocol.json',p);print('OUTPUT='+str(out),flush=True);return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['freeze','prepare','encode']);ap.add_argument('--run',type=Path);a=ap.parse_args()
    if a.phase=='freeze':freeze(study.read(ROOT/'configs/multiscale_ranker_v1.json'),ROOT/'outputs/multiscale_data'/new_job_id())
    else:
        if a.run is None:ap.error('--run required')
        {'prepare':study.prepare,'encode':study.encode}[a.phase](checked_output(a.run))
