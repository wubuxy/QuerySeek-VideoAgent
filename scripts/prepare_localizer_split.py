"""Prepare official-train source-disjoint localization annotations, no training.

Only video identity fields of val/test/fresh files are used for exclusion.
Annotation artifacts remain private under outputs/, with the original license.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import GroupShuffleSplit

from scripts.make_qvhighlights_subset import select_entries, source_video_id
from scripts.sampling_probe import checked_output, optional_version
from queryseek import PIPELINE_VERSION
from queryseek.ft.sampling import VID_WHITELIST
from queryseek.ingest.ffprobe import sha256_file
from queryseek.schemas import TimeWindow
from queryseek.tools.export import new_job_id


def valid_localization_rows(entries):
    valid, invalid = [], []
    seen = set()
    for entry in entries:
        qid = entry.get('qid')
        try:
            if qid in seen:
                raise ValueError('Duplicate official query ID')
            seen.add(qid)
            vid = entry['vid']
            if not VID_WHITELIST.fullmatch(vid) or '..' in vid:
                raise ValueError('Unsafe clip identity')
            duration = float(entry['duration'])
            if not math.isfinite(duration) or duration <= 0 or not entry['query'].strip():
                raise ValueError('Invalid duration/query')
            if not entry['relevant_windows']:
                raise ValueError('No annotated positive localization window')
            for start,end in entry['relevant_windows']:
                w = TimeWindow(start_sec=start,end_sec=end)
                if w.end_sec > duration:
                    raise ValueError('Window exceeds official metadata duration')
            valid.append(entry)
        except (KeyError,ValueError,TypeError) as exc:
            invalid.append({'qid':qid,'reason':type(exc).__name__})
    return valid, invalid


def source_disjoint_split(entries, excluded_sources, seed=42, validation_fraction=.2):
    eligible = sorted([e for e in entries if source_video_id(e['vid']) not in excluded_sources],key=lambda e:e['qid'])
    groups = [source_video_id(e['vid']) for e in eligible]
    if len(set(groups)) < 2 or not 0 < validation_fraction < 1:
        raise ValueError('At least two source groups and a valid validation fraction required')
    train_indices,val_indices = next(GroupShuffleSplit(n_splits=1,test_size=validation_fraction,
                                                       random_state=seed).split(eligible,groups=groups))
    train = [eligible[i] for i in train_indices]
    validation = [eligible[i] for i in val_indices]
    train_sources = {source_video_id(e['vid']) for e in train}
    validation_sources = {source_video_id(e['vid']) for e in validation}
    if train_sources & validation_sources or (train_sources|validation_sources) & set(excluded_sources):
        raise RuntimeError('Source-video isolation failed')
    return train,validation


def identity_sources(path, kind):
    if kind == 'official':
        # Deliberately never inspect query or ground-truth values in evaluation files.
        return {source_video_id(json.loads(line)['vid']) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}
    raw = json.loads(path.read_text(encoding='utf-8'))
    if kind == 'fresh':
        return {source_video_id(e['vid']) for e in raw['entries']}
    return {source_video_id(e.get('video_id') or Path(e['video']).stem) for e in raw['samples']}


def prepare(manifest, official_root, out):
    provenance = subprocess.run(['git','-C',str(official_root),'rev-parse','HEAD'],shell=False,
                                capture_output=True,text=True,timeout=30)
    if provenance.returncode or provenance.stdout.strip() != manifest['official_commit']:
        raise ValueError('Official annotation clone must match frozen commit')
    train_path = official_root/manifest['official_train']
    dirty = subprocess.run(['git','-C',str(official_root),'diff','--quiet','HEAD','--',
                            manifest['official_train'],'data/LICENSE'],shell=False,timeout=30)
    if dirty.returncode != 0:
        raise ValueError('Official annotation/license files modified')
    exclusions = set()
    exclusion_counts, hashes = {}, {}
    paths = [(official_root/name,'official') for name in manifest['official_identity_exclusions']]
    paths += [(ROOT/name,'development') for name in manifest['development_datasets']]
    paths += [(ROOT/manifest['fresh_selection'],'fresh')]
    for path,kind in paths:
        sources = identity_sources(path,kind)
        exclusions.update(sources)
        key = f'{kind}/{path.parent.name}/{path.name}'
        exclusion_counts[key] = len(sources)
        hashes[key] = sha256_file(path)
    entries = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    valid,invalid = valid_localization_rows(entries)
    train,validation = source_disjoint_split(valid,exclusions,manifest['group_split_seed'],
                                             manifest['internal_validation_source_fraction'])
    pilot_train = select_entries(train,manifest['pilot_train_sources'],manifest['pilot_selection_seed'],source_disjoint=True)
    pilot_val = select_entries(validation,manifest['pilot_validation_sources'],manifest['pilot_selection_seed'],source_disjoint=True)
    if len(pilot_train) != manifest['pilot_train_sources'] or len(pilot_val) != manifest['pilot_validation_sources']:
        raise ValueError('Not enough eligible sources for frozen pilot')
    out = checked_output(out)
    out.mkdir(parents=True,exist_ok=False)
    protocol = {'manifest':manifest,'pipeline_version':PIPELINE_VERSION,'official_commit':manifest['official_commit'],
                'official_train_sha256':sha256_file(train_path),'identity_input_hashes':hashes,
                'splitter_version':optional_version('scikit-learn'),'fresh_identity_only':True,
                'training_started':False,'video_downloads':0,'feature_encodings':0}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for name,rows in (('train',train),('internal_validation',validation),('pilot_train',pilot_train),('pilot_validation',pilot_val)):
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows),encoding='utf-8')
    (out/'DATA_LICENSE.txt').write_bytes((official_root/'data/LICENSE').read_bytes())
    (out/'prepare_localizer_split.py').write_bytes(Path(__file__).read_bytes())
    (out/'excluded_sources.json').write_text(json.dumps(sorted(exclusions)),encoding='utf-8')
    report = {'official_rows':len(entries),'valid_rows':len(valid),'invalid_rows':invalid,
              'excluded_rows':len(valid)-len(train)-len(validation),'excluded_source_count':len(exclusions),
              'identity_source_counts':exclusion_counts,'train_queries':len(train),'validation_queries':len(validation),
              'train_sources':len({source_video_id(e['vid']) for e in train}),
              'validation_sources':len({source_video_id(e['vid']) for e in validation}),
              'pilot_train_sources':len(pilot_train),'pilot_validation_sources':len(pilot_val),
              'train_validation_source_overlap':0,'evaluation_source_overlap':0,'training_started':False,
              'actual_video_duration_checked':False,'video_downloads':0,'fresh_queries_or_labels_used':False}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Independent selection must remain untouched during identity exclusion.
    if sha256_file(ROOT/manifest['fresh_selection']) != hashes[f'fresh/qvhighlights_fresh80/selection.json']:
        raise RuntimeError('Independent selection changed')
    print('OUTPUT='+str(out),flush=True)
    print(json.dumps({k:v for k,v in report.items() if k != 'identity_source_counts'},indent=2),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root',type=Path,default=Path(os.environ.get('QS_MOMENT_DETR_ROOT',
                        str(Path(os.environ.get('TEMP','/tmp'))/'qs_paper_review/moment_detr'))))
    args = parser.parse_args()
    manifest = json.loads((ROOT/'configs/localizer_data_v1.json').read_text(encoding='utf-8'))
    prepare(manifest,args.official_root,ROOT/'outputs/localizer_data'/new_job_id())
