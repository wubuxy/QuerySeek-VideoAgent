"""Bounded download and complete identity-only audit of exact interval metadata."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.localizer_study import read, rows, verify
from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.tools.export import new_job_id


def identity(value):
    """Strict YouTube source namespace, without guessing from other identifiers."""
    if not isinstance(value,str): raise ValueError('Non-string pretraining identity')
    match = re.fullmatch(r'([A-Za-z0-9_-]{11})(?:_[0-9.+-]+_[0-9.+-]+)?',value)
    if not match: raise ValueError('Unrecognized pretraining source identity')
    return match[1]


def audit_rows(path, groups):
    sources = set(); count = 0; invalid_count = 0; invalid = []; overlaps = {name:set() for name in groups}
    with path.open(encoding='utf-8') as file:
        for line_no,line in enumerate(file,1):
            if not line.strip(): continue
            row = json.loads(line); count += 1
            try: source = identity(row['vid'])
            except (ValueError,KeyError) as exc:
                invalid_count += 1
                if len(invalid)<20: invalid.append({'line':line_no,'error_type':type(exc).__name__})
                # Count separately; no captions/query/GT are retained.
                continue
            sources.add(source)
            for name,ids in groups.items():
                if source in ids: overlaps[name].add(source)
    complete = count>0 and invalid_count==0
    return {'rows':count,'sources':len(sources),'invalid_rows':invalid_count,'invalid_examples':invalid,
            'complete_identity_parse':complete,'overlaps':{name:{'sources':len(ids),'source_ids':sorted(ids)} for name,ids in overlaps.items()},
            'interval_subset_negative_claim_allowed':complete}, sources


def bounded_download(url, target, max_bytes, timeout, deadline):
    """No credentials/retries; caps actual received bytes even without length headers."""
    digest = hashlib.sha256(); count = 0
    with urllib.request.urlopen(url,timeout=timeout) as response, target.open('xb') as file:
        kind = response.headers.get('Content-Type','')
        if 'html' in kind.lower(): raise ValueError('HTML response is not official JSONL metadata')
        announced = response.headers.get('Content-Length')
        if announced and int(announced)>max_bytes: raise ValueError('Metadata exceeds byte cap')
        while True:
            if time.perf_counter()>deadline: raise TimeoutError('Metadata wall-time cap')
            block = response.read(min(1024*1024,max_bytes-count+1))
            if not block: break
            count += len(block)
            if count>max_bytes: raise ValueError('Metadata exceeds byte cap')
            file.write(block); digest.update(block)
        if not count: raise ValueError('Empty metadata response')
    return {'bytes':count,'sha256':digest.hexdigest(),'content_type':kind,'http_status':200}


def run(config,out):
    started = time.perf_counter(); study = checked_output(ROOT/config['study_run']); parent = verify(study)
    data = checked_output(ROOT/config['data_run'])
    for name,sha in parent['data_sha256'].items():
        if sha256_file(data/name)!=sha: raise ValueError('Original data manifest changed')
    out = checked_output(out); out.mkdir(parents=True,exist_ok=False)
    inputs = {str(data/name):sha256_file(data/name) for name in ['train.jsonl','internal_validation.jsonl','excluded_sources.json']}
    manifest = study/'feature_records.jsonl'; inputs[str(manifest)] = sha256_file(manifest)
    metadata = rows(manifest)
    groups = {'study_train':{source_video_id(r['entry']['vid']) for r in metadata if r['split']=='train'},
              'study_internal_validation':{source_video_id(r['entry']['vid']) for r in metadata if r['split']=='validation'},
              'official_train_partition_union':{source_video_id(r['vid']) for name in ['train.jsonl','internal_validation.jsonl'] for r in rows(data/name)},
              'excluded_identity_union':set(read(data/'excluded_sources.json'))}
    shutil.copyfile(__file__,out/'source_snapshot.py')
    write_json_atomic(out/'protocol.json',{'config':config,'input_sha256':inputs,'script_sha256':sha256_file(__file__),
        'group_sizes':{name:len(ids) for name,ids in groups.items()},'fresh_query_label_access':False,
        'exact_interval_metadata_only':True,'other_pretraining_metadata_audited':False})
    result = {'phase':'started','paid_api_calls':0,'weights_downloaded':False,'fresh_inference':False,
              'all_pretraining_sources_disjoint':None,'independent_generalization_verified':False}
    write_json_atomic(out/'status.json',result)
    try:
        target = out/'interval_900k.jsonl'
        transport = bounded_download(config['official_interval_url'],target,config['max_download_bytes'],
                config['timeout_sec'],started+config['wall_cap_sec'])
        result['transport'] = transport
        result['interval_identity_audit'],sources = audit_rows(target,groups)
        (out/'interval_source_ids.txt').write_text(''.join(s+'\n' for s in sorted(sources)),encoding='utf-8')
        result['phase'] = 'complete_interval_only' if result['interval_identity_audit']['complete_identity_parse'] else 'invalid_identities'
        # Complete interval absence never proves absence in the full PT corpus.
        result['all_pretraining_sources_disjoint'] = None
    except Exception as exc:
        result.update(phase='failed',error_type=type(exc).__name__)
        if isinstance(exc,(ValueError,TimeoutError)): result['reason'] = str(exc)
    for path,sha in inputs.items():
        if sha256_file(path)!=sha: raise ValueError('Frozen identity inputs changed')
    verify(study)
    result['elapsed_sec'] = time.perf_counter()-started
    write_json_atomic(out/'metrics.json',result); write_json_atomic(out/'status.json',{'phase':result['phase']})
    print(json.dumps(result,indent=2),flush=True); print('OUTPUT='+str(out),flush=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/univtg_source_audit_v1.json')
    args = parser.parse_args()
    run(read(ROOT/args.config),ROOT/'outputs/univtg_source_audit'/new_job_id())
