"""Frozen observation transfer study, with separately versioned official inputs.

No pretrained locator is loaded. GT enters only after every prediction is saved.
The optional checkpoint prefix audit uses pickletools disassembly, never pickle
execution or torch.load. Unknown model permission/epoch remains a closed gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from queryseek import PIPELINE_VERSION
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.univtg_inputs import text_hidden_states, valid_text_features, upstream_temporal_inputs
from queryseek.tools.export import new_job_id
from scripts.univtg_input_audit import verify_run, import_official, parse_showinfo, pixel_tensor
from scripts.localizer_study import read, rows, verify
from scripts.localizer_query_controls import frozen_inventory
from scripts.sampling_probe import checked_output, optional_version
from scripts.replay_localizer_curves import curve_predictions
from scripts.analyze_localizer_study import paired_clusters, per_query, candidate_diagnostic
from queryseek.eval.qvhighlights import eval_mr_brief


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prefix_epoch(blob):
    """Inspect a complete first ZIP data.pkl member, without executing pickle.

    Only a literal integer adjacent to the top-level-looking epoch key is
    reported as a candidate. This is metadata evidence, not weight authenticity.
    No inference is made from file names or the configured training budget.
    """
    if len(blob) < 30 or blob[:4] != b'PK\x03\x04':
        return {'epoch_candidate': None, 'reason': 'not_a_zip_prefix'}
    header = struct.unpack('<IHHHHHIIIHH', blob[:30])
    size, name_len, extra_len = header[7], header[9], header[10]
    name = blob[30:30+name_len].decode('utf-8', errors='replace')
    if not name.endswith('/data.pkl'):
        return {'epoch_candidate': None, 'reason': 'no_complete_sized_data_pickle', 'first_member': name}
    start = 30+name_len+extra_len
    unsized = size in (0, 0xffffffff)
    if not unsized and start+size > len(blob):
        return {'epoch_candidate': None, 'reason': 'truncated_data_pickle', 'first_member': name}
    data = blob[start:] if unsized else blob[start:start+size]
    if header[3] == 8:
        # Cap decompressed metadata as well as received bytes.
        dec = zlib.decompressobj(-15); data = dec.decompress(data, 4*1024*1024+1)
        if len(data)>4*1024*1024 or not dec.eof: raise ValueError('Oversized pickle metadata')
    elif header[3] != 0:
        return {'epoch_candidate': None, 'reason': 'unsupported_compression'}
    previous = None; found = []
    try:
        for op, arg, _ in pickletools.genops(data):
            if previous == 'epoch' and op.name in ('BININT', 'BININT1', 'BININT2', 'INT', 'LONG1', 'LONG4'):
                found.append(int(arg))
            if op.name not in ('MEMOIZE', 'BINPUT', 'LONG_BINPUT'):
                previous = arg if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE') else None
    except (ValueError, UnicodeError):
        return {'epoch_candidate': None, 'reason': 'incomplete_or_invalid_pickle', 'first_member': name}
    return {'epoch_candidate': found[0] if len(found)==1 else None,
            'epoch_candidates': found, 'first_member': name, 'reason': 'static_disassembly_only'}


def checkpoint_gate(cfg, out):
    """One bounded public prefix request; no model execution, no tensor load."""
    result = {'weights_license': 'unverified', 'locator_loading_allowed': False,
              'epoch': None, 'tensor_deserialization': False, 'full_checkpoint_downloaded': False,
              'code_license_does_not_establish_weight_permission': True}
    cap = cfg['checkpoint_prefix_cap_bytes']; start = time.perf_counter()
    request = urllib.request.Request(cfg['checkpoint_metadata_url'], headers={'Range': f'bytes=0-{cap-1}'})
    try:
        with urllib.request.urlopen(request, timeout=cfg['checkpoint_request_timeout_sec']) as response:
            data = response.read(cap)
            result.update(http_status=response.status, received_bytes=len(data),
                          content_type=response.headers.get('Content-Type'),
                          content_range=response.headers.get('Content-Range'))
        (out/'checkpoint_prefix.bin').write_bytes(data)
        result['prefix_sha256'] = hashlib.sha256(data).hexdigest()
        result['metadata'] = prefix_epoch(data)
        result['epoch'] = result['metadata']['epoch_candidate']
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    result['elapsed_sec'] = time.perf_counter()-start
    result['decision'] = 'locator gate closed; continue only independent CLIP input portability study'
    write_json_atomic(out/'checkpoint_gate.json', result)
    return result


def signature(record, cfg, weights_sha, source_sha, versions):
    return {'contract': cfg['version'], 'pipeline_version': PIPELINE_VERSION,
            'video_sha256': record['video_sha256'], 'duration_sec': record['duration_sec'],
            'width': record['width'], 'height': record['height'],
            'sampling': {'filter': 'fps=1/2', 'phase': 'ffmpeg_default_round_near', 'clip_length_sec': cfg['clip_length_sec']},
            'pixels': 'FFmpeg scale-short224 integer dimensions, center crop224, RGB24, no JPEG',
            'normalization': 'official mean/std with 1e-8 epsilon; image/token row-L2',
            'clip_weights_sha256': weights_sha, 'model': 'ViT-B-32/openai/QuickGELU/FP32',
            'token_cap': cfg['token_cap'], 'length_policy': cfg['length_policy'],
            'temporal': 'TEF[i/L,(i+1)/L], anchor(i+clip_length/2)/L; separate logical grid/source PTS candidates',
            'locator_weights_sha256': None, 'locator_status': 'not_loaded_unverified_permission_epoch',
            'source_sha256': source_sha, 'versions': versions}


def decode_pixels(record, out, max_frames):
    """Whole video, exact published fps/scale/crop; retain source ambiguity."""
    out=checked_output(out)
    if max_frames<=0: raise ValueError('Frame budget exceeded')
    width, height = record['width'], record['height']
    rh, rw = (int(height*224/width),224) if height>=width else (224,int(width*224/height))
    vf = f'showinfo@src,fps=1/2,showinfo@sample,scale={rw}:{rh},crop=224:224:{int((rw-224)/2)}:{int((rh-224)/2)}'
    cmd = ['ffmpeg','-hide_banner','-loglevel','info','-threads','1','-i',str(record['video']),
           '-frames:v',str(max_frames+1),'-an','-vf',vf,'-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    proc = subprocess.run(cmd, shell=False, capture_output=True, timeout=90)
    (out/'ffmpeg_showinfo.log').write_bytes(proc.stderr)
    if proc.returncode: raise ValueError('Official pixel decode failed')
    unit = 224*224*3
    if not proc.stdout or len(proc.stdout)%unit: raise ValueError('Incomplete RGB frame')
    count = len(proc.stdout)//unit
    if count>max_frames: raise ValueError('Frame budget exceeded')
    source = parse_showinfo(proc.stderr, 'src'); selected = parse_showinfo(proc.stderr, 'sample')
    if len(selected)!=count: raise ValueError('Missing sampled source audit')
    lookup = {}
    for item in source:
        lookup.setdefault(item['checksum'], []).append(item['source_or_output_pts_sec'])
    phase = []
    for item in selected:
        candidates = lookup.get(item['checksum'], [])
        grid = item['source_or_output_pts_sec']
        if not candidates or not 0 <= grid < record['duration_sec']:
            raise ValueError('Invalid logical/source observation')
        if any(not 0<=t<record['duration_sec'] for t in candidates):
            raise ValueError('Out of duration source timestamp')
        phase.append({'logical_clip_start_sec': grid, 'source_pts_candidates_sec': candidates,
                      'unique_source_pts': len(candidates)==1, 'checksum': item['checksum']})
    grid = np.array([p['logical_clip_start_sec'] for p in phase])
    if not np.array_equal(grid, np.arange(count)*2.): raise ValueError('Noncanonical official logical grid')
    write_json_atomic(out/'sampling.json', phase)
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(count,224,224,3).copy(), grid, phase


def validate_cache(folder, expected):
    meta = read(folder/'metadata.json')
    if meta['signature'] != expected or meta['cache_key'] != digest_json(expected):
        raise ValueError('Official cache contract changed')
    for name, sha in meta['files_sha256'].items():
        if sha256_file(folder/name)!=sha: raise ValueError('Official cache bytes changed')
    with np.load(folder/'features.npz', allow_pickle=False) as data:
        images, grid = data['images'], data['times']
        if images.ndim!=2 or len(images)!=len(grid) or not len(grid): raise ValueError('Invalid cache shape')
        if not np.isfinite(images).all() or not np.allclose(np.linalg.norm(images,axis=1),1.,atol=1e-5):
            raise ValueError('Nonfinite or unnormalized cache')
        if not np.array_equal(grid,np.arange(len(grid))*2.) or grid[-1]>=expected['duration_sec']:
            raise ValueError('Invalid logical cache timeline')
        tef, anchor = upstream_temporal_inputs(len(grid),expected['sampling']['clip_length_sec'])
        if not np.array_equal(data['tef'],tef.numpy()) or not np.array_equal(data['anchors'],anchor.numpy()):
            raise ValueError('Official temporal contract changed')
    return meta


def freeze(cfg, out):
    if cfg['clip_length_sec']!=2. or cfg['token_cap']!=32 or cfg['length_policy']!='upstream_nonzero':
        raise ValueError('This frozen study supports only the audited official input contract')
    prior = verify_run(checked_output(ROOT/cfg['input_audit_run']))
    study = checked_output(ROOT/cfg['study_run']); parent = verify(study)
    metadata = [r for r in rows(study/'feature_records.jsonl') if r['split']=='validation']
    if len(metadata)!=cfg['expected_queries'] or len({r['video'] for r in metadata})!=cfg['expected_sources']:
        raise ValueError('Frozen validation identity count changed')
    # Remove labels from the manifest consumed by all feature/inference stages.
    inputs = [{**{k:r[k] for k in ('video','video_sha256','duration_sec','width','height','images_path','images_sha256')},
               'qid':r['entry']['qid'], 'query':r['entry']['query'], 'vid':r['entry']['vid']} for r in metadata]
    repo = checked_output(ROOT/cfg['official_repo'])
    for args, expected in [(['rev-parse','HEAD'],cfg['official_commit']), (['status','--porcelain'],'')]:
        p = subprocess.run(['git','-C',str(repo),*args],shell=False,capture_output=True,timeout=30)
        if p.returncode or p.stdout.decode().strip()!=expected: raise ValueError('Official clone changed')
    out = checked_output(out); out.mkdir(parents=True,exist_ok=False)
    files = ['scripts/univtg_input_transfer.py','configs/univtg_input_transfer_v1.json',
             'scripts/univtg_input_audit.py','queryseek/retrieval/univtg_inputs.py',
             'queryseek/retrieval/curve_spans.py','queryseek/eval/qvhighlights.py',
             'scripts/replay_localizer_curves.py','scripts/analyze_localizer_study.py']
    files += [str((repo/name).relative_to(ROOT)) for name in ('LICENSE','run_on_video/data_utils.py',
                'run_on_video/clip/model.py','run_on_video/clip/clip.py','run_on_video/clip/simple_tokenizer.py',
                'run_on_video/clip/bpe_simple_vocab_16e6.txt.gz')]
    code = {}
    for name in files:
        target = out/'source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target); code[name]=sha256_file(target)
    preserved = frozen_inventory(study,parent)
    controls = read(ROOT/prior['config']['controls_run']/'protocol.json')
    preserved.update(controls['preserved_sha256'])
    for r in inputs:
        for path, sha in ((r['video'],r['video_sha256']),(r['images_path'],r['images_sha256'])):
            if sha256_file(path)!=sha: raise ValueError('Input bytes changed')
            preserved[path] = sha
    write_json_atomic(out/'inference_manifest.json',inputs)
    protocol = {'config':cfg,'source_sha256':code,'preserved_sha256':preserved,
                'manifest_sha256':sha256_file(out/'inference_manifest.json'),
                'clip_weights_sha256':sha256_file(cfg['weight_path']),
                'encoder_tensor_sha256':prior['encoder_weights_sha256'],
                'versions':{m:optional_version(m) for m in ('torch','open-clip-torch','numpy','safetensors')},
                'ffmpeg_version':subprocess.run(['ffmpeg','-version'],shell=False,capture_output=True,timeout=30).stdout.decode().splitlines()[0],
                'independent':False,'new_training_updates':0,'fresh_query_label_access':False,'mock':False}
    write_json_atomic(out/'protocol.json',protocol); write_json_atomic(out/'status.json',{'phase':'frozen'})
    print('OUTPUT='+str(out),flush=True)
    return out


def verify_protocol(out):
    p = read(out/'protocol.json')
    for name,sha in p['source_sha256'].items():
        if sha256_file(ROOT/name)!=sha or sha256_file(out/'source_snapshot'/name)!=sha:
            raise ValueError('Frozen source changed: '+name)
    for name,sha in p['preserved_sha256'].items():
        if sha256_file(ROOT/name)!=sha: raise ValueError('Historical input changed: '+name)
    if sha256_file(out/'inference_manifest.json')!=p['manifest_sha256'] or sha256_file(p['config']['weight_path'])!=p['clip_weights_sha256']:
        raise ValueError('Frozen manifest/encoder changed')
    return p


def predict_inputs(records, text, feature_index):
    """Only query/feature/timeline inputs; no label or GT argument."""
    old, new, pools = {}, {}, {}
    for r in records:
        qid = r['qid']; query = text[f'pool_{qid}']
        with np.load(r['images_path'],allow_pickle=False) as data:
            old[qid],_ = curve_predictions({'times':data['times'],'duration_sec':r['duration_sec']},data['images']@query)
        if r['video_sha256'] not in feature_index:
            new[qid],pools[qid] = [],[]; continue
        with np.load(Path(feature_index[r['video_sha256']]['folder'])/'features.npz',allow_pickle=False) as data:
            new[qid],pools[qid] = curve_predictions({'times':data['times'],'duration_sec':r['duration_sec']},data['images']@query)
    return old,new,pools


def evaluate_saved(out, write=False):
    """Separate label boundary: load GT only after immutable predictions exist."""
    protocol = verify_protocol(out); cfg = protocol['config']
    gt = {e['qid']:e for e in rows(ROOT/cfg['study_run']/'validation.jsonl')}
    pred = lambda name:{int(k):v for k,v in read(out/name).items()}
    old, new, pools = pred('old_cap32_predictions.json'),pred('official_cap32_predictions.json'),pred('official_pools.json')
    if set(old)!=set(gt) or set(new)!=set(gt): raise ValueError('Evaluation denominator changed')
    legacy = {int(k):v for k,v in read(ROOT/cfg['study_run']/'e4_predictions.json').items()}
    result = {'old1fps_cap32':dict(eval_mr_brief(old,gt)), 'official2s_cap32':dict(eval_mr_brief(new,gt)),
              'legacy_old1fps_cap77':dict(eval_mr_brief(legacy,gt)),
              'official_candidate_diagnostic':candidate_diagnostic(pools,new,gt),
              'paired_official_minus_old_cap32':paired_clusters(per_query(old,gt),per_query(new,gt),**cfg['bootstrap']),
              'independent':False,'locator_metrics':'unmeasured','algorithm_improvement_claim':False}
    result['paired_official_minus_old_cap32']['scope'] = 'joint input portability on previously observed internal validation, no checkpoint selection; not independent generalization'
    if write: write_json_atomic(out/'evaluation.json',result)
    return result


def replay(out):
    """Rebuild predictions/metrics from saved features, with zero model calls."""
    started=time.perf_counter();p=verify_protocol(out);cfg=p['config']
    metrics=read(out/'metrics.json')
    for name,sha in metrics['artifact_sha256'].items():
        if sha256_file(out/name)!=sha: raise ValueError('Saved study artifact changed')
    index=read(out/'feature_index.json')
    records=read(out/'inference_manifest.json')
    by_sha={r['video_sha256']:r for r in records}
    versions={**p['versions'],'ffmpeg':p['ffmpeg_version']}
    for sha,item in index.items():
        folder=checked_output(Path(item['folder']))
        if sha256_file(folder/'metadata.json')!=item['metadata_sha256']: raise ValueError('Cache metadata changed')
        validate_cache(folder,signature(by_sha[sha],cfg,p['clip_weights_sha256'],p['source_sha256'],versions))
    with np.load(out/'text_features.npz',allow_pickle=False) as data:
        text={k:data[k] for k in data.files}
    values=predict_inputs(records,text,index)
    for name,expected in zip(('old_cap32_predictions.json','official_cap32_predictions.json','official_pools.json'),values):
        actual={int(k):v for k,v in read(out/name).items()}
        if actual.keys()!=expected.keys(): raise ValueError('Prediction identities changed')
        for qid in actual:
            if np.asarray(actual[qid]).shape!=np.asarray(expected[qid]).shape or not np.allclose(actual[qid],expected[qid],atol=1e-7,rtol=1e-7):
                raise ValueError('Saved predictions cannot be rebuilt')
    evaluation=evaluate_saved(out)
    if evaluation!=read(out/'evaluation.json'): raise ValueError('Saved evaluation cannot be rebuilt')
    result={'passed':True,'queries':len(records),'caches':len(index),'model_calls':0,
            'new_training_updates':0,'predictions_and_intervals_rebuilt':True,'elapsed_sec':time.perf_counter()-started}
    write_json_atomic(out/('replay_'+new_job_id()+'.json'),result)
    return result


def run(out):
    started = time.perf_counter(); protocol = verify_protocol(out); cfg=protocol['config']
    if read(out/'status.json')['phase']!='frozen': raise ValueError('Run already started; replay existing artifacts instead')
    write_json_atomic(out/'status.json',{'phase':'running'})
    checkpoint_gate(cfg,out)
    torch.set_num_threads(cfg['torch_threads']); torch.use_deterministic_algorithms(True)
    import open_clip
    from safetensors.torch import load_file
    from scripts.localizer_pilot import model_weights_sha
    model,_,_ = open_clip.create_model_and_transforms('ViT-B-32',pretrained=None,force_quick_gelu=True,device='cpu')
    model.load_state_dict(load_file(cfg['weight_path'],device='cpu'),strict=True); model.eval()
    class Wrapper: _model=model
    if model_weights_sha(Wrapper())!=protocol['encoder_tensor_sha256']: raise ValueError('Encoder tensors changed')
    official = import_official(checked_output(ROOT/cfg['official_repo']))
    records = read(out/'inference_manifest.json'); text = {}; lengths = []
    for start in range(0,len(records),cfg['batch_size']):
        batch=records[start:start+cfg['batch_size']]
        tokens=official.tokenize([r['query'] for r in batch],max_valid_length=cfg['token_cap'])
        hidden,pool = text_hidden_states(model,tokens)
        valid=valid_text_features(hidden,tokens,length_policy=cfg['length_policy'])
        for r,h,q,t in zip(batch,valid,pool,tokens):
            text[f'tokens_{r["qid"]}']=h.numpy()
            text[f'pool_{r["qid"]}']=torch.nn.functional.normalize(q,dim=-1).numpy()
            lengths.append({'qid':r['qid'],'nonzero_length':len(h),'first_eot_length':int((t==49407).nonzero()[0])+1})
    np.savez(out/'text_features.npz',**text)
    text_contract={'cap':cfg['token_cap'],'length_policy':cfg['length_policy'],
        'lengths':lengths,'weights_sha256':protocol['clip_weights_sha256'],
        'manifest_sha256':protocol['manifest_sha256'],'source_sha256':protocol['source_sha256'],
        'text_features_sha256':sha256_file(out/'text_features.npz')}
    text_contract['cache_key']=digest_json(text_contract)
    write_json_atomic(out/'text_contract.json',text_contract)
    unique={r['video_sha256']:r for r in records}; index={}; failures=[]; frames=0; cache_hits=0
    versions={**protocol['versions'],'ffmpeg':protocol['ffmpeg_version']}
    for position,(sha,r) in enumerate(sorted(unique.items()),1):
        if time.perf_counter()-started>cfg['wall_cap_sec']: raise TimeoutError('Frozen study wall cap')
        sig=signature(r,cfg,protocol['clip_weights_sha256'],protocol['source_sha256'],versions)
        key=digest_json(sig); folder=checked_output(ROOT/'outputs/univtg_official_input_cache'/key)
        try:
            if (folder/'metadata.json').exists():
                meta=validate_cache(folder,sig); cache_hits+=1
            else:
                folder.mkdir(parents=True,exist_ok=False)
                pixels,grid,phase=decode_pixels(r,folder,cfg['max_new_frames']-frames)
                features=[]
                for start in range(0,len(pixels),cfg['batch_size']):
                    if time.perf_counter()-started>cfg['wall_cap_sec']: raise TimeoutError('Frozen study wall cap')
                    x=pixel_tensor(pixels[start:start+cfg['batch_size']])
                    with torch.inference_mode(): f=torch.nn.functional.normalize(model.encode_image(x),dim=-1)
                    if not torch.isfinite(f).all(): raise ValueError('Nonfinite image feature')
                    features.append(f.numpy()); frames+=len(f)
                tef,anchor=upstream_temporal_inputs(len(grid),cfg['clip_length_sec'])
                np.savez(folder/'features.npz',images=np.concatenate(features),times=grid,tef=tef.numpy(),anchors=anchor.numpy())
                meta={'signature':sig,'cache_key':key,'frames':len(grid),
                      'unique_source_pts_count':sum(p['unique_source_pts'] for p in phase),
                      'files_sha256':{name:sha256_file(folder/name) for name in ('features.npz','sampling.json','ffmpeg_showinfo.log')}}
                write_json_atomic(folder/'metadata.json',meta); validate_cache(folder,sig)
            index[sha]={'folder':str(folder),'cache_key':key,'frames':meta['frames'],
                        'metadata_sha256':sha256_file(folder/'metadata.json')}
        except TimeoutError:
            raise
        except Exception as exc:
            failures.append({'video_sha256':sha,'error_type':type(exc).__name__,
                             'reason':str(exc) if isinstance(exc,ValueError) else 'feature acquisition failed'})
        write_json_atomic(out/'feature_index.json',index); write_json_atomic(out/'failures.json',failures)
        write_json_atomic(out/'progress.json',{'videos_done':position,'videos_total':len(unique),'new_frames':frames,
                                             'failures':len(failures),'elapsed_sec':time.perf_counter()-started})
        print(f'VIDEO {position}/{len(unique)} frames={frames} failures={len(failures)} elapsed_sec={time.perf_counter()-started:.2f}',flush=True)
    old,new,pools=predict_inputs(records,text,index)
    for name,value in [('old_cap32_predictions.json',old),('official_cap32_predictions.json',new),('official_pools.json',pools)]:
        write_json_atomic(out/name,value)
    result=evaluate_saved(out,write=True)
    result.update(new_image_encodings=frames,feature_cache_hits=cache_hits,failed_videos=len(failures),
                  successful_queries=sum(r['video_sha256'] in index for r in records),total_queries=len(records),
                  vlm_calls=0,training_updates=0,locator_weights_loaded=False,fresh_query_label_access=False,
                  elapsed_sec=time.perf_counter()-started)
    result['artifact_sha256']={name:sha256_file(out/name) for name in ('text_features.npz','text_contract.json',
        'old_cap32_predictions.json','official_cap32_predictions.json','official_pools.json','feature_index.json',
        'failures.json','evaluation.json','checkpoint_gate.json')}
    verify_protocol(out)
    write_json_atomic(out/'metrics.json',result); write_json_atomic(out/'status.json',{'phase':'complete'})
    print(json.dumps(result,indent=2),flush=True); print('OUTPUT='+str(out),flush=True)
    return result


def mock(out):
    from scripts.make_synthetic_fixture import build_video
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    video=build_video(out/'fixture',duration=12.); folder=out/'pixel_probe';folder.mkdir()
    r={'video':video,'width':320,'height':240,'duration_sec':12.}
    pixels,grid,phase=decode_pixels(r,folder,20)
    pred,_=curve_predictions({'times':grid,'duration_sec':12.},np.linspace(0,1,len(grid)))
    result={'mock':True,'frames':len(pixels),'predictions':pred,'model_calls':0,
            'scope':'synthetic fixture plumbing only; no encoder or localization accuracy'}
    write_json_atomic(out/'metrics.json',result);print('OUTPUT='+str(out),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','run','replay','mock'))
    parser.add_argument('--config',default='configs/univtg_input_transfer_v1.json')
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.phase=='freeze': freeze(read(ROOT/args.config),ROOT/'outputs/univtg_input_transfer'/new_job_id())
    elif args.phase=='mock': mock(ROOT/'outputs/univtg_input_transfer_mock'/new_job_id())
    else:
        if args.run is None: parser.error('--run required')
        out=checked_output(args.run)
        if args.phase=='run':
            try: run(out)
            except Exception as exc:
                write_json_atomic(out/'status.json',{'phase':'failed','error_type':type(exc).__name__});raise
        else: print(json.dumps(replay(out),indent=2))
