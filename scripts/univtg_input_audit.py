"""Offline CLIP/token/TEF and bounded pixel preprocessing parity audit.

Official source is imported read-only from the pinned existing clone. Models
share cached tensors; no downloads, localizer weights, labels or tuning.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from queryseek.retrieval.univtg_inputs import text_hidden_states, valid_text_features, upstream_temporal_inputs, numeric_difference
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.tools.export import new_job_id
from scripts.localizer_query_controls import verify_controls
from scripts.localizer_study import verify, read, rows
from scripts.sampling_probe import checked_output, optional_version


def parse_showinfo(stderr, tag):
    result = []
    for line in stderr.decode(errors='replace').splitlines():
        if f'showinfo@{tag} ' not in line: continue
        match = re.search(r'\bn:\s*(\d+).*?pts_time:\s*([-+\d.eE]+).*?\bchecksum:([A-F0-9]+)',line)
        if match:
            result.append({'frame_index':int(match[1]),'source_or_output_pts_sec':float(match[2]),'checksum':match[3]})
    return result


def sample_pixels(video, width, height, count, out):
    """Replicate official fps/FFmpeg scale/crop, while auditing source PTS."""
    if min(width,height,count)<=0: raise ValueError('Invalid pixel audit dimensions')
    target = checked_output(out); target.mkdir(parents=True,exist_ok=False)
    resized = (int(height*224/width),224) if height>=width else (224,int(width*224/height))
    rh,rw = resized
    vf = f'showinfo@src,fps=1/2,showinfo@sample,scale={rw}:{rh},crop=224:224:{int((rw-224)/2)}:{int((rh-224)/2)}'
    cmd = ['ffmpeg','-hide_banner','-loglevel','info','-threads','1','-i',str(video),
           '-frames:v',str(count),'-an','-vf',vf,'-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    proc = subprocess.run(cmd,shell=False,capture_output=True,timeout=90)
    if proc.returncode: raise RuntimeError('FFmpeg pixel audit failed')
    if len(proc.stdout)!=count*224*224*3: raise ValueError('Incomplete official pixel sample')
    source = parse_showinfo(proc.stderr,'src'); sampled = parse_showinfo(proc.stderr,'sample')[:count]
    if len(sampled)!=count: raise ValueError('Missing sampled frame audit')
    records = []
    for item in sampled:
        matches = [s for s in source if s['checksum']==item['checksum']]
        if not matches: raise ValueError('FPS output not traceable to source pixels')
        # Duplicated pictures expose ambiguity; never claim their first PTS is unique.
        records.append({'official_output_grid_sec':item['source_or_output_pts_sec'],
                        'source_pts_candidates_sec':[s['source_or_output_pts_sec'] for s in matches],
                        'unique_source_pts':len(matches)==1,'source_frame_index_for_same_pixel_control':matches[0]['frame_index'],
                        'checksum':item['checksum']})
    pixels = np.frombuffer(proc.stdout,dtype=np.uint8).reshape(count,224,224,3).copy()
    (target/'ffmpeg_showinfo.log').write_bytes(proc.stderr)
    np.save(target/'official_pixels.npy',pixels,allow_pickle=False)
    write_json_atomic(target/'sampling.json',records)
    return pixels,records


def pipeline_same_frame(video, frame_index, out):
    path = checked_output(out)
    vf = f"select='eq(n,{int(frame_index)})',scale='if(gt(iw,ih),512,-2)':'if(gt(iw,ih),-2,512)'"
    proc = subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(video),
                           '-vf',vf,'-frames:v','1','-fps_mode','vfr','-q:v','3',str(path)],
                          shell=False,capture_output=True,timeout=90)
    if proc.returncode or not path.is_file(): raise RuntimeError('Same-source-frame JPEG control failed')
    return path


def freeze(config,out):
    verify_controls(checked_output(ROOT/config['controls_run']))
    study = checked_output(ROOT/config['study_run']); parent = verify(study)
    metadata = rows(study/'feature_records.jsonl')
    train = [r for r in metadata if r['split']=='train']
    queries = [r['entry']['query'] for r in sorted(train,key=lambda r:r['entry']['qid'])[:config['train_query_count']]]
    texts = config['synthetic_texts'][:-1]+[config['synthetic_texts'][-1]*config['long_probe_repetitions']]
    selected = []; seen = set()
    for r in sorted(train,key=lambda r:r['entry']['vid']):
        if r['entry']['vid'] in seen: continue
        seen.add(r['entry']['vid'])
        selected.append({k:r[k] for k in ['video','video_sha256','width','height','duration_sec',
                                       'images_path','images_sha256','cache_key']})
        if len(selected)==config['video_count']: break
    repo = checked_output(ROOT/config['official_repo'])
    commit = subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],shell=False,capture_output=True,timeout=30)
    dirty = subprocess.run(['git','-C',str(repo),'status','--porcelain'],shell=False,capture_output=True,timeout=30)
    if commit.returncode or dirty.returncode or commit.stdout.decode().strip()!=config['official_commit'] or dirty.stdout.strip():
        raise ValueError('Pinned official clone changed')
    out = checked_output(out); out.mkdir(parents=True,exist_ok=False)
    names = ['scripts/univtg_input_audit.py','queryseek/retrieval/univtg_inputs.py','configs/univtg_input_audit_v1.json']
    names += [str((repo/name).relative_to(ROOT)) for name in ['run_on_video/clip/model.py','run_on_video/clip/clip.py',
               'run_on_video/clip/simple_tokenizer.py','run_on_video/clip/bpe_simple_vocab_16e6.txt.gz','LICENSE',
               'run_on_video/video_loader.py','main_gradio.py']]
    source = {}
    for name in names:
        target = out/'source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target); source[name] = sha256_file(target)
    weights = Path(config['weight_path'])
    if not weights.is_file(): raise ValueError('Offline cached CLIP weights missing; no download allowed')
    inputs = {str(weights):sha256_file(weights),str(study/'feature_records.jsonl'):sha256_file(study/'feature_records.jsonl')}
    for r in selected:
        inputs[r['video']] = r['video_sha256']; inputs[r['images_path']] = r['images_sha256']
    protocol = {'config':config,'texts':queries+texts,'real_train_text_count':len(queries),'videos':selected,
                'source_sha256':source,'input_sha256':inputs,'encoder_weights_sha256':read(study/'features_summary.json')['weights_sha256'],
                'versions':{m:optional_version(m) for m in ['torch','open-clip-torch','numpy','safetensors']},
                'mock':False,'fresh_query_label_access':False,'GT_inference':False,'localizer_weights_loaded':False,
                'upstream_tokenizer_default_cap':32,'scope':config['scope']}
    write_json_atomic(out/'protocol.json',protocol)
    write_json_atomic(out/'status.json',{'phase':'frozen'})
    print('OUTPUT='+str(out),flush=True)
    return out


def verify_run(out):
    protocol = read(out/'protocol.json')
    for path,sha in protocol['input_sha256'].items():
        if sha256_file(path)!=sha: raise ValueError('Frozen input changed')
    for name,sha in protocol['source_sha256'].items():
        if sha256_file(ROOT/name)!=sha or sha256_file(out/'source_snapshot'/name)!=sha:
            raise ValueError('Frozen audit source changed')
    verify_controls(checked_output(ROOT/protocol['config']['controls_run']))
    return protocol


def import_official(repo):
    # Suppress .pyc writes to the preserved upstream clone.
    previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location('_qs_official_univtg_clip',repo/'run_on_video/clip/__init__.py',
                 submodule_search_locations=[str(repo/'run_on_video/clip')])
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally: sys.dont_write_bytecode = previous
    return module


def pixel_tensor(pixels):
    tensor = torch.from_numpy(pixels.copy()).permute(0,3,1,2).float()/255.
    mean = tensor.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
    std = tensor.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
    return (tensor-mean)/(std+1e-8)


def run(out):
    started = time.perf_counter(); protocol = verify_run(out); cfg = protocol['config']
    torch.set_num_threads(cfg['torch_threads']); torch.use_deterministic_algorithms(True)
    write_json_atomic(out/'status.json',{'phase':'running'})
    result = {'mock':False,'scope':cfg['scope'],'independent':False,'new_training_updates':0,
              'vlm_calls':0,'weights_downloaded':False,'localizer_weights_loaded':False,'fresh_inference':False}
    try:
        import open_clip
        from safetensors.torch import load_file
        from scripts.localizer_pilot import model_weights_sha
        model,_,preprocess = open_clip.create_model_and_transforms('ViT-B-32',pretrained=None,device='cpu',force_quick_gelu=True)
        state = load_file(cfg['weight_path'],device='cpu'); model.load_state_dict(state,strict=True); model.eval()
        class Wrapper: _model = model
        if model_weights_sha(Wrapper())!=protocol['encoder_weights_sha256']:
            raise ValueError('Cached model tensors differ from frozen feature encoder')
        official = import_official(checked_output(ROOT/cfg['official_repo']))
        reference = official.clip.build_model(dict(state)).float().eval()
        tokens = official.tokenize(protocol['texts'],max_valid_length=cfg['upstream_max_valid_tokens'])
        standard_tokens = open_clip.get_tokenizer('ViT-B-32')(protocol['texts'])
        result['tokenization'] = {'context_length':tokens.shape[1],'upstream_valid_token_cap':32,
           'different_from_default_openclip_rows':[i for i in range(len(tokens)) if not torch.equal(tokens[i],standard_tokens[i])],
           'nonzero_lengths':(tokens!=0).sum(1).tolist(),'first_eot_lengths':(tokens==49407).int().argmax(1).add(1).tolist()}
        # Report the official builder's half-rounding path separately, then restore
        # the same original FP32 tensors before the primary architecture comparison.
        rounded_hidden,_ = text_hidden_states(reference,tokens)
        reference.load_state_dict(state,strict=True)
        with torch.inference_mode(): official_output = reference.encode_text(tokens)
        ours,pooled = text_hidden_states(model,tokens)
        native = official_output['last_hidden_state'].float()
        raw = numeric_difference(ours.numpy(),native.detach().numpy())
        normalized = numeric_difference(torch.nn.functional.normalize(ours,dim=-1).numpy(),
                torch.nn.functional.normalize(native.detach(),dim=-1).numpy())
        result['text_parity'] = {'raw_hidden':raw,'row_normalized_hidden':normalized,
             'projected_pool':numeric_difference(pooled.numpy(),official_output['pooler_output'].detach().numpy()),
             'half_rounding_control':numeric_difference(rounded_hidden.numpy(),native.detach().numpy()),
             'passed_frozen_tolerances':raw['max_abs']<=cfg['primary_raw_atol'] and normalized['max_abs']<=cfg['primary_normalized_atol']}
        features = valid_text_features(ours,tokens,length_policy='upstream_nonzero')
        np.savez(out/'text_features.npz',**{f'text_{i}':x.numpy() for i,x in enumerate(features)})
        result['text_parity']['exported_lengths'] = [len(x) for x in features]
        del rounded_hidden,state,official_output
        result['videos'] = []; observations = 0
        for i,r in enumerate(protocol['videos']):
            if time.perf_counter()-started>cfg['cost_cap_sec']: raise TimeoutError('Input audit wall cap')
            folder = out/'pixel_probes'/str(i)
            pixels,phase = sample_pixels(r['video'],r['width'],r['height'],cfg['video_frames_each'],folder)
            x = pixel_tensor(pixels)
            with torch.inference_mode():
                ours_visual = torch.nn.functional.normalize(model.encode_image(x),dim=-1)
                native_visual = torch.nn.functional.normalize(reference.encode_image(x),dim=-1)
            observations += 2*len(x)
            same = []
            for j,stamp in enumerate(phase):
                path = pipeline_same_frame(r['video'],stamp['source_frame_index_for_same_pixel_control'],folder/f'same_source_{j}.jpg')
                with Image.open(path) as image: same.append(preprocess(image.convert('RGB')))
            same = torch.stack(same)
            with torch.inference_mode():
                same_visual = torch.nn.functional.normalize(model.encode_image(same),dim=-1)
            observations += len(same)
            with np.load(r['images_path'],allow_pickle=False) as data:
                times = data['times']; indices = [int(np.argmin(np.abs(times-s['official_output_grid_sec']))) for s in phase]
                cache = data['images'][indices]
            visual_parity = numeric_difference(ours_visual.numpy(),native_visual.numpy())
            delta = numeric_difference(x.numpy(),same.numpy())
            tef,anchors = upstream_temporal_inputs(len(times)//2,cfg['clip_length_sec'])
            item = {'video_sha256':r['video_sha256'],'sampling':phase,'cached_grid_pts_sec':times[indices].tolist(),
                    'identical_pixel_encoder_parity':visual_parity,'encoder_parity_passed':visual_parity['max_abs']<=cfg['primary_normalized_atol'],
                    'same_source_frame_preprocess_delta':delta,
                    'same_source_frame_feature_cosine':(ours_visual*same_visual).sum(-1).tolist(),
                    'official_vs_existing_grid_cache_cosine':(ours_visual*torch.from_numpy(cache)).sum(-1).tolist(),
                    'official_tef_first_last':tef[[0,-1]].tolist(),'official_anchor_first_last':anchors[[0,-1]].tolist(),
                    'official_anchor_is_not_center_convention':True}
            np.savez(folder/'compared_features.npz',official=ours_visual.numpy(),same_pixels_native=native_visual.numpy(),
                     pipeline_same_source=same_visual.numpy(),existing_grid_cache=cache)
            result['videos'].append(item)
            write_json_atomic(out/'metrics.json',result)
        if observations>cfg['max_new_image_forward_observations']: raise ValueError('Image observation cap exceeded')
        result['new_image_forward_observations'] = observations
        result['new_unique_source_images'] = cfg['video_count']*cfg['video_frames_each']
        result['all_identical_input_encoder_parity_passed'] = result['text_parity']['passed_frozen_tolerances'] and all(r['encoder_parity_passed'] for r in result['videos'])
        result['existing_feature_cache_is_official_compatible'] = False
        result['decision'] = 'token extraction can be audited separately; existing pool/PTS-gap/1fps/JPEG cache cannot be relabeled as official inputs'
        verify_run(out)
        result['preserved_inputs_unchanged'] = True
        result['elapsed_sec'] = time.perf_counter()-started
        write_json_atomic(out/'metrics.json',result); write_json_atomic(out/'status.json',{'phase':'complete'})
    except Exception as exc:
        result.update(error_type=type(exc).__name__,elapsed_sec=time.perf_counter()-started)
        if isinstance(exc,(ValueError,TimeoutError)): result['reason']=str(exc)
        write_json_atomic(out/'failure.json',result); write_json_atomic(out/'status.json',{'phase':'failed'})
        raise
    print(json.dumps(result,indent=2),flush=True); print('OUTPUT='+str(out),flush=True)
    return result


def mock(out):
    from scripts.make_synthetic_fixture import build_video
    out = checked_output(out); out.mkdir(parents=True,exist_ok=False)
    video = build_video(out/'fixture')
    pixels,phase = sample_pixels(video,320,240,3,out/'pixel_probe')
    tef,anchor = upstream_temporal_inputs(len(pixels),2.)
    result = {'mock':True,'scope':'synthetic plumbing, not encoder/locator accuracy','sampling':phase,
              'pixel_shape':list(pixels.shape),'tef':tef.tolist(),'anchor':anchor.tolist(),'model_calls':0,
              'encoder_parity':'unmeasured','localization_metrics':'unmeasured'}
    write_json_atomic(out/'metrics.json',result); print('OUTPUT='+str(out),flush=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['freeze','run','mock'])
    parser.add_argument('--config',default='configs/univtg_input_audit_v1.json'); parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.phase=='freeze': freeze(read(ROOT/args.config),ROOT/'outputs/univtg_input_audit'/new_job_id())
    elif args.phase=='mock': mock(ROOT/'outputs/univtg_input_audit_mock'/new_job_id())
    else:
        if args.run is None: parser.error('--run required')
        run(checked_output(args.run))
