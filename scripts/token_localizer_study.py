"""Frozen data x capacity experiment: native UniVTG, token text, five losses.

Four fixed update budgets; no validation/model selection during training. Reuse
all image features, encode only missing train text tokens, resume own checkpoints.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from queryseek import PIPELINE_VERSION
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.token_localizer import (make_network,make_criterion,training_target,collate_inputs,
    collate_targets,total_loss,decode_dense)
from queryseek.retrieval.univtg_inputs import text_hidden_states,valid_text_features
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.tools.export import new_job_id
from scripts.localizer_study import read,rows,verify,load_records
from scripts.candidate_ranker_study import verify_inputs as verify_ranker,digest
from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output,optional_version
from scripts.univtg_input_audit import import_official
from scripts.replay_localizer_curves import curve_predictions
from scripts.analyze_localizer_study import per_query,paired_clusters,candidate_diagnostic


def state_sha(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_checkpoint(path,value):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def freeze(cfg,out):
    ranker=checked_output(ROOT/cfg['ranker_run']);prior=verify_ranker(ranker)
    previous=read(ranker/'metrics.json')
    for name,sha in previous['artifact_sha256'].items():
        if sha256_file(ranker/name)!=sha:raise ValueError('Previous fitted ranker changed')
    study=checked_output(ROOT/cfg['study_run']);parent=verify(study);records=load_records(study,parent)
    by_qid={r['entry']['qid']:r for r in records};scales={}
    for scale in cfg['train_scales']:
        ids=[r['qid'] for r in rows(study/f'train_{scale}.jsonl') if r['qid'] in by_qid]
        if any(by_qid[q]['split']!='train' for q in ids):raise ValueError('Invalid training selection')
        scales[str(scale)]=ids
    if not set(scales['64'])<set(scales['256']):raise ValueError('Nested source scale changed')
    train_sources={source_video_id(by_qid[q]['entry']['vid']) for q in scales['256']}
    val_sources={source_video_id(r['entry']['vid']) for r in records if r['split']=='validation'}
    excluded=set(read(ROOT/parent['config']['data_run']/'excluded_sources.json'))
    if train_sources&val_sources or (train_sources|val_sources)&excluded:raise ValueError('Source leakage')
    summary={s:{'queries':len(ids),'sources':len({source_video_id(by_qid[q]['entry']['vid']) for q in ids})} for s,ids in scales.items()}
    if summary!={'64':{'queries':64,'sources':63},'256':{'queries':256,'sources':254}} or len(val_sources)!=64:
        raise ValueError('Expected original source identities')
    repo=checked_output(ROOT/cfg['official_repo'])
    for name,expected in read(ROOT/cfg['input_transfer_run']/'protocol.json')['source_sha256'].items():
        if str(repo.relative_to(ROOT)) in name.replace('\\','/') and sha256_file(ROOT/name)!=expected:
            raise ValueError('Pinned official source changed')
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    names=['scripts/token_localizer_study.py','queryseek/retrieval/token_localizer.py','configs/token_localizer_v1.json',
           'queryseek/retrieval/univtg_inputs.py','scripts/univtg_input_audit.py','queryseek/embeddings/openclip_encoder.py',
           'queryseek/eval/qvhighlights.py','queryseek/retrieval/curve_spans.py','scripts/analyze_localizer_study.py',
           'scripts/replay_localizer_curves.py']
    names += [str(f.relative_to(ROOT)) for f in (ROOT/'queryseek/vendor/univtg').glob('*.py')]
    names += ['queryseek/vendor/univtg/LICENSE']
    names += [str((repo/name).relative_to(ROOT)) for name in ('LICENSE','run_on_video/clip/clip.py',
              'run_on_video/clip/simple_tokenizer.py','run_on_video/clip/bpe_simple_vocab_16e6.txt.gz','main/dataset.py')]
    code={}
    for name in names:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target);code[name]=sha256_file(target)
    preserved=dict(prior['preserved_sha256'])
    preserved.update({str(f):sha256_file(f) for f in ranker.rglob('*') if f.is_file()})
    for r in records:
        for path,sha in ((r['images_path'],r['images_sha256']),(r['text_path'],r['text_sha256']),(r['video'],r['video_sha256'])):
            if sha256_file(path)!=sha:raise ValueError('Original feature/video changed')
            preserved[path]=sha
    shutil.copyfile(ROOT/'docs/EVAL_TOKEN_LOCALIZER.md',out/'protocol_notes_frozen.md')
    manifest=[{'qid':r['entry']['qid'],'source':source_video_id(r['entry']['vid']),'split':r['split'],
               'query':r['entry']['query'],'query_sha256':hashlib.sha256(r['entry']['query'].encode()).hexdigest(),
               'video_sha256':r['video_sha256'],'duration_sec':r['duration_sec'],'images_path':r['images_path'],
               'images_sha256':r['images_sha256'],'cache_key':r['cache_key']} for r in records]
    write_json_atomic(out/'inference_manifest.json',manifest)
    p={'config':cfg,'source_sha256':code,'preserved_sha256':preserved,'scales':scales,'scale_summary':summary,
       'manifest_sha256':sha256_file(out/'inference_manifest.json'),'protocol_notes_sha256':sha256_file(out/'protocol_notes_frozen.md'),
       'clip_weights_sha256':sha256_file(cfg['weight_path']),
       'encoder_tensor_sha256':read(study/'features_summary.json')['weights_sha256'],
       'versions':{m:optional_version(m) for m in ('torch','open-clip-torch','numpy','safetensors','threadpoolctl')},
       'pipeline_version':PIPELINE_VERSION,'source_overlap':0,'excluded_overlap':0,'fresh_query_label_access':False,
       'mock':False,'paper_reproduction':False,'independent':False,'external_locator_weights':False}
    write_json_atomic(out/'protocol.json',p);write_json_atomic(out/'status.json',{'phase':'frozen'})
    print('OUTPUT='+str(out),flush=True);return out


def verify_protocol(out):
    p=read(out/'protocol.json')
    for name,sha in p['source_sha256'].items():
        if sha256_file(ROOT/name)!=sha or sha256_file(out/'source_snapshot'/name)!=sha:raise ValueError('Frozen source changed: '+name)
    for name,sha in p['preserved_sha256'].items():
        if sha256_file(ROOT/name)!=sha:raise ValueError('Previous artifact changed: '+name)
    for name,key in [('inference_manifest.json','manifest_sha256'),('protocol_notes_frozen.md','protocol_notes_sha256')]:
        if sha256_file(out/name)!=p[key]:raise ValueError('Frozen manifest/notes changed')
    if sha256_file(p['config']['weight_path'])!=p['clip_weights_sha256']:raise ValueError('CLIP weights changed')
    return p


def load_inference_records(out):
    visual={};result=[]
    for item in read(out/'inference_manifest.json'):
        path=item['images_path']
        if path not in visual:
            if sha256_file(path)!=item['images_sha256']:raise ValueError('Visual corruption')
            with np.load(path,allow_pickle=False) as f:visual[path]=(f['images'],f['times'])
        images,times=visual[path];result.append({**item,'images':images,'times':times})
    return result


def sampling_plan(records,targets,steps,batch_size,seed):
    groups=defaultdict(list)
    for r in records:groups[r['source']].append(r['qid'])
    if len(groups)<batch_size:raise ValueError('Batch must contain distinct source videos')
    sources=sorted(groups);rng=np.random.default_rng(seed);result=[]
    for _ in range(steps):
        chosen=rng.choice(len(sources),batch_size,replace=False)
        qids=[int(rng.choice(sorted(groups[sources[i]]))) for i in chosen]
        positives=[int(rng.choice(targets[q]['timestamp_window'].nonzero().flatten().numpy())) for q in qids]
        result.append({'qids':qids,'positive_positions':positives})
    return result


def prepare(out,p):
    cfg=p['config'];records=load_inference_records(out)
    cache_signature={'pipeline_version':PIPELINE_VERSION,'clip_weights_sha256':p['clip_weights_sha256'],
        'token_cap':cfg['token_cap'],'length_policy':cfg['length_policy'],'manifest_sha256':p['manifest_sha256'],
        'source_sha256':p['source_sha256'],'versions':p['versions']}
    cache=checked_output(ROOT/'outputs/token_localizer_text_cache'/digest(cache_signature))
    if (cache/'metadata.json').exists():
        meta=read(cache/'metadata.json')
        if meta['signature']!=cache_signature or sha256_file(cache/'features.npz')!=meta['features_sha256']:
            raise ValueError('Token cache corruption')
    else:
        cache.mkdir(parents=True,exist_ok=False);text={};lengths=[]
        old=ROOT/cfg['input_transfer_run'];old_manifest={r['qid']:r for r in read(old/'inference_manifest.json')}
        old_contract={r['qid']:r for r in read(old/'text_contract.json')['lengths']}
        with np.load(old/'text_features.npz',allow_pickle=False) as data:
            for r in records:
                if r['split']=='validation':
                    q=r['qid']
                    if old_manifest[q]['query']!=r['query']:raise ValueError('Cached validation text differs')
                    text[f'tokens_{q}']=data[f'tokens_{q}'];text[f'pool_{q}']=data[f'pool_{q}']
                    lengths.append({**old_contract[q],'reused':True})
        missing=[r for r in records if f'tokens_{r["qid"]}' not in text]
        if missing:
            import open_clip
            from safetensors.torch import load_file
            from scripts.localizer_pilot import model_weights_sha
            model,_,_=open_clip.create_model_and_transforms('ViT-B-32',pretrained=None,force_quick_gelu=True,device='cpu')
            model.load_state_dict(load_file(cfg['weight_path'],device='cpu'),strict=True);model.eval()
            class Wrapper:_model=model
            if model_weights_sha(Wrapper())!=p['encoder_tensor_sha256']:raise ValueError('CLIP tensor SHA differs')
            tokenizer=import_official(checked_output(ROOT/cfg['official_repo'])).tokenize
            for start in range(0,len(missing),cfg['text_batch_size']):
                batch=missing[start:start+cfg['text_batch_size']]
                ids=tokenizer([r['query'] for r in batch],max_valid_length=cfg['token_cap'])
                hidden,pool=text_hidden_states(model,ids)
                valid=valid_text_features(hidden,ids,length_policy=cfg['length_policy'])
                for r,h,q,token_ids in zip(batch,valid,pool,ids):
                    qid=r['qid'];text[f'tokens_{qid}']=h.numpy()
                    text[f'pool_{qid}']=torch.nn.functional.normalize(q,dim=-1).numpy()
                    lengths.append({'qid':qid,'nonzero_length':len(h),'first_eot_length':int((token_ids==49407).nonzero()[0])+1,'reused':False})
            del model,hidden,pool;gc.collect()
        np.savez(cache/'features.npz',**text)
        meta={'signature':cache_signature,'cache_key':digest(cache_signature),'features_sha256':sha256_file(cache/'features.npz'),
              'new_text_encodings':len(missing),'reused_text_features':len(records)-len(missing),'new_image_encodings':0,'lengths':lengths}
        write_json_atomic(cache/'metadata.json',meta)
    write_json_atomic(out/'text_cache.json',{'folder':str(cache),'metadata_sha256':sha256_file(cache/'metadata.json'),
                                           'features_sha256':meta['features_sha256'],'cache_key':meta['cache_key']})
    with np.load(cache/'features.npz',allow_pickle=False) as f:
        text_features={r['qid']:f[f'tokens_{r["qid"]}'] for r in records}
        pools={r['qid']:f[f'pool_{r["qid"]}'] for r in records}
    target_path=out/'train_targets.pt'
    if target_path.exists():targets=torch.load(target_path,weights_only=True)
    else:
        old_records=rows(ROOT/cfg['study_run']/'feature_records.jsonl')
        entries={r['entry']['qid']:r['entry'] for r in old_records if r['split']=='train'}
        targets={r['qid']:training_target(r['times'],r['duration_sec'],entries[r['qid']]) for r in records if r['split']=='train'}
        atomic_checkpoint(target_path,targets)
    if set(targets)!={r['qid'] for r in records if r['split']=='train'}:raise ValueError('Training target identity mismatch')
    for scale,qids in p['scales'].items():
        path=out/f'sampling_plan_{scale}.json'
        planned=sampling_plan([r for r in records if r['qid'] in set(qids)],targets,cfg['steps'],cfg['batch_size'],cfg['seed'])
        if path.exists() and read(path)!=planned:raise ValueError('Frozen source sampling plan changed')
        if not path.exists():write_json_atomic(path,planned)
    manifest={'target_sha256':sha256_file(target_path),'plans_sha256':{s:sha256_file(out/f'sampling_plan_{s}.json') for s in p['scales']},
              'validation_target_access':False,'text_cache_metadata_sha256':sha256_file(cache/'metadata.json')}
    if (out/'training_inputs.json').exists() and read(out/'training_inputs.json')!=manifest:raise ValueError('Prepared inputs changed')
    write_json_atomic(out/'training_inputs.json',manifest)
    return records,text_features,pools,targets


def learning_rate(cfg,step):
    warm=min(1.,(step+1)/cfg['warmup_steps'])
    return cfg['learning_rate']*warm*cfg['decay_gamma']**sum(step>=s for s in cfg['decay_steps'])


def fit_arm(out,p,records,text,targets,arm):
    cfg=p['config'];scale,capacity=arm.split('/');folder=out/'models'/scale/capacity
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():
        fit=read(folder/'fit.json')
        if sha256_file(folder/f'step_{cfg["steps"]:04d}.pt')!=fit['checkpoint_sha256']:raise ValueError('Completed arm changed')
        return fit
    plan=read(out/f'sampling_plan_{scale}.json');lookup={r['qid']:r for r in records}
    torch.manual_seed(cfg['seed']);model=make_network(cfg['capacities'][capacity]);initial_sha=state_sha(model)
    criterion=make_criterion()
    if criterion.weight_dict!=cfg['loss_weights']:raise ValueError('Official loss weights differ')
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    step=0;history=[];previous_sec=0.
    if (folder/'progress.json').exists():
        progress=read(folder/'progress.json');path=folder/f'step_{progress["checkpoint_step"]:04d}.pt'
        if sha256_file(path)!=progress['checkpoint_sha256']:raise ValueError('Resume checkpoint changed')
        state=torch.load(path,weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);step=state['step'];history=state['history'];previous_sec=state['training_sec']
        if state['initial_sha256']!=initial_sha:raise ValueError('Resume initialization differs')
    model.train();started=time.perf_counter();last_parts=None
    for index in range(step,cfg['steps']):
        completed=sum(read(f)['training_sec'] for f in (out/'models').glob('*/*/fit.json'))
        if completed+previous_sec+time.perf_counter()-started>cfg['wall_cap_sec']:raise TimeoutError('Frozen training wall cap')
        ids=plan[index]['qids'];batch=[lookup[q] for q in ids]
        x=collate_inputs(batch,text);y=collate_targets(batch,targets,None,plan[index]['positive_positions'])
        lr=learning_rate(cfg,index)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);outputs=model(**x);loss,parts=total_loss(criterion,outputs,y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite total loss')
        loss.backward()
        if any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):
            raise ValueError('Missing or nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);optimizer.step()
        last_parts={k:float(v.detach()) for k,v in parts.items()}
        if (index+1)%cfg['log_every']==0:
            row={'step':index+1,'loss':float(loss.detach()),'components':last_parts,'learning_rate':lr}
            history.append(row);write_json_atomic(folder/'history.json',history)
            print('TRAIN',arm,'step',index+1,'loss',row['loss'],'elapsed_sec',round(previous_sec+time.perf_counter()-started,2),flush=True)
            write_json_atomic(out/'progress.json',{'arm':arm,'step':index+1,'total_steps':cfg['steps'],
                'completed_arms':len(list((out/'models').glob('*/*/fit.json'))),'current_training_sec':previous_sec+time.perf_counter()-started})
        if (index+1)%cfg['checkpoint_every']==0 or index+1==cfg['steps']:
            path=folder/f'step_{index+1:04d}.pt'
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                   'step':index+1,'initial_sha256':initial_sha,'history':history,
                   'training_sec':previous_sec+time.perf_counter()-started}
            atomic_checkpoint(path,state)
            write_json_atomic(folder/'progress.json',{'checkpoint_step':index+1,'checkpoint_sha256':sha256_file(path)})
    # A process can exit after the final checkpoint but before fit.json. Rebuild
    # its probe batch without repeating an optimizer update on exact resume.
    x=collate_inputs([lookup[q] for q in plan[-1]['qids']],text)
    final=folder/f'step_{cfg["steps"]:04d}.pt';checkpoint=torch.load(final,weights_only=True)
    restored=make_network(cfg['capacities'][capacity]);restored.load_state_dict(checkpoint['model']);restored.eval();model.eval()
    with torch.inference_mode():
        a=model(**x);b=restored(**x)
        if any(not torch.equal(a[k],b[k]) for k in ('pred_logits','pred_spans','saliency_scores')):raise ValueError('Exact checkpoint reload failed')
    result={'steps':cfg['steps'],'sampled_queries':cfg['steps']*cfg['batch_size'],'capacity':cfg['capacities'][capacity],
            'parameter_count':sum(v.numel() for v in model.parameters() if v.requires_grad),'initial_sha256':initial_sha,
            'final_tensor_sha256':state_sha(model),'checkpoint_sha256':sha256_file(final),'reload_exact':True,
            'sampling_plan_sha256':sha256_file(out/f'sampling_plan_{scale}.json'),'source_scale':p['scale_summary'][scale],
            'training_sec':checkpoint['training_sec'],'last_loss_components':last_parts or history[-1]['components']}
    write_json_atomic(folder/'fit.json',result);return result


def infer(model,records,text,cfg):
    model.eval();dense={};curve={};pools={}
    with torch.inference_mode():
        for r in records:
            result=model(**collate_inputs([r],text));prob=result['pred_logits'][0,:,0].numpy();offset=result['pred_spans'][0].numpy()
            q=r['qid'];dense[q],pools[q]=decode_dense(r,prob,offset,**cfg['dense_decoder'])
            curve[q],_=curve_predictions(r,prob)
    return {'dense':dense,'curve':curve},pools


def evaluate_saved(out,p,records,all_predictions,pools):
    cfg=p['config'];old_records=rows(ROOT/cfg['study_run']/'feature_records.jsonl')
    gt={r['entry']['qid']:r['entry'] for r in old_records if r['split']=='validation'}
    baseline={int(q):v for q,v in read(out/'e4_predictions.json').items()}
    if set(baseline)!=set(gt):raise ValueError('Baseline denominator differs')
    stats={'e4':dict(eval_mr_brief(baseline,gt)),'arms':{},'contrasts':{}}
    for arm,methods in all_predictions.items():
        if any(set(v)!=set(gt) for v in methods.values()):raise ValueError('Validation denominator differs')
        stats['arms'][arm]={'metrics':{decoder:dict(eval_mr_brief(pred,gt)) for decoder,pred in methods.items()},
                            'dense_diagnostic':candidate_diagnostic(pools[arm],methods['dense'],gt)}
        for decoder,pred in methods.items():
            stats['contrasts'][arm+'/'+decoder+'_minus_e4']=paired_clusters(per_query(baseline,gt),per_query(pred,gt),**cfg['bootstrap'])
    for decoder in ('dense','curve'):
        for capacity in ('small','large'):
            stats['contrasts']['data_effect/'+capacity+'/'+decoder]=paired_clusters(
                per_query(all_predictions['64/'+capacity][decoder],gt),per_query(all_predictions['256/'+capacity][decoder],gt),**cfg['bootstrap'])
        for scale in ('64','256'):
            stats['contrasts']['capacity_effect/'+scale+'/'+decoder]=paired_clusters(
                per_query(all_predictions[scale+'/small'][decoder],gt),per_query(all_predictions[scale+'/large'][decoder],gt),**cfg['bootstrap'])
        # Interaction: difference of capacity effects, on the same source clusters.
        first=per_query(all_predictions['64/small'][decoder],gt);second=per_query(all_predictions['64/large'][decoder],gt)
        third=per_query(all_predictions['256/small'][decoder],gt);fourth=per_query(all_predictions['256/large'][decoder],gt)
        zero={q:{'source':first[q]['source'],'ap':0.,'r1_05':0.} for q in gt}
        delta={q:{'source':first[q]['source'],'ap':fourth[q]['ap']-third[q]['ap']-second[q]['ap']+first[q]['ap'],
                  'r1_05':fourth[q]['r1_05']-third[q]['r1_05']-second[q]['r1_05']+first[q]['r1_05']} for q in gt}
        stats['contrasts']['interaction/'+decoder]=paired_clusters(zero,delta,**cfg['bootstrap'])
    for v in stats['contrasts'].values():v['scope']='fixed-final-update conditional internal factorial study; prior observation and design bias remain; no multiplicity adjustment'
    return stats


def run(out):
    started=time.perf_counter();p=verify_protocol(out);cfg=p['config']
    if (out/'metrics.json').exists() and read(out/'status.json')['phase']=='complete':raise ValueError('Completed; use replay, do not retrain')
    torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=cfg['torch_threads']):
        records,text,pool,targets=prepare(out,p)
        write_json_atomic(out/'status.json',{'phase':'training','validation_evaluated':False})
        fits={arm:fit_arm(out,p,records,text,targets,arm) for arm in cfg['order']}
        # Only now, after all four immutable final checkpoints exist, evaluate.
        for capacity in cfg['capacities']:
            if fits['64/'+capacity]['initial_sha256']!=fits['256/'+capacity]['initial_sha256']:
                raise ValueError('Initial weights differ across data scales')
            if fits['64/'+capacity]['parameter_count']!=fits['256/'+capacity]['parameter_count']:raise ValueError('Capacity changed with source scale')
        val=[r for r in records if r['split']=='validation'];baseline={}
        for r in val:baseline[r['qid']],_=curve_predictions(r,r['images']@pool[r['qid']])
        write_json_atomic(out/'e4_predictions.json',baseline)
        old_baseline={int(q):v for q,v in read(ROOT/cfg['input_transfer_run']/'old_cap32_predictions.json').items()}
        if set(baseline)!=set(old_baseline):raise ValueError('Same-feature baseline identities changed')
        for q in baseline:
            if np.asarray(baseline[q]).shape!=np.asarray(old_baseline[q]).shape or not np.allclose(baseline[q],old_baseline[q],rtol=1e-7,atol=1e-7):
                raise ValueError('Same-feature E4 baseline changed')
        all_predictions={};pools={};train_metrics={}
        for arm in cfg['order']:
            scale,capacity=arm.split('/');model=make_network(cfg['capacities'][capacity])
            ckpt=out/'models'/scale/capacity/f'step_{cfg["steps"]:04d}.pt'
            model.load_state_dict(torch.load(ckpt,weights_only=True)['model'])
            all_predictions[arm],pools[arm]=infer(model,val,text,cfg)
            for decoder,pred in all_predictions[arm].items():write_json_atomic(out/(arm.replace('/','_')+'_'+decoder+'_predictions.json'),pred)
            write_json_atomic(out/(arm.replace('/','_')+'_pools.json'),pools[arm])
            train=[r for r in records if r['qid'] in set(p['scales'][scale])]
            pred,_=infer(model,train,text,cfg)
            old={r['entry']['qid']:r['entry'] for r in rows(ROOT/cfg['study_run']/'feature_records.jsonl') if r['split']=='train'}
            gt={r['qid']:old[r['qid']] for r in train}
            train_metrics[arm]={d:dict(eval_mr_brief(v,gt)) for d,v in pred.items()}
            write_json_atomic(out/(arm.replace('/','_')+'_train_predictions.json'),pred)
        result=evaluate_saved(out,p,records,all_predictions,pools)
    result.update(fits=fits,train_metrics=train_metrics,mock=False,independent=False,new_image_encodings=0,vlm_calls=0,
                  external_locator_weights=False,fresh_query_label_access=False,elapsed_sec=time.perf_counter()-started,
                  text_preparation=read(Path(read(out/'text_cache.json')['folder'])/'metadata.json'))
    # Keep metadata concise while preserving the full token contract separately.
    result['text_preparation']={k:result['text_preparation'][k] for k in ('new_text_encodings','reused_text_features','new_image_encodings','features_sha256')}
    result['artifact_sha256']={str(f.relative_to(out)):sha256_file(f) for f in out.rglob('*') if f.is_file() and
        (f.suffix=='.pt' or f.name.endswith('_predictions.json') or f.name.endswith('_pools.json') or f.name in
         ('training_inputs.json','text_cache.json','sampling_plan_64.json','sampling_plan_256.json','fit.json','history.json'))}
    verify_protocol(out);write_json_atomic(out/'metrics.json',result);write_json_atomic(out/'status.json',{'phase':'complete'})
    print('RESULT='+json.dumps({'e4':result['e4'],'arms':result['arms'],'elapsed_sec':result['elapsed_sec']}),flush=True)
    print('OUTPUT='+str(out),flush=True);return result


def replay(out):
    started=time.perf_counter();p=verify_protocol(out);cfg=p['config'];saved=read(out/'metrics.json')
    for name,sha in saved['artifact_sha256'].items():
        if sha256_file(out/name)!=sha:raise ValueError('Saved trained artifact changed')
    torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=cfg['torch_threads']):
        records,text,pool,targets=prepare(out,p);val=[r for r in records if r['split']=='validation']
        all_predictions={};pools={};train_metrics={}
        old_entries={r['entry']['qid']:r['entry'] for r in rows(ROOT/cfg['study_run']/'feature_records.jsonl') if r['split']=='train'}
        for r in records:
            if r['split']=='validation' and r['qid'] in targets:raise ValueError('Validation target leaked into training')
            if r['split']=='train':
                rebuilt=training_target(r['times'],r['duration_sec'],old_entries[r['qid']])
                if any(not torch.equal(rebuilt[k],targets[r['qid']][k]) for k in rebuilt):raise ValueError('Training targets changed')
        for arm in cfg['order']:
            scale,capacity=arm.split('/');model=make_network(cfg['capacities'][capacity])
            model.load_state_dict(torch.load(out/'models'/scale/capacity/f'step_{cfg["steps"]:04d}.pt',weights_only=True)['model'])
            all_predictions[arm],pools[arm]=infer(model,val,text,cfg)
            for decoder,pred in all_predictions[arm].items():
                actual={int(q):v for q,v in read(out/(arm.replace('/','_')+'_'+decoder+'_predictions.json')).items()}
                if set(actual)!=set(pred):raise ValueError('Replay identities differ')
                for q in actual:
                    if not np.array_equal(np.asarray(actual[q]),np.asarray(pred[q])):raise ValueError('Prediction replay differs')
            actual_pool={int(q):v for q,v in read(out/(arm.replace('/','_')+'_pools.json')).items()}
            for q in pools[arm]:
                if not np.array_equal(np.asarray(actual_pool[q]),np.asarray(pools[arm][q])):raise ValueError('Dense pool replay differs')
            train=[r for r in records if r['qid'] in set(p['scales'][scale])];train_pred,_=infer(model,train,text,cfg)
            truth={r['qid']:old_entries[r['qid']] for r in train}
            train_metrics[arm]={d:dict(eval_mr_brief(v,truth)) for d,v in train_pred.items()}
            stored=read(out/(arm.replace('/','_')+'_train_predictions.json'))
            for d,pr in train_pred.items():
                for q in pr:
                    if not np.array_equal(np.asarray(pr[q]),np.asarray(stored[d][str(q)])):raise ValueError('Train prediction replay differs')
        stats=evaluate_saved(out,p,records,all_predictions,pools)
    if any(stats[k]!=saved[k] for k in stats):raise ValueError('Metrics or factorial contrasts differ')
    if train_metrics!=saved['train_metrics']:raise ValueError('Train metric replay differs')
    result={'passed':True,'models_reloaded':4,'validation_queries':len(val),'prediction_files':8,
            'all_metrics_and_factorial_intervals_rebuilt':True,'train_targets_predictions_metrics_rebuilt':True,
            'new_training_updates':0,'new_text_encodings':0,
            'new_image_encodings':0,'elapsed_sec':time.perf_counter()-started}
    write_json_atomic(out/('replay_'+new_job_id()+'.json'),result);print(json.dumps(result,indent=2));return result


def mock(out):
    from scripts.make_synthetic_fixture import build_video
    from queryseek.ingest.ffprobe import probe
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False);video=build_video(out/'fixture',duration=12.)
    duration=float(probe(video)['format']['duration']);rng=np.random.default_rng(12)
    records=[];text={};targets={}
    entry={'relevant_windows':[[2.,8.]],'relevant_clip_ids':[1,2,3],'saliency_scores':[[3,3,3]]*3}
    for i in range(4):
        r={'qid':i,'source':f'synthetic{i}','times':np.arange(12,dtype=float),'images':rng.normal(size=(12,8)).astype(np.float32),
           'duration_sec':duration};records.append(r);text[i]=rng.normal(size=(3+i,8)).astype(np.float32)
        targets[i]=training_target(r['times'],duration,entry)
    torch.set_num_threads(1);torch.manual_seed(1);model=make_network({'hidden':16,'heads':4,'layers':1,'ffn':32},8)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3);criterion=make_criterion()
    plan=sampling_plan(records,targets,4,2,12)
    for batch in plan:
        rr=[records[q] for q in batch['qids']];x=collate_inputs(rr,text);y=collate_targets(rr,targets,None,batch['positive_positions'])
        optimizer.zero_grad(set_to_none=True);loss,parts=total_loss(criterion,model(**x),y);loss.backward();optimizer.step()
    pred,_=infer(model,records,text,{'dense_decoder':{'top_k':5,'nms_iou':.7}})
    result={'mock':True,'steps':4,'five_losses_finite':all(bool(torch.isfinite(v)) for v in parts.values()),
            'generated_fixture':str(video),'predictions':pred,'scope':'synthetic mechanics, not real accuracy or independent video sources'}
    write_json_atomic(out/'metrics.json',result);print('OUTPUT='+str(out));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('freeze','run','replay','mock'))
    parser.add_argument('--config',default='configs/token_localizer_v1.json');parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.phase=='freeze':freeze(read(ROOT/args.config),ROOT/'outputs/token_localizer'/new_job_id())
    elif args.phase=='mock':mock(ROOT/'outputs/token_localizer_mock'/new_job_id())
    else:
        if args.run is None:parser.error('--run required')
        out=checked_output(args.run)
        if args.phase=='replay':replay(out)
        else:
            try:run(out)
            except Exception as exc:
                write_json_atomic(out/'status.json',{'phase':'failed','error_type':type(exc).__name__});raise
