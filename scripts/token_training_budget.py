"""Frozen continuation control for token-localizer training budget/schedule.

No feature encoding, old checkpoint mutations, validation selection, or new
network/loss implementation. Native frozen trainer is reused via child runs.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts import token_localizer_study as base
from scripts.localizer_study import read,rows
from scripts.sampling_probe import checked_output
from scripts.analyze_localizer_study import per_query,paired_clusters,candidate_diagnostic
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.tools.export import new_job_id
from queryseek.retrieval.token_localizer import make_network,make_criterion,collate_inputs,collate_targets,total_loss
from queryseek.eval.qvhighlights import eval_mr_brief


def child_protocol(parent,cfg,name):
    p=copy.deepcopy(parent);c=p['config'];arm=cfg['curves'][name]
    c.update(steps=cfg['steps'],decay_steps=arm['decay_steps'],wall_cap_sec=cfg['wall_cap_sec'],
             torch_threads=cfg['torch_threads'],order=[cfg['scale']+'/'+cfg['capacity']])
    return p


def check_prefix(old_cfg,new_cfg,old_plan,new_plan,step):
    if old_plan[:step]!=new_plan[:step] or len(old_plan)<step or len(new_plan)<step:
        raise ValueError('Sampling prefix differs')
    mutable={'steps','decay_steps','wall_cap_sec','torch_threads','order'}
    if {k:v for k,v in old_cfg.items() if k not in mutable}!={k:v for k,v in new_cfg.items() if k not in mutable}:
        raise ValueError('Non-schedule training setting changed')
    if any(base.learning_rate(old_cfg,i)!=base.learning_rate(new_cfg,i) for i in range(step)):
        raise ValueError('Learning-rate prefix differs')


def load_inputs(parent):
    records=base.load_inference_records(parent);pointer=read(parent/'text_cache.json');cache=Path(pointer['folder'])
    if (sha256_file(cache/'metadata.json')!=pointer['metadata_sha256'] or
            sha256_file(cache/'features.npz')!=pointer['features_sha256']):
        raise ValueError('Frozen text feature changed')
    with np.load(cache/'features.npz',allow_pickle=False) as f:
        text={r['qid']:f[f'tokens_{r["qid"]}'] for r in records}
    targets=torch.load(parent/'train_targets.pt',weights_only=True)
    if set(targets)!={r['qid'] for r in records if r['split']=='train'}:
        raise ValueError('Validation qid in training targets')
    return records,text,targets


def freeze(cfg,out):
    parent=checked_output(ROOT/cfg['parent_run']);p=base.verify_protocol(parent)
    if read(parent/'status.json')['phase']!='complete':raise ValueError('Completed parent required')
    for name,sha in read(parent/'metrics.json')['artifact_sha256'].items():
        if sha256_file(parent/name)!=sha:raise ValueError('Parent fitted artifact changed')
    if cfg['order']!=['early','delayed'] or cfg['scale']!='256' or cfg['capacity']!='large':raise ValueError('Fixed two-arm design required')
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    records,_,targets=load_inputs(parent)
    train=[r for r in records if r['qid'] in set(p['scales'][cfg['scale']])]
    plan=base.sampling_plan(train,targets,cfg['steps'],p['config']['batch_size'],p['config']['seed'])
    old_plan=read(parent/f'sampling_plan_{cfg["scale"]}.json')
    if plan[:len(old_plan)]!=old_plan:raise ValueError('Original sampling prefix changed')
    write_json_atomic(out/'sampling_plan.json',plan)
    for name in cfg['order']:
        cp=child_protocol(p,cfg,name)
        check_prefix(p['config'],cp['config'],old_plan,plan,cfg['curves'][name]['resume_step'])
    sources={**p['source_sha256']}
    for name in ['scripts/token_training_budget.py','configs/token_training_budget_v1.json','tests/test_token_training_budget.py']:
        sources[name]=sha256_file(ROOT/name)
    for name,sha in sources.items():
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        if sha256_file(target)!=sha:raise ValueError('Snapshot copy differs')
    shutil.copyfile(ROOT/'docs/EVAL_TOKEN_TRAINING_BUDGET.md',out/'protocol_notes_frozen.md')
    historical={str(f.relative_to(ROOT)):sha256_file(f) for f in parent.rglob('*') if f.is_file()}
    result={'config':cfg,'parent_protocol':p,'parent_artifacts_sha256':historical,'source_sha256':sources,
            'plan_sha256':sha256_file(out/'sampling_plan.json'),'notes_sha256':sha256_file(out/'protocol_notes_frozen.md'),
            'new_training_updates_planned':sum(cfg['steps']-cfg['curves'][n]['resume_step'] for n in cfg['order']),
            'validation_targets_enter_training':False,'new_image_encodings':0,'new_text_encodings':0,
            'vlm_calls':0,'fresh_query_label_access':False,'independent':False,'paper_reproduction':False}
    write_json_atomic(out/'protocol.json',result);write_json_atomic(out/'status.json',{'phase':'frozen'})
    print('OUTPUT='+str(out),flush=True);return out


def verify(out):
    p=read(out/'protocol.json');cfg=p['config'];base.verify_protocol(ROOT/cfg['parent_run'])
    for name,sha in p['parent_artifacts_sha256'].items():
        if sha256_file(ROOT/name)!=sha:raise ValueError('Parent artifact changed: '+name)
    for name,sha in p['source_sha256'].items():
        if sha256_file(ROOT/name)!=sha or sha256_file(out/'source_snapshot'/name)!=sha:raise ValueError('Frozen source changed: '+name)
    if sha256_file(out/'sampling_plan.json')!=p['plan_sha256'] or sha256_file(out/'protocol_notes_frozen.md')!=p['notes_sha256']:
        raise ValueError('Frozen plan/notes changed')
    return p


def seed_child(parent,out,p,name):
    cfg=p['config'];cp=child_protocol(p['parent_protocol'],cfg,name);root=out/'arms'/name
    scale=cfg['scale'];capacity=cfg['capacity'];step=cfg['curves'][name]['resume_step']
    plan=read(out/'sampling_plan.json');old_plan=read(parent/f'sampling_plan_{scale}.json')
    check_prefix(p['parent_protocol']['config'],cp['config'],old_plan,plan,step)
    root.mkdir(parents=True,exist_ok=True)
    plan_path=root/f'sampling_plan_{scale}.json'
    if plan_path.exists():
        if read(plan_path)!=plan:raise ValueError('Child plan changed')
    else:write_json_atomic(plan_path,plan)
    folder=root/'models'/scale/capacity;folder.mkdir(parents=True,exist_ok=True)
    source=parent/f'models/{scale}/{capacity}/step_{step:04d}.pt'
    target=folder/source.name
    seed={'source':str(source.resolve()),'checkpoint_sha256':sha256_file(source),'step':step,
          'sampling_prefix_exact':True,'learning_rate_prefix_exact':True}
    if (root/'seed.json').exists():
        if read(root/'seed.json')!=seed or sha256_file(target)!=seed['checkpoint_sha256']:raise ValueError('Seed checkpoint changed')
    else:
        if (folder/'progress.json').exists():raise ValueError('Unregistered child checkpoint')
        shutil.copyfile(source,target)
        write_json_atomic(root/'seed.json',seed)
        write_json_atomic(folder/'progress.json',{'checkpoint_step':step,'checkpoint_sha256':seed['checkpoint_sha256']})
    return root,cp


def fixed_losses(model,records,text,targets,plan,batches):
    model.eval();criterion=make_criterion();lookup={r['qid']:r for r in records};parts=[]
    with torch.inference_mode():
        for batch in plan[:batches]:
            rr=[lookup[q] for q in batch['qids']]
            y=collate_targets(rr,targets,None,batch['positive_positions'])
            loss,components=total_loss(criterion,model(**collate_inputs(rr,text)),y)
            parts.append({'total':float(loss),**{k:float(v) for k,v in components.items()}})
    return {k:float(np.mean([v[k] for v in parts])) for k in parts[0]}


def snapshot_path(out,p,name,step):
    cfg=p['config'];rel=Path('models')/cfg['scale']/cfg['capacity']/f'step_{step:04d}.pt'
    return ROOT/cfg['parent_run']/rel if step<=cfg['curves'][name]['resume_step'] else out/'arms'/name/rel


def summarize(out,p,predictions,diagnostics,losses,truth):
    cfg=p['config'];parent=ROOT/cfg['parent_run'];baseline={int(q):v for q,v in read(parent/'e4_predictions.json').items()}
    val_gt={q:r for q,r in truth.items() if q in baseline};train_gt={q:r for q,r in truth.items() if q in set(p['parent_protocol']['scales'][cfg['scale']])}
    stats={'e4':dict(eval_mr_brief(baseline,val_gt)),'snapshots':{},'contrasts':{}}
    for key,pr in predictions.items():
        if set(pr['validation']['dense'])!=set(val_gt) or set(pr['train']['dense'])!=set(train_gt):raise ValueError('Prediction denominator changed')
        stats['snapshots'][key]={'validation':{d:dict(eval_mr_brief(v,val_gt)) for d,v in pr['validation'].items()},
                                 'train':{d:dict(eval_mr_brief(v,train_gt)) for d,v in pr['train'].items()},
                                 'dense_diagnostic':diagnostics[key],'fixed_train_batch_losses':losses[key]}
    comparisons={'primary_delayed1600_minus_original800':('early/800','delayed/1600'),
                 'updates_only_early1600_minus_early800':('early/800','early/1600'),
                 'schedule_at1600_delayed_minus_early':('early/1600','delayed/1600'),
                 'schedule_at800_delayed_minus_early':('early/800','delayed/800')}
    for name,(first,second) in comparisons.items():
        for d in ('dense','curve'):
            contrast=paired_clusters(per_query(predictions[first]['validation'][d],val_gt),per_query(predictions[second]['validation'][d],val_gt),**cfg['bootstrap'])
            contrast['scope']='fixed endpoints on previously observed internal validation; no checkpoint selection or multiplicity adjustment; joint primary intervention'
            stats['contrasts'][name+'/'+d]=contrast
    for key in ('early/1600','delayed/1600'):
        for d in ('dense','curve'):
            contrast=paired_clusters(per_query(baseline,val_gt),per_query(predictions[key]['validation'][d],val_gt),**cfg['bootstrap'])
            contrast['scope']='conditional internal validation; no multiplicity adjustment'
            stats['contrasts'][key+'/'+d+'_minus_e4']=contrast
    return stats


def evaluate(out,p,records,text,targets,replay=False):
    cfg=p['config'];parent=ROOT/cfg['parent_run'];train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='validation']
    truth={r['entry']['qid']:r['entry'] for r in rows(ROOT/p['parent_protocol']['config']['study_run']/'feature_records.jsonl')}
    predictions={};diagnostics={};losses={};memo={};plan=read(out/'sampling_plan.json');snapshots={}
    parent_predictions={split:{d:{int(q):v for q,v in pred.items()} for d,pred in
        ({d:read(parent/f'256_large_{d}_predictions.json') for d in ('dense','curve')} if split=='validation'
         else read(parent/'256_large_train_predictions.json')).items()} for split in ('train','validation')}
    for name in cfg['order']:
        for step in cfg['evaluation_steps']:
            key=f'{name}/{step}';path=snapshot_path(out,p,name,step);sha=sha256_file(path);snapshots[key]=sha
            if sha not in memo:
                model=make_network(p['parent_protocol']['config']['capacities'][cfg['capacity']]);model.load_state_dict(torch.load(path,weights_only=True)['model'])
                vpred,pool=base.infer(model,val,text,p['parent_protocol']['config']);tpred,_=base.infer(model,train,text,p['parent_protocol']['config'])
                diag=candidate_diagnostic(pool,vpred['dense'],{r['qid']:truth[r['qid']] for r in val})
                ll=fixed_losses(model,train,text,targets,plan,cfg['fixed_loss_batches'])
                memo[sha]=({'validation':vpred,'train':tpred},diag,ll,pool)
            pr,diag,ll,pool=memo[sha];predictions[key]=pr;diagnostics[key]=diag;losses[key]=ll
            folder=out/'predictions'/name/str(step)
            for split in ('train','validation'):
                for d,pred in pr[split].items():
                    file=folder/f'{split}_{d}.json'
                    if replay:
                        if read(file)!={str(q):[list(w) for w in ws] for q,ws in pred.items()}:raise ValueError('Snapshot prediction replay differs')
                    else:write_json_atomic(file,pred)
            file=folder/'validation_pool.json'
            if replay:
                if read(file)!={str(q):[list(w) for w in ws] for q,ws in pool.items()}:raise ValueError('Pool replay differs')
            else:write_json_atomic(file,pool)
            if key=='early/800':
                for split in ('train','validation'):
                    for d,prr in pr[split].items():
                        for q,ws in prr.items():
                            if not np.array_equal(np.asarray(ws),np.asarray(parent_predictions[split][d][q])):raise ValueError('Original 800-step prediction changed')
            print('EVAL',key,'replayed' if replay else 'saved',flush=True)
    result=summarize(out,p,predictions,diagnostics,losses,truth);result['snapshot_sha256']=snapshots
    if not replay:
        with (out/'learning_curves.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle);writer.writerow(['curve','step','decoder','train_mAP','validation_mAP','validation_R1_05','validation_R1_07','validation_short_mAP','fixed_train_batch_loss'])
            for key,st in result['snapshots'].items():
                for d in ('dense','curve'):
                    v=st['validation'][d];writer.writerow([*key.split('/'),d,st['train'][d]['MR-full-mAP'],v['MR-full-mAP'],v['MR-full-R1@0.5'],v['MR-full-R1@0.7'],v['MR-short-mAP'],st['fixed_train_batch_losses']['total']])
    return result


def run(out,replay=False):
    started=time.perf_counter();p=verify(out);cfg=p['config'];parent=ROOT/cfg['parent_run']
    if not replay and read(out/'status.json')['phase']=='complete':raise ValueError('Completed; use replay')
    torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=cfg['torch_threads']):
        records,text,targets=load_inputs(parent)
        fits={}
        if not replay:
            write_json_atomic(out/'status.json',{'phase':'training','validation_evaluated':False})
            for name in cfg['order']:
                root,cp=seed_child(parent,out,p,name)
                seed_sec=torch.load(snapshot_path(out,p,name,cfg['curves'][name]['resume_step']),weights_only=True)['training_sec']
                completed_new=sum(fit['training_sec']-torch.load(snapshot_path(out,p,n,cfg['curves'][n]['resume_step']),weights_only=True)['training_sec'] for n,fit in fits.items())
                cp['config']['wall_cap_sec']=cfg['wall_cap_sec']-completed_new+seed_sec
                fits[name]=base.fit_arm(root,cp,records,text,targets,cfg['scale']+'/'+cfg['capacity'])
            write_json_atomic(out/'status.json',{'phase':'evaluating','all_endpoints_saved':True})
        else:
            saved=read(out/'metrics.json')
            for name,sha in saved['artifact_sha256'].items():
                if sha256_file(out/name)!=sha:raise ValueError('Completed artifact changed: '+name)
            fits=saved['fits']
        result=evaluate(out,p,records,text,targets,replay)
    verify(out)
    if replay:
        if any(saved[k]!=v for k,v in result.items()):raise ValueError('Metrics or contrasts replay differs')
        report={'passed':True,'snapshot_labels':len(result['snapshots']),'unique_models_reloaded':len(set(result['snapshot_sha256'].values())),
                'training_and_validation_predictions_rebuilt':True,'all_losses_metrics_contrasts_rebuilt':True,
                'new_training_updates':0,'new_text_encodings':0,'new_image_encodings':0,'elapsed_sec':time.perf_counter()-started}
        write_json_atomic(out/('replay_'+new_job_id()+'.json'),report);print(json.dumps(report),flush=True);return report
    new_sec=sum(fit['training_sec']-torch.load(snapshot_path(out,p,name,cfg['curves'][name]['resume_step']),weights_only=True)['training_sec'] for name,fit in fits.items())
    result.update(fits=fits,new_training_sec=new_sec,new_training_updates=p['new_training_updates_planned'],elapsed_sec=time.perf_counter()-started,
                  mock=False,independent=False,new_text_encodings=0,new_image_encodings=0,vlm_calls=0,fresh_query_label_access=False)
    result['artifact_sha256']={str(f.relative_to(out)):sha256_file(f) for f in out.rglob('*') if f.is_file() and
                              ('predictions' in f.parts or f.suffix=='.pt' or f.name in ('fit.json','seed.json','history.json','learning_curves.csv','sampling_plan.json'))}
    write_json_atomic(out/'metrics.json',result);write_json_atomic(out/'status.json',{'phase':'complete'})
    print('RESULT='+json.dumps({'endpoints':{k:v for k,v in result['snapshots'].items() if k.endswith('/1600')},'contrasts':result['contrasts'],'new_training_sec':new_sec}),flush=True)
    print('OUTPUT='+str(out),flush=True);return result


def mock(out):
    from scripts.make_synthetic_fixture import build_video
    from queryseek.retrieval.token_localizer import training_target
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False);video=build_video(out/'fixture',duration=12.)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(12);times=np.arange(12,dtype=float)
    records=[{'qid':i,'source':f'synthetic{i}','times':times,'duration_sec':12.,
              'images':rng.normal(size=(12,512)).astype(np.float32)*.1} for i in range(4)]
    text={r['qid']:rng.normal(size=(4,512)).astype(np.float32)*.1 for r in records}
    entry={'relevant_windows':[[2.,8.]],'relevant_clip_ids':[1,2,3],'saliency_scores':[[3,3,3]]*3}
    targets={r['qid']:training_target(times,12.,entry) for r in records}
    cfg=read(ROOT/'configs/token_localizer_v1.json')
    cfg.update(steps=4,batch_size=2,checkpoint_every=2,log_every=1,warmup_steps=1,
               decay_steps=[2,3],capacities={'large':{'hidden':16,'heads':4,'layers':1,'ffn':32}})
    parent=out/'parent';parent.mkdir();plan=base.sampling_plan(records,targets,8,2,cfg['seed'])
    write_json_atomic(parent/'sampling_plan_256.json',plan[:4])
    pp={'config':cfg,'scale_summary':{'256':{'sources':4,'queries':4}},'scales':{'256':[0,1,2,3]}}
    base.fit_arm(parent,pp,records,text,targets,'256/large')
    frozen={'config':{'steps':8,'scale':'256','capacity':'large','torch_threads':1,'wall_cap_sec':60,
             'curves':{'early':{'resume_step':4,'decay_steps':[2,3]},'delayed':{'resume_step':2,'decay_steps':[4,6]}}},'parent_protocol':pp}
    write_json_atomic(out/'sampling_plan.json',plan);fits={}
    for name in ('early','delayed'):
        root,cp=seed_child(parent,out,frozen,name)
        fits[name]=base.fit_arm(root,cp,records,text,targets,'256/large')
    reference=out/'reference';reference.mkdir();write_json_atomic(reference/'sampling_plan_256.json',plan)
    direct=base.fit_arm(reference,child_protocol(pp,frozen['config'],'delayed'),records,text,targets,'256/large')
    assert direct['final_tensor_sha256']==fits['delayed']['final_tensor_sha256']
    model=make_network(cfg['capacities']['large']);model.load_state_dict(torch.load(out/'arms/delayed/models/256/large/step_0008.pt',weights_only=True)['model'])
    predictions,_=base.infer(model,records,text,cfg)
    loss_probe=fixed_losses(model,records,text,targets,plan,2)
    if loss_probe!=fixed_losses(model,records,text,targets,plan,2):raise ValueError('Fixed loss probe is not repeatable')
    result={'mock':True,'generated_fixture':str(video),'continuation_equals_uninterrupted':True,
            'new_training_updates':22,'continuation_updates':10,'endpoint_steps_each':8,'predictions':predictions,
            'fixed_eval_losses_repeatable':True,'fixed_eval_losses':loss_probe,
            'scope':'synthetic mechanics, not real accuracy or independent sources'}
    write_json_atomic(out/'metrics.json',result);print('OUTPUT='+str(out),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['freeze','run','replay','mock'])
    parser.add_argument('--config',default='configs/token_training_budget_v1.json');parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.phase=='freeze':freeze(read(ROOT/args.config),ROOT/'outputs/token_training_budget'/new_job_id())
    elif args.phase=='mock':mock(ROOT/'outputs/token_training_budget_mock'/new_job_id())
    else:
        if args.run is None:parser.error('--run required')
        out=checked_output(args.run)
        try:run(out,args.phase=='replay')
        except Exception as exc:
            if args.phase=='run':write_json_atomic(out/'status.json',{'phase':'failed','error_type':type(exc).__name__})
            raise
