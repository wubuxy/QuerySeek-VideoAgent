"""Frozen mask/relative-event loss repair; reuse immutable training machinery."""
from __future__ import annotations
import argparse,copy,csv,json,shutil,sys,time
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts import token_localizer_study as base
from scripts import token_training_budget as budget
from scripts.localizer_study import read,rows
from scripts.sampling_probe import checked_output
from scripts.analyze_localizer_study import per_query,paired_clusters,candidate_diagnostic
from queryseek.retrieval.masked_event_localizer import MaskedLocalizer,criterion
from queryseek.retrieval.token_localizer import collate_inputs,collate_targets,total_loss,training_target
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.tools.export import new_job_id


class SanityModel(MaskedLocalizer):
    def train(self,mode=True):return super().train(False)


@contextmanager
def trainer_bindings(mode,weights=None,rng=None,sanity=False):
    """Scoped factory injection preserves the historical source and fit loop.

    Runs are sequential, in a dedicated process. No cross-agent/thread mutation.
    All injected behavior is snapshotted with the protocol and restored on exit.
    """
    old_network,old_criterion=base.make_network,base.make_criterion
    def network(spec):
        model=(SanityModel if sanity else MaskedLocalizer)(spec)
        if weights is not None:model.core.load_state_dict(weights,strict=True)
        if rng is not None:torch.set_rng_state(rng)
        return model
    base.make_network=network;base.make_criterion=lambda:criterion(mode)
    try:yield
    finally:base.make_network=old_network;base.make_criterion=old_criterion


def settings(original,cfg,sanity=False):
    p=copy.deepcopy(original);c=p['config']
    for k in ('steps','warmup_steps','decay_steps','learning_rate','checkpoint_every','log_every','torch_threads','wall_cap_sec'):
        c[k]=cfg[k]
    if sanity:
        c.update(steps=cfg['sanity']['steps'],batch_size=8,learning_rate=cfg['sanity']['learning_rate'],warmup_steps=1,
                 decay_steps=[],checkpoint_every=100,capacities={'large':cfg['sanity']['spec']},wall_cap_sec=300)
    return p


def sanity_selection(records,entries,n):
    result=[];seen=set()
    for short in (True,False):
        selected=[]
        for r in sorted(records,key=lambda r:r['qid']):
            lengths=[b-a for a,b in entries[r['qid']]['relevant_windows']]
            eligible=max(lengths)<=10 if short else min(lengths)>30
            if eligible and r['source'] not in seen:
                selected.append(r);seen.add(r['source'])
                if len(selected)==n:break
        if len(selected)!=n:raise ValueError('Insufficient disjoint sanity sources')
        result.extend(selected)
    return result


def freeze(cfg,out):
    parent=checked_output(ROOT/cfg['parent_run']);pp=budget.verify(parent)
    if read(parent/'status.json')['phase']!='complete':raise ValueError('Completed parent required')
    for name,sha in read(parent/'metrics.json')['artifact_sha256'].items():
        if sha256_file(parent/name)!=sha:raise ValueError('Previous training artifact changed')
    original=ROOT/pp['config']['parent_run'];records,text,targets=budget.load_inputs(original)
    train=[r for r in records if r['split']=='train'];old=pp['parent_protocol']
    entries={r['entry']['qid']:r['entry'] for r in rows(ROOT/old['config']['study_run']/'feature_records.jsonl') if r['split']=='train'}
    sanity=sanity_selection(train,entries,cfg['sanity']['queries_per_duration_bin'])
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    plan=base.sampling_plan(train,targets,cfg['steps'],old['config']['batch_size'],cfg['sampling_seed'])
    write_json_atomic(out/'sampling_plan_256.json',plan)
    sources={**pp['source_sha256']}
    for name in ['scripts/event_localizer_repair.py','queryseek/retrieval/masked_event_localizer.py',
                 'configs/event_localizer_repair_v1.json','tests/test_masked_event_localizer.py','tests/test_event_localizer_repair.py']:
        sources[name]=sha256_file(ROOT/name)
    for name in sources:
        dst=out/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    shutil.copyfile(ROOT/'docs/EVAL_EVENT_LOCALIZER_REPAIR.md',out/'protocol_notes_frozen.md')
    preserved={str(f.relative_to(ROOT)):sha256_file(f) for f in parent.rglob('*') if f.is_file()}
    for f in (ROOT/cfg['diagnosis_run']).rglob('*'):
        if f.is_file():preserved[str(f.relative_to(ROOT))]=sha256_file(f)
    p={'config':cfg,'parent_protocol':pp,'source_sha256':sources,'preserved_sha256':preserved,
       'checkpoint_sha256':sha256_file(parent/cfg['checkpoint']),'plan_sha256':sha256_file(out/'sampling_plan_256.json'),
       'notes_sha256':sha256_file(out/'protocol_notes_frozen.md'),'sanity_qids':[r['qid'] for r in sanity],
       'source_overlap':0,'pipeline_version':'2','validation_targets_enter_training':False,'fresh_query_label_access':False}
    write_json_atomic(out/'protocol.json',p);write_json_atomic(out/'status.json',{'phase':'frozen'})
    print('OUTPUT='+str(out),flush=True);return out


def verify(out):
    p=read(out/'protocol.json');budget.verify(ROOT/p['config']['parent_run'])
    for name,sha in p['preserved_sha256'].items():
        if sha256_file(ROOT/name)!=sha:raise ValueError('Historical artifact changed: '+name)
    for name,sha in p['source_sha256'].items():
        if sha256_file(ROOT/name)!=sha or sha256_file(out/'source_snapshot'/name)!=sha:raise ValueError('Frozen source changed: '+name)
    if sha256_file(out/'sampling_plan_256.json')!=p['plan_sha256'] or sha256_file(out/'protocol_notes_frozen.md')!=p['notes_sha256']:
        raise ValueError('Plan or notes changed')
    return p


def common_losses(model,records,text,targets,plan):
    return {mode:budget_losses(model,records,text,targets,plan,mode) for mode in ('mask_only','event_relative')}


def budget_losses(model,records,text,targets,plan,mode):
    model.eval();c=criterion(mode);lookup={r['qid']:r for r in records};values=[]
    with torch.inference_mode():
        for batch in plan[:4]:
            rr=[lookup[q] for q in batch['qids']];y=collate_targets(rr,targets,None,batch['positive_positions'])
            loss,parts=total_loss(c,model(**collate_inputs(rr,text)),y)
            values.append({'total':float(loss),**{k:float(v) for k,v in parts.items()}})
    return {k:float(np.mean([v[k] for v in values])) for k in values[0]}


def sanity(out,p,records,text,targets,entries):
    cfg=p['config'];cp=settings(p['parent_protocol']['parent_protocol'],cfg,True)
    rr=[r for q in p['sanity_qids'] for r in records if r['qid']==q]
    selected=[int(targets[r['qid']]['timestamp_window'].nonzero().flatten()[len(targets[r['qid']]['timestamp_window'].nonzero())//2]) for r in rr]
    plan=[{'qids':[r['qid'] for r in rr],'positive_positions':selected}]*cfg['sanity']['steps']
    result={}
    for mode in cfg['modes']:
        child=out/'sanity'/mode;child.mkdir(parents=True,exist_ok=True);write_json_atomic(child/'sampling_plan_256.json',plan)
        with trainer_bindings(mode,sanity=True):fit=base.fit_arm(child,cp,rr,text,targets,'256/large')
        model=SanityModel(cfg['sanity']['spec']);model.load_state_dict(torch.load(child/f'models/256/large/step_{cfg["sanity"]["steps"]:04d}.pt',weights_only=True)['model'])
        pred,pool=base.infer(model,rr,text,cp['config']);gt={r['qid']:entries[r['qid']] for r in rr}
        result[mode]={'fit':fit,'train_metrics':{d:dict(eval_mr_brief(v,gt)) for d,v in pred.items()},
                      'diagnostic':candidate_diagnostic(pool,pred['dense'],gt),'predictions':pred}
    result['scope']='eight selected training queries, dropout disabled; memorization diagnostic, not retrieval accuracy or independent generalization'
    write_json_atomic(out/'sanity_results.json',result);return result


def evaluate(out,p,records,text,targets,replay=False):
    cfg=p['config'];pp=p['parent_protocol'];original=ROOT/pp['config']['parent_run'];parent=ROOT/cfg['parent_run']
    old=pp['parent_protocol'];entries={r['entry']['qid']:r['entry'] for r in rows(ROOT/old['config']['study_run']/'feature_records.jsonl')}
    splits={s:[r for r in records if r['split']==s] for s in ('train','validation')}
    baseline={int(q):v for q,v in read(original/'e4_predictions.json').items()}
    old_predictions={d:{int(q):v for q,v in read(parent/f'predictions/delayed/1600/validation_{d}.json').items()} for d in ('dense','curve')}
    result={'models':{},'e4':dict(eval_mr_brief(baseline,{q:entries[q] for q in baseline})),'contrasts':{}}
    all_pred={};plan=read(out/'sampling_plan_256.json')
    for mode in cfg['modes']:
        model=MaskedLocalizer(old['config']['capacities']['large']);path=out/mode/f'models/256/large/step_{cfg["steps"]:04d}.pt'
        model.load_state_dict(torch.load(path,weights_only=True)['model']);stats={};all_pred[mode]={}
        for split,rr in splits.items():
            pred,pool=base.infer(model,rr,text,old['config']);all_pred[mode][split]=pred;gt={r['qid']:entries[r['qid']] for r in rr}
            if any(set(v)!=set(gt) for v in pred.values()):raise ValueError('Prediction denominator differs')
            stats[split]={'metrics':{d:dict(eval_mr_brief(v,gt)) for d,v in pred.items()},'diagnostic':candidate_diagnostic(pool,pred['dense'],gt)}
            for name,data in {**pred,'pool':pool}.items():
                file=out/'predictions'/mode/f'{split}_{name}.json';normalized={str(q):[list(w) for w in ws] for q,ws in data.items()}
                if replay:
                    if read(file)!=normalized:raise ValueError('Prediction or pool replay differs')
                else:write_json_atomic(file,data)
        stats['common_fixed_train_losses']=common_losses(model,splits['train'],text,targets,plan);result['models'][mode]=stats
    gt={r['qid']:entries[r['qid']] for r in splits['validation']}
    for d in ('dense','curve'):
        pairs={'primary_event_relative_minus_mask_only':(all_pred['mask_only']['validation'][d],all_pred['event_relative']['validation'][d])}
        for mode in cfg['modes']:
            pairs[mode+'_minus_parent']=(old_predictions[d],all_pred[mode]['validation'][d])
            pairs[mode+'_minus_e4']=(baseline,all_pred[mode]['validation'][d])
        for name,(first,second) in pairs.items():
            st=paired_clusters(per_query(first,gt),per_query(second,gt),**cfg['bootstrap'])
            st['scope']='fixed repair endpoints on previously observed internal validation; objective bundle, no multiplicity adjustment or independent generalization'
            result['contrasts'][name+'/'+d]=st
    return result


def run(out,replay=False):
    started=time.perf_counter();p=verify(out);cfg=p['config'];pp=p['parent_protocol'];parent=ROOT/cfg['parent_run'];original=ROOT/pp['config']['parent_run']
    if not replay and read(out/'status.json')['phase']=='complete':raise ValueError('Completed; use replay')
    torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=cfg['torch_threads']):
        records,text,targets=budget.load_inputs(original);old=pp['parent_protocol']
        seed=torch.load(parent/cfg['checkpoint'],weights_only=True)
        entries={r['entry']['qid']:r['entry'] for r in rows(ROOT/old['config']['study_run']/'feature_records.jsonl') if r['split']=='train'}
        if replay:
            saved=read(out/'metrics.json')
            for name,sha in saved['artifact_sha256'].items():
                if sha256_file(out/name)!=sha:raise ValueError('Completed artifact changed: '+name)
            fits=saved['fits']
        else:
            write_json_atomic(out/'status.json',{'phase':'train_only_sanity','validation_evaluated':False})
            sanity(out,p,records,text,targets,entries)
            write_json_atomic(out/'status.json',{'phase':'fine_tuning','validation_evaluated':False});fits={}
            for mode in cfg['modes']:
                child=out/mode;child.mkdir(parents=True,exist_ok=True);plan=read(out/'sampling_plan_256.json')
                file=child/'sampling_plan_256.json'
                if file.exists() and read(file)!=plan:raise ValueError('Child sampling changed')
                if not file.exists():write_json_atomic(file,plan)
                cp=settings(old,cfg);cp['config']['wall_cap_sec']=cfg['wall_cap_sec']-sum(v['training_sec'] for v in fits.values())
                with trainer_bindings(mode,seed['model'],seed['torch_rng']):
                    fits[mode]=base.fit_arm(child,cp,records,text,targets,'256/large')
                if fits[mode]['sampling_plan_sha256']!=p['plan_sha256']:raise ValueError('Sampling hash changed')
            if fits['mask_only']['initial_sha256']!=fits['event_relative']['initial_sha256']:raise ValueError('Warm-start weights differ')
            write_json_atomic(out/'status.json',{'phase':'evaluating','all_endpoints_saved':True})
        result=evaluate(out,p,records,text,targets,replay)
    verify(out)
    if replay:
        if any(saved[k]!=v for k,v in result.items()):raise ValueError('Loss/metric/contrast replay differs')
        report={'passed':True,'models_reloaded':2,'all_predictions_pools_common_losses_metrics_contrasts_rebuilt':True,
                'new_training_updates':0,'new_image_encodings':0,'new_text_encodings':0,'elapsed_sec':time.perf_counter()-started}
        write_json_atomic(out/('replay_'+new_job_id()+'.json'),report);print(json.dumps(report),flush=True);return report
    result.update(fits=fits,new_training_updates=cfg['steps']*len(cfg['modes']),new_training_sec=sum(v['training_sec'] for v in fits.values()),
                  sanity_training_sec=sum(v['fit']['training_sec'] for k,v in read(out/'sanity_results.json').items() if k!='scope'),
                  elapsed_sec=time.perf_counter()-started,mock=False,independent=False,new_text_encodings=0,new_image_encodings=0,vlm_calls=0,fresh_query_label_access=False)
    result['artifact_sha256']={str(f.relative_to(out)):sha256_file(f) for f in out.rglob('*') if f.is_file() and
                              ('predictions' in f.parts or f.suffix=='.pt' or f.name in ('fit.json','history.json','sampling_plan_256.json','sanity_results.json'))}
    write_json_atomic(out/'metrics.json',result);write_json_atomic(out/'status.json',{'phase':'complete'})
    print('RESULT='+json.dumps(result['models']),flush=True);print('OUTPUT='+str(out),flush=True);return result


def mock(out):
    from scripts.make_synthetic_fixture import build_video
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False);video=build_video(out/'fixture',duration=12.)
    torch.set_num_threads(1);rng=np.random.default_rng(8);records=[];text={};targets={}
    for q in range(4):
        n=8+q;r={'qid':q,'source':str(q),'times':np.arange(n,dtype=float),'duration_sec':float(n),'images':rng.normal(size=(n,512)).astype(np.float32)*.1}
        records.append(r);text[q]=rng.normal(size=(4,512)).astype(np.float32)*.1
        targets[q]=training_target(r['times'],r['duration_sec'],{'relevant_windows':[[2.,4.]],'relevant_clip_ids':[1],'saliency_scores':[[3,3,3]]})
    cfg=read(ROOT/'configs/token_localizer_v1.json');cfg.update(steps=4,batch_size=2,log_every=1,checkpoint_every=2,warmup_steps=1,decay_steps=[2,3],capacities={'large':{'hidden':16,'heads':4,'layers':1,'ffn':32}})
    p={'config':cfg,'scale_summary':{'256':{'queries':4,'sources':4}}};fits={};plan=base.sampling_plan(records,targets,4,2,12)
    for mode in ('mask_only','event_relative'):
        child=out/mode;child.mkdir();write_json_atomic(child/'sampling_plan_256.json',plan)
        with trainer_bindings(mode):fits[mode]=base.fit_arm(child,p,records,text,targets,'256/large')
    assert fits['mask_only']['initial_sha256']==fits['event_relative']['initial_sha256']
    report={'mock':True,'generated_fixture':str(video),'two_modes_train_and_reload':True,'scope':'synthetic mechanisms, not real accuracy'}
    write_json_atomic(out/'metrics.json',report);print('OUTPUT='+str(out),flush=True);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['freeze','run','replay','mock'])
    parser.add_argument('--config',default='configs/event_localizer_repair_v1.json');parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.phase=='freeze':freeze(read(ROOT/args.config),ROOT/'outputs/event_localizer_repair'/new_job_id())
    elif args.phase=='mock':mock(ROOT/'outputs/event_localizer_repair_mock'/new_job_id())
    else:
        if args.run is None:parser.error('--run required')
        out=checked_output(args.run)
        try:run(out,args.phase=='replay')
        except Exception as exc:
            if args.phase=='run':write_json_atomic(out/'status.json',{'phase':'failed','error_type':type(exc).__name__})
            raise
