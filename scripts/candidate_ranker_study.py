"""Frozen, source-group candidate-quality regression with sklearn; no encoder.

Candidate generation and feature extraction never receive GT. Only train labels
reach fit; all internal-validation predictions are saved before label evaluation.
All source folds are fixed at freeze, with no result-based model selection.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from queryseek import PIPELINE_VERSION
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.candidate_ranker import (FEATURE_NAMES,candidate_table,training_targets,
    source_equal_weights,model_inputs,rank_candidates,fit_ranker)
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.tools.export import new_job_id
from scripts.localizer_study import read,rows,verify,load_records
from scripts.localizer_query_controls import frozen_inventory
from scripts.univtg_input_transfer import verify_protocol as verify_transfer
from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output,optional_version
from scripts.analyze_localizer_study import per_query,paired_clusters,candidate_diagnostic


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def source_folds(items,n_splits):
    sources=[item['source'] for item in items];result=[]
    for train,test in GroupKFold(n_splits=n_splits).split(np.zeros(len(items)),groups=sources):
        if set(sources[i] for i in train)&set(sources[i] for i in test):raise ValueError('Source fold leakage')
        result.append({'train_qids':[items[i]['qid'] for i in train], 'test_qids':[items[i]['qid'] for i in test]})
    if sorted(q for f in result for q in f['test_qids'])!=sorted(item['qid'] for item in items):
        raise ValueError('OOF denominator changed')
    return result


def verify_inputs(out):
    p=read(out/'protocol.json')
    if 'protocol_notes_sha256' in p and sha256_file(out/'protocol_notes_frozen.md')!=p['protocol_notes_sha256']:
        raise ValueError('Frozen experiment notes changed')
    for path,sha in p.get('preserved_sha256',{}).items():
        if sha256_file(ROOT/path)!=sha:raise ValueError('Frozen historical input changed: '+path)
    for name,sha in p['source_sha256'].items():
        if sha256_file(ROOT/name)!=sha or sha256_file(out/'source_snapshot'/name)!=sha:
            raise ValueError('Frozen ranker source changed: '+name)
    return p


def freeze(cfg,out):
    if cfg['controls']!=['full','structure'] or cfg['model']['early_stopping'] is not False:
        raise ValueError('Fixed scorers and no checkpoint selection required')
    prior=checked_output(ROOT/cfg['input_transfer_run']);verify_transfer(prior)
    historical=read(prior/'historical_audit.json')
    for path,sha in historical['input_sha256'].items():
        if sha256_file(path)!=sha:raise ValueError('Historical control changed')
    study=checked_output(ROOT/cfg['study_run']);parent=verify(study)
    records=load_records(study,parent)
    sources={split:{source_video_id(r['entry']['vid']) for r in records if r['split']==split} for split in ('train','validation')}
    if sources['train']&sources['validation']:raise ValueError('Train/validation source overlap')
    for split in sources:
        if len(sources[split])!=cfg['expected_'+split+'_sources'] or sum(r['split']==split for r in records)!=cfg['expected_'+split+'_queries']:
            raise ValueError('Frozen study identity count changed')
    excluded=set(read(ROOT/parent['config']['data_run']/'excluded_sources.json'))
    if (sources['train']|sources['validation'])&excluded:raise ValueError('Excluded source overlap')
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    code={}
    names=['scripts/candidate_ranker_study.py','queryseek/retrieval/candidate_ranker.py',
           'configs/candidate_ranker_v1.json','queryseek/retrieval/curve_spans.py','queryseek/retrieval/fusion.py',
           'queryseek/eval/qvhighlights.py','scripts/localizer_study.py','scripts/analyze_localizer_study.py',
           'scripts/make_qvhighlights_subset.py']
    for name in names:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target);code[name]=sha256_file(target)
    preserved=frozen_inventory(study,parent);preserved.update(historical['input_sha256'])
    preserved.update(read(prior/'protocol.json')['preserved_sha256'])
    preserved.update({str(f):sha256_file(f) for f in prior.rglob('*') if f.is_file()})
    for r in records:
        for key,sha_key in [('video','video_sha256'),('images_path','images_sha256'),('text_path','text_sha256')]:
            if sha256_file(r[key])!=r[sha_key]:raise ValueError('Source feature/video bytes changed')
            preserved[r[key]]=r[sha_key]
        meta=read(Path(r['images_path']).parent/'feature_meta.json')
        if meta['signature']['pipeline_version']!=PIPELINE_VERSION or meta['signature']['video_sha256']!=r['video_sha256']:
            raise ValueError('Pipeline/video signature mismatch')
    items=[{'qid':r['entry']['qid'],'source':source_video_id(r['entry']['vid'])} for r in records if r['split']=='train']
    shutil.copyfile(ROOT/'docs/EVAL_CANDIDATE_RANKER.md',out/'protocol_notes_frozen.md')
    protocol={'config':cfg,'source_sha256':code,'preserved_sha256':preserved,'folds':source_folds(items,cfg['folds']),
              'protocol_notes_sha256':sha256_file(out/'protocol_notes_frozen.md'),
              'feature_names':FEATURE_NAMES,'pipeline_version':PIPELINE_VERSION,
              'versions':{m:optional_version(m) for m in ('scikit-learn','numpy','joblib','threadpoolctl')},
              'source_overlap':0,'excluded_overlap':0,'independent':False,'validation_previously_observed':True,
              'new_image_encodings':0,'encoder_calls':0,'vlm_calls':0,'fresh_query_label_access':False,'mock':False}
    write_json_atomic(out/'protocol.json',protocol);write_json_atomic(out/'status.json',{'phase':'frozen'})
    print('OUTPUT='+str(out),flush=True);return out


def prepare_records(records,out,protocol):
    """Persist GT-free candidates/features separately from train-only targets."""
    arrays={};items=[];failures=[]
    for record in records:
        entry=record['entry'];qid=entry['qid'];split=record['split']
        try:
            windows,features=candidate_table(record['times'],record['images']@record['query_embedding'],record['duration_sec'])
            arrays[f'w_{qid}']=windows;arrays[f'x_{qid}']=features
            # Validation labels are deliberately absent from this artifact.
            if split=='train':arrays[f'y_{qid}']=training_targets(windows,entry['relevant_windows'],record['duration_sec'])
        except ValueError as exc:
            arrays[f'w_{qid}']=np.empty((0,3));arrays[f'x_{qid}']=np.empty((0,len(FEATURE_NAMES)))
            if split=='train':arrays[f'y_{qid}']=np.empty(0)
            failures.append({'qid':qid,'split':split,'error_type':type(exc).__name__,'reason':str(exc)})
        items.append({'qid':qid,'source':source_video_id(entry['vid']),'split':split,'duration_sec':record['duration_sec'],
                      'candidates':len(arrays[f'w_{qid}'])})
    np.savez(out/'candidate_tables.npz',**arrays)
    write_json_atomic(out/'candidate_manifest.json',items);write_json_atomic(out/'failures.json',failures)
    signature={'pipeline_version':PIPELINE_VERSION,'config':protocol['config'],'source_sha256':protocol['source_sha256'],
               'input_sha256':protocol.get('preserved_sha256',{}),'feature_names':FEATURE_NAMES}
    write_json_atomic(out/'candidate_cache.json',{'signature':signature,'cache_key':digest(signature),
        'files_sha256':{name:sha256_file(out/name) for name in ('candidate_tables.npz','candidate_manifest.json','failures.json')}})
    return items,arrays


def fit_one(items,arrays,qids,cfg,control,folder):
    usable=[r for r in items if r['qid'] in set(qids) and r['candidates']>0]
    if not usable:raise ValueError('No training candidates')
    x=np.concatenate([arrays[f'x_{r["qid"]}'] for r in usable]);y=np.concatenate([arrays[f'y_{r["qid"]}'] for r in usable])
    weights=source_equal_weights([r['source'] for r in usable],[r['candidates'] for r in usable])
    started=time.perf_counter();model=fit_ranker(x,y,weights,cfg['model'],control)
    folder.mkdir(parents=True,exist_ok=False);joblib.dump(model,folder/'model.joblib',compress=3)
    model_sha=sha256_file(folder/'model.joblib')
    restored=joblib.load(folder/'model.joblib') # Only this locally-created, SHA-identified checkpoint.
    if not np.array_equal(model.predict(model_inputs(x,control)),restored.predict(model_inputs(x,control))):
        raise ValueError('Ranker reload mismatch')
    metadata={'control':control,'qids':qids,'usable_queries':len(usable),'sources':len({r['source'] for r in usable}),
              'candidate_rows':len(x),'iterations':int(model.n_iter_),'effective_parameters':model.get_params(),
              'feature_columns':list(FEATURE_NAMES if control=='full' else FEATURE_NAMES[:5]),
              'model_sha256':model_sha,'reload_exact':True,
              'weighted_train_mse':float(np.average((model.predict(model_inputs(x,control))-y)**2,weights=weights)),
              'fit_and_save_sec':time.perf_counter()-started}
    write_json_atomic(folder/'metadata.json',metadata)
    return model


def infer(model,items,arrays,control,cfg):
    result={}
    for item in items:
        qid=item['qid'];x=arrays[f'x_{qid}'];windows=arrays[f'w_{qid}']
        scores=None if model is None else model.predict(model_inputs(x,control)) if len(x) else np.empty(0)
        result[qid]=rank_candidates(windows,scores,cfg['top_k'],cfg['nms_iou'])
    return result


def experiments(items,arrays,cfg,out,folds):
    started=time.perf_counter();train=[r for r in items if r['split']=='train'];val=[r for r in items if r['split']=='validation']
    predictions={'oof':{'e4':infer(None,train,arrays,'full',cfg)},
                 'validation':{'e4':infer(None,val,arrays,'full',cfg)},
                 'resubstitution':{'e4':infer(None,train,arrays,'full',cfg)}}
    with threadpool_limits(limits=cfg['threads']):
        for control in cfg['controls']:
            oof={}
            for number,fold in enumerate(folds):
                if time.perf_counter()-started>cfg['wall_cap_sec']:raise TimeoutError('Frozen ranker wall cap')
                model=fit_one(train,arrays,fold['train_qids'],cfg,control,out/'models'/control/f'fold_{number}')
                holdout=[r for r in train if r['qid'] in set(fold['test_qids'])]
                oof.update(infer(model,holdout,arrays,control,cfg))
                print('FIT',control,'fold',number,'heldout_queries',len(holdout),flush=True)
            if set(oof)!={r['qid'] for r in train}:raise ValueError('OOF query coverage changed')
            predictions['oof'][control]=oof
            model=fit_one(train,arrays,[r['qid'] for r in train],cfg,control,out/'models'/control/'all_train')
            predictions['validation'][control]=infer(model,val,arrays,control,cfg)
            predictions['resubstitution'][control]=infer(model,train,arrays,control,cfg)
    for stage,methods in predictions.items():
        for name,pred in methods.items():write_json_atomic(out/f'{stage}_{name}_predictions.json',pred)
    return predictions,time.perf_counter()-started


def evaluate(predictions,gt,items,arrays,cfg):
    result={}
    for stage,methods in predictions.items():
        truth=gt['validation' if stage=='validation' else 'train']
        if any(set(pred)!=set(truth) for pred in methods.values()):raise ValueError('Full evaluation denominator changed')
        pool={qid:arrays[f'w_{qid}'].tolist() for qid in truth}
        stats={name:dict(eval_mr_brief(pred,truth)) for name,pred in methods.items()}
        contrasts={name:paired_clusters(per_query(methods[reference],truth),per_query(methods['full'],truth),**cfg['bootstrap'])
                   for name,reference in [('full_minus_e4','e4'),('full_minus_structure','structure')]}
        for value in contrasts.values():value['scope']='conditional source-group internal study; no tuning on reported scores, prior observation/design bias remains'
        result[stage]={'metrics':stats,'contrasts':contrasts,
                       'diagnostic':{name:candidate_diagnostic(pool,pred,truth) for name,pred in methods.items()}}
    return result


def run(out):
    started=time.perf_counter();protocol=verify_inputs(out);cfg=protocol['config']
    if read(out/'status.json')['phase']!='frozen':raise ValueError('Already started; use replay')
    write_json_atomic(out/'status.json',{'phase':'running'})
    study=ROOT/cfg['study_run'];records=load_records(study,verify(study))
    with threadpool_limits(limits=cfg['threads']):
        items,arrays=prepare_records(records,out,protocol)
    predictions,fit_sec=experiments(items,arrays,cfg,out,protocol['folds'])
    # No validation entry/GT is passed into fit, candidate construction or inference.
    gt={split:{r['entry']['qid']:r['entry'] for r in records if r['split']==split} for split in ('train','validation')}
    result=evaluate(predictions,gt,items,arrays,cfg)
    baseline={int(q):v for q,v in read(study/'e4_predictions.json').items()}
    for qid in baseline:
        if np.asarray(baseline[qid]).shape!=np.asarray(predictions['validation']['e4'][qid]).shape or not np.allclose(
            np.asarray(baseline[qid]),np.asarray(predictions['validation']['e4'][qid]),atol=1e-7,rtol=1e-7):
            raise ValueError('Original E4 predictions changed')
    result.update(mock=False,independent=False,new_image_encodings=0,encoder_calls=0,vlm_calls=0,downloads=0,
                  fits=len(cfg['controls'])*(cfg['folds']+1),fit_and_prediction_sec=fit_sec,
                  failed_queries=len(read(out/'failures.json')),elapsed_sec=time.perf_counter()-started,
                  default_promotion=False,fresh_query_label_access=False)
    result['artifact_sha256']={str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*')
        if p.is_file() and (p.name in ('model.joblib','metadata.json','candidate_tables.npz','candidate_cache.json',
                                     'candidate_manifest.json','failures.json') or p.name.endswith('_predictions.json'))}
    verify_inputs(out);write_json_atomic(out/'metrics.json',result);write_json_atomic(out/'status.json',{'phase':'complete'})
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2),flush=True)
    print('OUTPUT='+str(out),flush=True);return result


def replay(out):
    started=time.perf_counter();p=verify_inputs(out);cfg=p['config'];saved=read(out/'metrics.json')
    for name,sha in saved['artifact_sha256'].items():
        if sha256_file(out/name)!=sha:raise ValueError('Saved ranker artifact changed')
    with np.load(out/'candidate_tables.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    items=read(out/'candidate_manifest.json');train=[r for r in items if r['split']=='train'];val=[r for r in items if r['split']=='validation']
    predictions={'oof':{'e4':infer(None,train,arrays,'full',cfg)},'validation':{'e4':infer(None,val,arrays,'full',cfg)},
                 'resubstitution':{'e4':infer(None,train,arrays,'full',cfg)}}
    with threadpool_limits(limits=cfg['threads']):
        for control in cfg['controls']:
            oof={}
            for number,fold in enumerate(p['folds']):
                model=joblib.load(out/'models'/control/f'fold_{number}'/'model.joblib')
                oof.update(infer(model,[r for r in train if r['qid'] in set(fold['test_qids'])],arrays,control,cfg))
            model=joblib.load(out/'models'/control/'all_train'/'model.joblib')
            predictions['oof'][control]=oof
            predictions['validation'][control]=infer(model,val,arrays,control,cfg)
            predictions['resubstitution'][control]=infer(model,train,arrays,control,cfg)
    for stage,methods in predictions.items():
        for name,pred in methods.items():
            actual={int(q):v for q,v in read(out/f'{stage}_{name}_predictions.json').items()}
            if set(actual)!=set(pred):raise ValueError('Prediction identity mismatch')
            for q in pred:
                if not np.array_equal(np.asarray(actual[q]),np.asarray(pred[q])):raise ValueError('Prediction replay mismatch')
    records=load_records(ROOT/cfg['study_run'],verify(ROOT/cfg['study_run']))
    # Also regenerate every candidate/feature/target from frozen original arrays.
    with threadpool_limits(limits=cfg['threads']):
        for r in records:
            q=r['entry']['qid'];w,x=candidate_table(r['times'],r['images']@r['query_embedding'],r['duration_sec'])
            if not np.array_equal(w,arrays[f'w_{q}']) or not np.array_equal(x,arrays[f'x_{q}']):raise ValueError('Candidate feature replay mismatch')
            if r['split']=='train' and not np.array_equal(training_targets(w,r['entry']['relevant_windows'],r['duration_sec']),arrays[f'y_{q}']):
                raise ValueError('Train target replay mismatch')
            if r['split']=='validation' and f'y_{q}' in arrays:raise ValueError('Validation target in inference table')
    gt={split:{r['entry']['qid']:r['entry'] for r in records if r['split']==split} for split in ('train','validation')}
    evaluation=evaluate(predictions,gt,items,arrays,cfg)
    if any(evaluation[stage]!=saved[stage] for stage in evaluation):raise ValueError('Evaluation replay mismatch')
    result={'passed':True,'models_reloaded':saved['fits'],'prediction_files':9,'candidate_queries':len(items),
            'new_fits':0,'new_image_encodings':0,'all_predictions_features_targets_metrics_intervals_rebuilt':True,
            'elapsed_sec':time.perf_counter()-started}
    write_json_atomic(out/('replay_'+new_job_id()+'.json'),result);print(json.dumps(result,indent=2));return result


def mock(out):
    from scripts.make_synthetic_fixture import build_video
    from queryseek.ingest.ffprobe import probe
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    video=build_video(out/'fixture',duration=12.);info=probe(video)
    # Synthetic curves based on generated fixture duration; no image encoder.
    duration=float(info['format']['duration']);t=np.arange(0,duration,1.)
    records=[];rng=np.random.default_rng(12)
    for i in range(8):
        a=2.+i%3;b=a+4.;scores=rng.normal(0,.01,len(t))+((t>=a)&(t<b))
        records.append({'split':'train' if i<6 else 'validation','times':t,'duration_sec':duration,
            'images':scores[:,None],'query_embedding':np.ones(1),
            'entry':{'qid':i,'vid':f'Synth{i:06d}_0.0_12.0','query':'synthetic','relevant_windows':[[a,b]]}})
    cfg=read(ROOT/'configs/candidate_ranker_v1.json');cfg.update(folds=2,threads=1)
    cfg['model']={**cfg['model'],'max_iter':3,'min_samples_leaf':2}
    p={'config':cfg,'source_sha256':{},'mock':True}
    items,arrays=prepare_records(records,out,p)
    folds=source_folds([r for r in items if r['split']=='train'],2)
    predictions,sec=experiments(items,arrays,cfg,out,folds)
    result={'mock':True,'fits':6,'model_iterations':3,'generated_video':str(video),
            'scope':'synthetic plumbing only; not source-independent data or real accuracy','elapsed_sec':sec}
    write_json_atomic(out/'metrics.json',result);print('OUTPUT='+str(out));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','run','replay','mock'));parser.add_argument('--run',type=Path)
    parser.add_argument('--config',default='configs/candidate_ranker_v1.json');args=parser.parse_args()
    if args.phase=='freeze':freeze(read(ROOT/args.config),ROOT/'outputs/candidate_ranker'/new_job_id())
    elif args.phase=='mock':mock(ROOT/'outputs/candidate_ranker_mock'/new_job_id())
    else:
        if args.run is None:parser.error('--run required')
        out=checked_output(args.run)
        if args.phase=='replay':replay(out)
        else:
            try:run(out)
            except Exception as exc:
                write_json_atomic(out/'status.json',{'phase':'failed','error_type':type(exc).__name__});raise
