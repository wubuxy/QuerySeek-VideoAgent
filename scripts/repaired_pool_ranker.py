"""Predeclared calibration of the repaired dense pool using existing sklearn glue."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from scripts.localizer_study import read,rows
from scripts import event_localizer_repair as repair
from scripts.sampling_probe import checked_output
from scripts.analyze_localizer_study import per_query,paired_clusters
from queryseek.retrieval.candidate_ranker import candidate_features,training_targets,source_equal_weights,fit_ranker,model_inputs,rank_candidates
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.tools.export import new_job_id


def freeze(cfg,out):
    upstream=ROOT/cfg['upstream_run']
    if (upstream/'metrics.json').exists():raise ValueError('Must predeclare before upstream validation metrics exist')
    cp=read(upstream/'protocol.json');out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    names=['scripts/repaired_pool_ranker.py','configs/repaired_pool_ranker_v1.json','tests/test_repaired_pool_ranker.py',
           'queryseek/retrieval/candidate_ranker.py','configs/candidate_ranker_v1.json']
    source={n:sha256_file(ROOT/n) for n in names}
    p={'config':cfg,'upstream_protocol_sha256':sha256_file(upstream/'protocol.json'),'source_sha256':source,
       'model_config':read(ROOT/cfg['reference_ranker_config'])['model'],'upstream_metrics_absent_at_freeze':True,
       'generator_choice_before_validation':'event_relative fixed by hypothesis, never choose from observed scores',
       'new_image_encodings':0,'new_text_encodings':0,'vlm_calls':0,'fresh_query_label_access':False,'independent':False}
    if p['model_config']['max_iter']!=100 or p['model_config']['early_stopping'] is not False:raise ValueError('Fixed existing ranker required')
    write_json_atomic(out/'protocol.json',p);write_json_atomic(out/'status.json',{'phase':'frozen'})
    print('OUTPUT='+str(out),flush=True);return out


def verify(out):
    p=read(out/'protocol.json');u=ROOT/p['config']['upstream_run']
    if sha256_file(u/'protocol.json')!=p['upstream_protocol_sha256']:raise ValueError('Upstream protocol changed')
    for n,sha in p['source_sha256'].items():
        if sha256_file(ROOT/n)!=sha:raise ValueError('Frozen calibration source changed')
    return p


def prepare(p):
    cfg=p['config'];u=ROOT/cfg['upstream_run'];cp=repair.verify(u)
    if read(u/'status.json')['phase']!='complete':raise ValueError('Upstream fits must complete')
    um=read(u/'metrics.json')
    for n,sha in um['artifact_sha256'].items():
        if sha256_file(u/n)!=sha:raise ValueError('Upstream fitted artifact changed')
    pp=cp['parent_protocol'];original=ROOT/pp['config']['parent_run'];manifest=read(original/'inference_manifest.json')
    entries={r['entry']['qid']:r['entry'] for r in rows(ROOT/pp['parent_protocol']['config']['study_run']/'feature_records.jsonl')}
    pointer=read(original/'text_cache.json');cache=Path(pointer['folder'])
    if sha256_file(cache/'features.npz')!=pointer['features_sha256']:raise ValueError('Text cache changed')
    result={};byq={r['qid']:r for r in manifest}
    with np.load(cache/'features.npz',allow_pickle=False) as text:
        visual={}
        for split in ('train','validation'):
            pools=read(u/'predictions'/cfg['generator']/f'{split}_pool.json')
            for q,windows in pools.items():
                q=int(q);r=byq[q];assert r['split']==split
                if r['images_path'] not in visual:
                    if sha256_file(r['images_path'])!=r['images_sha256']:raise ValueError('Visual cache changed')
                    with np.load(r['images_path'],allow_pickle=False) as f:visual[r['images_path']]=(f['times'],f['images'])
                times,images=visual[r['images_path']];scores=images@text[f'pool_{q}']
                x=candidate_features(times,scores,r['duration_sec'],windows)
                result[q]={'windows':np.asarray(windows,float),'features':x,'source':r['source'],'split':split,
                           'target':training_targets(windows,entries[q]['relevant_windows'],r['duration_sec']) if split=='train' else None}
    return result,entries,u,um


def predict(model,table,control,cfg):
    return {q:rank_candidates(r['windows'],model.predict(model_inputs(r['features'],control)),cfg['top_k'],cfg['nms_iou']) for q,r in table.items()}


def run(out,replay=False):
    started=time.perf_counter();p=verify(out);cfg=p['config'];table,entries,u,um=prepare(p)
    train={q:r for q,r in table.items() if r['split']=='train'};val={q:r for q,r in table.items() if r['split']=='validation'}
    if len(train)!=256 or len(val)!=65:raise ValueError('Frozen denominators changed')
    if any(r['target'] is not None for r in val.values()):raise ValueError('Validation target leakage')
    truth={q:entries[q] for q in val};train_gt={q:entries[q] for q in train}
    original=ROOT/read(u/'protocol.json')['parent_protocol']['config']['parent_run']
    e4={int(q):v for q,v in read(original/'e4_predictions.json').items()}
    raw={int(q):v for q,v in read(u/'predictions'/cfg['generator']/'validation_dense.json').items()}
    stats={'models':{},'contrasts':{},'raw':dict(eval_mr_brief(raw,truth)),'e4':dict(eval_mr_brief(e4,truth))};predictions={}
    with threadpool_limits(limits=4):
        for control in cfg['controls']:
            file=out/f'{control}.joblib'
            if replay:
                saved=read(out/'metrics.json')
                if sha256_file(file)!=saved['artifact_sha256'][file.name]:raise ValueError('Calibrator changed')
                model=joblib.load(file)
            else:
                if file.exists():raise ValueError('Existing fit; use replay')
                xx=np.concatenate([r['features'] for r in train.values()]);yy=np.concatenate([r['target'] for r in train.values()])
                weights=source_equal_weights([r['source'] for r in train.values()],[len(r['windows']) for r in train.values()])
                model=fit_ranker(xx,yy,weights,p['model_config'],control);joblib.dump(model,file)
            if model.n_iter_!=100:raise ValueError('Fixed fitting budget changed')
            pred=predict(model,val,control,cfg);tp=predict(model,train,control,cfg);predictions[control]=pred
            stats['models'][control]={'validation':dict(eval_mr_brief(pred,truth)),'train_resubstitution':dict(eval_mr_brief(tp,train_gt)),'iterations':model.n_iter_}
            for name,pr in [('validation',pred),('train',tp)]:
                path=out/f'{control}_{name}_predictions.json';normalized={str(q):[list(w) for w in ws] for q,ws in pr.items()}
                if replay:
                    if read(path)!=normalized:raise ValueError('Calibration prediction replay differs')
                else:write_json_atomic(path,pr)
            for name,base in [('raw',raw),('e4',e4)]:
                st=paired_clusters(per_query(base,truth),per_query(pred,truth),**cfg['bootstrap'])
                st['scope']='predeclared fixed-calibrator internal comparison; no generator OOF or independent generalization claim'
                stats['contrasts'][control+'_minus_'+name]=st
        stats['contrasts']['full_minus_structure']=paired_clusters(per_query(predictions['structure'],truth),per_query(predictions['full'],truth),**cfg['bootstrap'])
    if replay:
        if any(saved[k]!=v for k,v in stats.items()):raise ValueError('Calibration metrics replay differs')
        result={'passed':True,'models_reloaded':2,'new_fits':0,'all_predictions_metrics_intervals_rebuilt':True,'elapsed_sec':time.perf_counter()-started}
        write_json_atomic(out/('replay_'+new_job_id()+'.json'),result);print(json.dumps(result),flush=True);return result
    stats.update(train_candidates=sum(len(r['windows']) for r in train.values()),validation_candidates=sum(len(r['windows']) for r in val.values()),
                 upstream_metrics_sha256=sha256_file(u/'metrics.json'),elapsed_sec=time.perf_counter()-started,
                 generator_training_seen_by_calibrator=True,train_scores_are_resubstitution=True,independent=False,
                 new_image_encodings=0,new_text_encodings=0,vlm_calls=0,fresh_query_label_access=False)
    stats['artifact_sha256']={f.name:sha256_file(f) for f in out.iterdir() if f.suffix=='.joblib' or f.name.endswith('_predictions.json')}
    write_json_atomic(out/'metrics.json',stats);write_json_atomic(out/'status.json',{'phase':'complete'})
    print(json.dumps(stats),flush=True);return stats


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['freeze','run','replay']);parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.phase=='freeze':freeze(read(ROOT/'configs/repaired_pool_ranker_v1.json'),ROOT/'outputs/repaired_pool_ranker'/new_job_id())
    else:
        if args.run is None:parser.error('--run required')
        run(checked_output(args.run),args.phase=='replay')
