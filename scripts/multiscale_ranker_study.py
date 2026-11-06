"""Frozen dataset expansion x multiscale candidates; only existing sklearn fits."""
from __future__ import annotations
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import localizer_study as data
from scripts.candidate_ranker_study import source_folds,digest
from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output
from scripts.analyze_localizer_study import per_query,paired_clusters,candidate_diagnostic
from queryseek.retrieval.multiscale_candidates import multiscale_table
from queryseek.retrieval.candidate_ranker import candidate_table,training_targets,source_equal_weights,fit_ranker,model_inputs,rank_candidates
from queryseek.eval.qvhighlights import eval_mr_brief,temporal_iou
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.tools.export import new_job_id

ARMS={'e4_full_base':('e4','full','base'),'union_full_base':('union','full','base'),
      'union_structure_base':('union','structure','base'),'union_full_expanded':('union','full','expanded'),
      'union_structure_expanded':('union','structure','expanded')}


def freeze(cfg,expanded,out):
    dp=data.verify(expanded);parent=ROOT/cfg['study_run'];data.verify(parent)
    if dp['extension']['config']!=cfg:raise ValueError('Frozen data/model configuration mismatch')
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    names=['scripts/multiscale_ranker_study.py','queryseek/retrieval/multiscale_candidates.py',
           'configs/multiscale_ranker_v1.json','queryseek/retrieval/candidate_ranker.py',
           'queryseek/retrieval/curve_spans.py','queryseek/eval/qvhighlights.py',
           'scripts/analyze_localizer_study.py','scripts/candidate_ranker_study.py',
           'configs/candidate_ranker_v1.json','tests/test_multiscale_candidates.py']
    sources={n:sha256_file(ROOT/n) for n in names}
    for n in names:
        dst=out/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
    preserved={str(f.relative_to(ROOT)):sha256_file(f) for f in [parent/'protocol.json',parent/'features_summary.json',
              parent/'feature_records.jsonl',parent/'e4_predictions.json',parent/'metrics.json']}
    p={'config':cfg,'expanded_data':str(expanded.relative_to(ROOT)),
       'expanded_protocol_sha256':sha256_file(expanded/'protocol.json'),'source_sha256':sources,
       'preserved_sha256':preserved,'model':data.read(ROOT/'configs/candidate_ranker_v1.json')['model'],
       'arms':ARMS,'pipeline_version':'2','fresh_query_label_access':False,
       'new_locator_or_encoder_parameters':0,'validation_previously_observed':True,'independent':False,
       'stop_rule':'five fixed all-train fits plus three source folds for each expanded control, no score-based reruns'}
    write_json_atomic(out/'protocol.json',p);write_json_atomic(out/'status.json',{'phase':'frozen'})
    print('OUTPUT='+str(out),flush=True);return out


def verify(out):
    p=data.read(out/'protocol.json');expanded=ROOT/p['expanded_data'];data.verify(expanded)
    if sha256_file(expanded/'protocol.json')!=p['expanded_protocol_sha256']:raise ValueError('Data protocol changed')
    for n,h in p['preserved_sha256'].items():
        if sha256_file(ROOT/n)!=h:raise ValueError('Historical input changed: '+n)
    for n,h in p['source_sha256'].items():
        if sha256_file(ROOT/n)!=h or sha256_file(out/'source_snapshot'/n)!=h:raise ValueError('Frozen source changed: '+n)
    return p


def prepare(out,p):
    cfg=p['config'];expanded=ROOT/p['expanded_data'];records=data.load_records(expanded,data.verify(expanded))
    base={r['entry']['qid'] for r in data.rows(ROOT/cfg['study_run']/'feature_records.jsonl') if r['split']=='train'}
    items=[];arrays={}
    for r in records:
        e=r['entry'];q=e['qid'];split=r['split'];s=r['images']@r['query_embedding']
        for pool in ('e4','union'):
            w,x=(candidate_table(r['times'],s,r['duration_sec']) if pool=='e4' else
                 multiscale_table(r['times'],s,r['duration_sec'],cfg['lengths_sec'],cfg['stride_fraction'],cfg['min_stride_sec']))
            arrays[f'{pool}_w_{q}']=w;arrays[f'{pool}_x_{q}']=x
            if split=='train':arrays[f'{pool}_y_{q}']=training_targets(w,e['relevant_windows'],r['duration_sec'])
        items.append({'qid':q,'source':source_video_id(e['vid']),'split':split,'base':q in base,'duration_sec':r['duration_sec']})
    train=[i for i in items if i['split']=='train'];val=[i for i in items if i['split']=='validation']
    if len(base)!=256 or len(val)!=65 or {r['qid'] for r in train if r['base']}!=base:raise ValueError('Original denominators changed')
    if {r['source'] for r in train}&{r['source'] for r in val}:raise ValueError('Source overlap')
    if any(k.startswith(('union_y_','e4_y_')) and int(k.rsplit('_',1)[1]) in {r['qid'] for r in val} for k in arrays):raise ValueError('Validation target leakage')
    np.savez(out/'candidate_tables.npz',**arrays);write_json_atomic(out/'candidate_manifest.json',items)
    folds=source_folds(train,cfg['folds']);write_json_atomic(out/'folds.json',folds)
    signature={'pipeline_version':'2','config':cfg,'source_sha256':p['source_sha256'],
               'feature_manifest_sha256':sha256_file(expanded/'feature_records.jsonl')}
    write_json_atomic(out/'candidate_cache.json',{'signature':signature,'cache_key':digest(signature)})
    print('TABLES',len(train),len(val),'union rows',sum(len(arrays[f'union_w_{r["qid"]}']) for r in items),flush=True)
    return items,arrays,folds


def infer(model,items,arrays,pool,control,cfg):
    result={}
    for r in items:
        q=r['qid'];x=arrays[f'{pool}_x_{q}'];w=arrays[f'{pool}_w_{q}']
        score=None if model is None else model.predict(model_inputs(x,control))
        result[q]=rank_candidates(w,score,cfg['top_k'],cfg['nms_iou'])
    return result


def fit_or_reload(out,name,items,arrays,pool,control,p,replay):
    file=out/'models'/(name+'.joblib')
    if replay:return joblib.load(file)
    if file.exists():raise ValueError('Existing fit; use replay')
    xx=np.concatenate([arrays[f'{pool}_x_{r["qid"]}'] for r in items]);yy=np.concatenate([arrays[f'{pool}_y_{r["qid"]}'] for r in items])
    weights=source_equal_weights([r['source'] for r in items],[len(arrays[f'{pool}_w_{r["qid"]}']) for r in items])
    start=time.perf_counter();model=fit_ranker(xx,yy,weights,p['model'],control)
    if model.n_iter_!=100:raise ValueError('Training budget changed')
    file.parent.mkdir(exist_ok=True);joblib.dump(model,file,compress=3)
    restored=joblib.load(file)
    if not np.array_equal(model.predict(model_inputs(xx,control)),restored.predict(model_inputs(xx,control))):raise ValueError('Reload mismatch')
    write_json_atomic(file.with_suffix('.json'),{'queries':len(items),'sources':len({r['source'] for r in items}),
         'rows':len(xx),'iterations':model.n_iter_,'fit_sec':time.perf_counter()-start,'reload_exact':True,'control':control,'pool':pool})
    print('FIT',name,len(items),'queries',len(xx),'rows',flush=True);return model


def predictions(out,p,items,arrays,folds,replay):
    cfg=p['config'];train=[r for r in items if r['split']=='train'];val=[r for r in items if r['split']=='validation']
    result={'validation':{'e4':infer(None,val,arrays,'e4','full',cfg)},
            'oof':{'e4':infer(None,train,arrays,'e4','full',cfg)},'resubstitution':{}}
    for name,(pool,control,scale) in ARMS.items():
        rr=[r for r in train if scale=='expanded' or r['base']]
        model=fit_or_reload(out,name,rr,arrays,pool,control,p,replay)
        result['validation'][name]=infer(model,val,arrays,pool,control,cfg)
        # Comparable train denominator across all fits; contains unseen extra data for base arms.
        result['resubstitution'][name]=infer(model,train,arrays,pool,control,cfg)
    for control in ('full','structure'):
        oof={}
        for i,fold in enumerate(folds):
            rr=[r for r in train if r['qid'] in set(fold['train_qids'])]
            model=fit_or_reload(out,f'oof_{control}_{i}',rr,arrays,'union',control,p,replay)
            holdout=[r for r in train if r['qid'] in set(fold['test_qids'])]
            oof.update(infer(model,holdout,arrays,'union',control,cfg))
        if set(oof)!={r['qid'] for r in train}:raise ValueError('OOF coverage mismatch')
        result['oof'][control]=oof
    for stage,methods in result.items():
        for name,pred in methods.items():
            file=out/f'{stage}_{name}_predictions.json';normalized={str(q):[list(w) for w in ws] for q,ws in pred.items()}
            if replay:
                if data.read(file)!=normalized:raise ValueError('Prediction replay differs: '+str(file))
            else:write_json_atomic(file,pred)
    return result


def coverage(pool,gt):
    result={}
    for name,low,high in [('short',0,10),('middle',10,30),('long',30,float('inf'))]:
        values=[max((temporal_iou(a,b,c,d) for a,b,*_ in pool[q]),default=0.) for q,e in gt.items()
                for c,d in e['relevant_windows'] if low<d-c<=high]
        result[name]={'events':len(values),**{f'hits_{t}':sum(v>=t for v in values) for t in (.5,.7,.9)}}
    return result


def evaluate(pred,items,arrays,p):
    cfg=p['config'];expanded=ROOT/p['expanded_data'];records=data.rows(expanded/'feature_records.jsonl')
    truth={s:{r['entry']['qid']:r['entry'] for r in records if r['split']==s} for s in ('train','validation')}
    stats={}
    for stage,methods in pred.items():
        gt=truth['validation' if stage=='validation' else 'train']
        for pp in methods.values():
            if set(pp)!=set(gt):raise ValueError('Evaluation denominator changed')
        stats[stage]={name:dict(eval_mr_brief(pp,gt)) for name,pp in methods.items()}
    val=pred['validation'];gt=truth['validation'];pairs={
        'expanded_union_minus_e4':('e4','union_full_expanded'),
        'geometry_effect_base':('e4_full_base','union_full_base'),
        'data_effect_full':('union_full_base','union_full_expanded'),
        'semantics_effect_expanded':('union_structure_expanded','union_full_expanded'),
        'base_union_minus_e4':('e4','union_full_base')}
    stats['contrasts']={name:paired_clusters(per_query(val[a],gt),per_query(val[b],gt),**cfg['bootstrap']) for name,(a,b) in pairs.items()}
    stats['oof_contrasts']={name:paired_clusters(per_query(pred['oof'][a],truth['train']),per_query(pred['oof']['full'],truth['train']),**cfg['bootstrap'])
                            for name,a in [('full_minus_e4','e4'),('full_minus_structure','structure')]}
    stats['diagnostic']={}
    for split in ('train','validation'):
        gt=truth[split];stats['diagnostic'][split]={}
        for pool in ('e4','union'):
            ws={q:arrays[f'{pool}_w_{q}'].tolist() for q in gt}
            pp=val['e4' if pool=='e4' else 'union_full_expanded'] if split=='validation' else pred['oof']['e4' if pool=='e4' else 'full']
            stats['diagnostic'][split][pool]={**candidate_diagnostic(ws,pp,gt),'event_coverage':coverage(ws,gt),
                                            'candidates':sum(len(w) for w in ws.values())}
    baseline={int(q):v for q,v in data.read(ROOT/cfg['study_run']/'e4_predictions.json').items()}
    if baseline!={q:[list(w) for w in ws] for q,ws in val['e4'].items()}:raise ValueError('Historical E4 changed')
    return stats


def run(out,replay=False):
    start=time.perf_counter();p=verify(out);cfg=p['config']
    if replay:
        saved=data.read(out/'metrics.json')
        for n,h in saved['artifact_sha256'].items():
            if sha256_file(out/n)!=h:raise ValueError('Artifact changed: '+n)
        items=data.read(out/'candidate_manifest.json');folds=data.read(out/'folds.json')
        with np.load(out/'candidate_tables.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    else:
        if data.read(out/'status.json')['phase']!='frozen':raise ValueError('Already started')
        write_json_atomic(out/'status.json',{'phase':'running'})
        with threadpool_limits(limits=cfg['threads']):items,arrays,folds=prepare(out,p)
    with threadpool_limits(limits=cfg['threads']):pred=predictions(out,p,items,arrays,folds,replay)
    # All held-out predictions are persisted before evaluation labels are read.
    stats=evaluate(pred,items,arrays,p)
    if replay:
        if stats!=saved['results']:raise ValueError('Metric/interval replay differs')
        result={'passed':True,'models_reloaded':11,'new_fits':0,'all_predictions_metrics_intervals_exact':True,'elapsed_sec':time.perf_counter()-start}
        write_json_atomic(out/('replay_'+new_job_id()+'.json'),result);print(json.dumps(result),flush=True);return result
    result={'results':stats,'elapsed_sec':time.perf_counter()-start,'fits':11,'default_promotion':False,'independent':False,
            'new_ranker_image_encodings':0,'new_ranker_text_encodings':0,'vlm_calls':0,'fresh_query_label_access':False,
            'expanded_data':p['expanded_data'],'train_sources':len({r['source'] for r in items if r['split']=='train'}),
            'train_queries':sum(r['split']=='train' for r in items),'validation_sources':64,'validation_queries':65,
            'train_metrics_scope':'expanded arms resubstitution; base arms mixed base resubstitution plus extra-source holdout; use OOF for train generalization'}
    result['artifact_sha256']={str(f.relative_to(out)):sha256_file(f) for f in out.rglob('*') if f.is_file() and
         (f.suffix=='.joblib' or f.name.endswith('_predictions.json') or f.name in ('candidate_tables.npz','candidate_manifest.json','folds.json','candidate_cache.json') or f.parent.name=='models')}
    verify(out);write_json_atomic(out/'metrics.json',result);write_json_atomic(out/'status.json',{'phase':'complete'})
    print(json.dumps({'elapsed_sec':result['elapsed_sec'],'validation':stats['validation'],'contrasts':stats['contrasts']}),flush=True);return result


def mock(video,out):
    """Generated-video mechanics only; never report synthetic retrieval accuracy."""
    out=checked_output(out);out.mkdir(parents=True,exist_ok=False)
    cfg=data.read(ROOT/'configs/multiscale_ranker_v1.json');cfg={**cfg,'folds':3}
    p={'config':cfg,'model':data.read(ROOT/'configs/candidate_ranker_v1.json')['model']}
    records=data.study_mock_examples(video,{})
    arrays={};items=[]
    for r in records:
        q=r['entry']['qid'];s=r['images']@r['query_embedding']
        for pool in ('e4','union'):
            w,x=candidate_table(r['times'],s,r['duration_sec']) if pool=='e4' else multiscale_table(r['times'],s,r['duration_sec'],cfg['lengths_sec'])
            arrays[f'{pool}_w_{q}']=w;arrays[f'{pool}_x_{q}']=x
            if r['split']=='train':arrays[f'{pool}_y_{q}']=training_targets(w,r['entry']['relevant_windows'],r['duration_sec'])
        items.append({'qid':q,'source':source_video_id(r['entry']['vid']),'split':r['split'],'base':q<3})
    folds=source_folds([r for r in items if r['split']=='train'],3)
    with threadpool_limits(limits=1):
        first=predictions(out,p,items,arrays,folds,False);second=predictions(out,p,items,arrays,folds,True)
    if first!=second:raise ValueError('Mock replay mismatch')
    result={'mock':True,'passed':True,'scope':'generated synthetic fixture mechanics only; no real accuracy claim',
            'models_fit_and_reloaded':11,'predictions_exact':True,'vlm_calls':0}
    write_json_atomic(out/'metrics.json',result);print('OUTPUT='+str(out),flush=True);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['freeze','run','replay','mock']);ap.add_argument('--run',type=Path);ap.add_argument('--data-run',type=Path);ap.add_argument('--video',type=Path,default=ROOT/'assets/demo/sample.mp4');a=ap.parse_args()
    if a.phase=='mock':mock(a.video,ROOT/'outputs/multiscale_ranker_mock'/new_job_id())
    elif a.phase=='freeze':
        if a.data_run is None:ap.error('--data-run required')
        freeze(data.read(ROOT/'configs/multiscale_ranker_v1.json'),checked_output(a.data_run),ROOT/'outputs/multiscale_ranker'/new_job_id())
    else:
        if a.run is None:ap.error('--run required')
        run(checked_output(a.run),a.phase=='replay')
