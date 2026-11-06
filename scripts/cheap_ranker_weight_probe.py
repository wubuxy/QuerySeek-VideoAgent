"""One fixed three-fold loss-weight intervention; no new media or features."""
import argparse,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.audit_ranker_bottleneck import PARENT,overlaps
from scripts import multiscale_ranker_study as prior
from scripts.localizer_study import read,rows
from scripts.analyze_localizer_study import per_query,paired_clusters
from scripts.sampling_probe import checked_output
from queryseek.retrieval.candidate_ranker import fit_ranker,rank_candidates
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.tools.export import new_job_id

def balanced_weights(windows,targets,gt):
    """Half mass to background, half equally to represented events, per query."""
    y=np.asarray(targets,dtype=float)
    if not len(y) or len(windows)!=len(y) or not np.isfinite(y).all() or np.any((y<0)|(y>1)):
        raise ValueError('Invalid utilities')
    pos=y>0;neg=~pos
    if not pos.any() or not neg.any():return np.full(len(y),1/len(y))
    assignment=overlaps(windows,gt).argmax(axis=1)
    represented=np.unique(assignment[pos]);weights=np.zeros(len(y))
    weights[neg]=.5/neg.sum()
    for event in represented:
        mask=pos&(assignment==event);weights[mask]=.5/(len(represented)*mask.sum())
    assert np.isclose(weights.sum(),1.) and (weights>0).all()
    return weights

def run(decode_run,out=None):
    started=time.perf_counter();p=prior.verify(PARENT);cfg=p['model']
    baseline=read(decode_run/'metrics.json')
    replay=out is not None
    out=checked_output(out or ROOT/'outputs/cheap_ranker_weights'/new_job_id())
    source_names=['scripts/cheap_ranker_weight_probe.py','scripts/audit_ranker_bottleneck.py',
                  'queryseek/retrieval/candidate_ranker.py','queryseek/retrieval/fusion.py','queryseek/eval/qvhighlights.py']
    inputs=[PARENT/'candidate_tables.npz',PARENT/'folds.json',PARENT/'metrics.json',decode_run/'metrics.json',decode_run/'dedupe_predictions.json']
    if not replay:
        out.mkdir(parents=True)
        protocol={'model':cfg,'folds':3,'nms_iou':.5,'top_k':5,'foreground_mass':.5,
          'weighting':'equal source and query mass, half background and half equal represented GT events; unchanged targets/features',
          'scope':'training-source OOF development only; no old/new validation or fresh use',
          'stop':'exactly three fixed fits, no tuning; pass requires short>=E4 and full/middle/long mAP and R1@.5 within 1pp of dedupe control',
          'bootstrap':{'seed':12,'resamples':5000},'decode_run':str(decode_run.relative_to(ROOT)),
          'source_sha256':{n:sha256_file(ROOT/n) for n in source_names},
          'input_sha256':{str(f.relative_to(ROOT)):sha256_file(f) for f in inputs}}
        write_json_atomic(out/'protocol.json',protocol)
        for n in source_names:
            dst=out/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    else:protocol=read(out/'protocol.json')
    for n,h in protocol['source_sha256'].items():assert sha256_file(ROOT/n)==sha256_file(out/'source_snapshot'/n)==h
    for n,h in protocol['input_sha256'].items():assert sha256_file(ROOT/n)==h
    items={r['qid']:r for r in read(PARENT/'candidate_manifest.json') if r['split']=='train'}
    truth={r['entry']['qid']:r['entry'] for r in rows(ROOT/p['expanded_data']/'feature_records.jsonl') if r['split']=='train'}
    with np.load(PARENT/'candidate_tables.npz',allow_pickle=False) as z:
        arrays={f'{kind}_{q}':z[f'union_{kind}_{q}'] for q in items for kind in ('w','x','y')}
    result={};fit_info=[]
    with threadpool_limits(limits=4):
        for i,fold in enumerate(read(PARENT/'folds.json')):
            train=fold['train_qids'];test=fold['test_qids']
            assert not {items[q]['source'] for q in train}&{items[q]['source'] for q in test}
            path=out/f'fold_{i}.joblib';tick=time.perf_counter()
            if replay:
                assert sha256_file(path)==read(out/'metrics.json')['model_sha256'][path.name]
                model=joblib.load(path)
            else:
                counts=Counter(items[q]['source'] for q in train)
                xx=np.concatenate([arrays[f'x_{q}'] for q in train]);yy=np.concatenate([arrays[f'y_{q}'] for q in train])
                ww=np.concatenate([balanced_weights(arrays[f'w_{q}'],arrays[f'y_{q}'],truth[q]['relevant_windows'])/counts[items[q]['source']] for q in train])
                ww/=ww.mean();model=fit_ranker(xx,yy,ww,cfg);assert model.n_iter_==100
                joblib.dump(model,path,compress=3)
                assert np.array_equal(model.predict(xx),joblib.load(path).predict(xx))
            for q in test:result[q]=rank_candidates(arrays[f'w_{q}'],model.predict(arrays[f'x_{q}']),5,.5)
            fit_info.append({'fold':i,'train_queries':len(train),'test_queries':len(test),'elapsed_sec':time.perf_counter()-tick})
    normalized={str(q):[list(w) for w in ws] for q,ws in result.items()}
    if replay:assert read(out/'predictions.json')==normalized
    else:write_json_atomic(out/'predictions.json',normalized)
    metrics=dict(eval_mr_brief(result,truth))
    control={int(q):v for q,v in read(decode_run/'dedupe_predictions.json').items()}
    contrast=paired_clusters(per_query(control,truth),per_query(result,truth),**protocol['bootstrap'])
    contrast['scope']='conditional source OOF development, not independent confirmation'
    checks={k:metrics[k]>=baseline['results']['dedupe'][k]-1. for k in
            ['MR-full-mAP','MR-middle-mAP','MR-long-mAP','MR-full-R1@0.5']}
    checks['short_at_least_e4']=metrics['MR-short-mAP']>=baseline['e4']['MR-short-mAP']
    if replay:
        saved=read(out/'metrics.json');assert metrics==saved['metrics'] and contrast==saved['contrast']
        report={'passed':True,'models_reloaded':3,'new_fits':0,'predictions_metrics_intervals_exact':True,'elapsed_sec':time.perf_counter()-started}
        write_json_atomic(out/'replay.json',report)
    else:
        report={'metrics':metrics,'contrast':contrast,'gate_checks':checks,'passes_gate':all(checks.values()),
                'fits':fit_info,'new_fits':3,'new_encodings':0,'downloads':0,'vlm_calls':0,'default_promotion':False,
                'elapsed_sec':time.perf_counter()-started,'model_sha256':{f.name:sha256_file(f) for f in out.glob('*.joblib')}}
        write_json_atomic(out/'metrics.json',report)
    print('OUTPUT='+str(out));print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--decode-run',type=Path,required=True);ap.add_argument('--replay',type=Path);a=ap.parse_args()
    run(checked_output(a.decode_run),checked_output(a.replay) if a.replay else None)
