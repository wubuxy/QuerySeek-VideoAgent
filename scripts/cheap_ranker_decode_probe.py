"""One preregistered zero-fit decoder contrast using saved source-OOF scores."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_ranker_bottleneck import PARENT,overlaps,matched_redundancy
from scripts import multiscale_ranker_study as prior
from scripts.localizer_study import read,rows
from scripts.analyze_localizer_study import per_query,paired_clusters,candidate_diagnostic
from scripts.sampling_probe import checked_output
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.retrieval.candidate_ranker import rank_candidates,source_equal_weights
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.tools.export import new_job_id

def decode(windows,scores,nms_iou):
    return rank_candidates(windows,scores,top_k=5,nms_iou=nms_iou)

def run(diagnostic):
    start=time.perf_counter();p=prior.verify(PARENT);d=read(diagnostic/'metrics.json')
    assert sha256_file(diagnostic/'oof_scores.npz')==d['score_sha256']
    for name,h in d['input_sha256'].items():assert sha256_file(ROOT/name)==h
    out=checked_output(ROOT/'outputs/cheap_ranker_decode'/new_job_id());out.mkdir(parents=True)
    protocol={'parent':str(PARENT.relative_to(ROOT)),'diagnostic':str(diagnostic.relative_to(ROOT)),
       'arms':{'original':.9,'dedupe':.5},'top_k':5,'new_fits':0,'downloads':0,'vlm_calls':0,
       'new_encodings':0,'scope':'training-source OOF development; no validation/fresh evaluation',
       'choice':'.5 is the existing generic nms_dedupe default, fixed once; no threshold sweep',
       'stop':'one zero-fit contrast; do not promote if short stays below E4 or middle/long mAP drops over 1pp versus original',
       'bootstrap':{'seed':12,'resamples':5000},'source_sha256':sha256_file(__file__),
       'scores_sha256':d['score_sha256'],'candidate_sha256':sha256_file(PARENT/'candidate_tables.npz'),
       'dependency_sha256':{n:sha256_file(ROOT/n) for n in ['queryseek/retrieval/fusion.py',
         'queryseek/retrieval/candidate_ranker.py','queryseek/eval/qvhighlights.py','scripts/audit_ranker_bottleneck.py']}}
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json_atomic(out/'protocol.json',protocol)
    items=[r for r in read(PARENT/'candidate_manifest.json') if r['split']=='train']
    truth={r['entry']['qid']:r['entry'] for r in rows(ROOT/p['expanded_data']/'feature_records.jsonl') if r['split']=='train'}
    with np.load(PARENT/'candidate_tables.npz',allow_pickle=False) as z:
        arrays={f'{kind}_{r["qid"]}':z[f'union_{kind}_{r["qid"]}'] for r in items for kind in ('w','y')}
    weights=source_equal_weights([r['source'] for r in items],[len(arrays[f'w_{r["qid"]}']) for r in items])
    mass={name:{'positive_mass':0.,'utility_mass':0.,'zero_prediction_squared_loss_mass':0.} for name in ('short','middle','long')}
    prevalence={name:{'candidates':0,'positives':0} for name in ('short','middle','long')}
    with np.load(diagnostic/'oof_scores.npz',allow_pickle=False) as z:scores={int(q):z[q] for q in z.files}
    pred={name:{} for name in protocol['arms']};offset=0
    for r in items:
        q=r['qid'];w=arrays[f'w_{q}'];y=arrays[f'y_{q}'];ww=weights[offset:offset+len(w)];offset+=len(w)
        ov=overlaps(w,truth[q]['relevant_windows']);gtlength=np.array([b-a for a,b in truth[q]['relevant_windows']])[ov.argmax(axis=1)]
        for name,lo,hi in [('short',0,10),('middle',10,30),('long',30,float('inf'))]:
            mask=(y>0)&(gtlength>lo)&(gtlength<=hi)
            mass[name]['positive_mass']+=float(ww[mask].sum())
            mass[name]['utility_mass']+=float((ww[mask]*y[mask]).sum())
            mass[name]['zero_prediction_squared_loss_mass']+=float((ww[mask]*y[mask]**2).sum())
            bucket=(w[:,1]-w[:,0]>lo)&(w[:,1]-w[:,0]<=hi)
            prevalence[name]['candidates']+=int(bucket.sum());prevalence[name]['positives']+=int((bucket&(y>0)).sum())
        for name,threshold in protocol['arms'].items():pred[name][q]=decode(w,scores[q],threshold)
    assert {str(q):[list(w) for w in ws] for q,ws in pred['original'].items()}==read(PARENT/'oof_full_predictions.json')
    for name,pp in pred.items():write_json_atomic(out/(name+'_predictions.json'),pp)
    results={name:dict(eval_mr_brief(pp,truth)) for name,pp in pred.items()}
    diagnostic_result={}
    pool={r['qid']:arrays[f'w_{r["qid"]}'].tolist() for r in items}
    for name,pp in pred.items():
        red=[matched_redundancy(pp[q],truth[q]['relevant_windows']) for q in truth]
        diagnostic_result[name]={**candidate_diagnostic(pool,pp,truth),
              'duplicate_matches':sum(x['duplicate_matches'] for x in red)}
    e4={int(q):v for q,v in read(PARENT/'oof_e4_predictions.json').items()}
    compare=paired_clusters(per_query(pred['original'],truth),per_query(pred['dedupe'],truth),**protocol['bootstrap'])
    compare['scope']='conditional training-source OOF development, not independent confirmation'
    for field in list(next(iter(mass.values()))):
        total=sum(mass[k][field] for k in mass)
        for k in mass:mass[k][field+'_share']=mass[k][field]/total
    e4_metrics=dict(eval_mr_brief(e4,truth))
    accepted=(results['dedupe']['MR-short-mAP']>=e4_metrics['MR-short-mAP'] and
          all(results['dedupe'][f'MR-{s}-mAP']>=results['original'][f'MR-{s}-mAP']-1. for s in ('full','middle','long')))
    result={'results':results,'e4':e4_metrics,'contrast':compare,
            'diagnostics':diagnostic_result,'training_mass':mass,'candidate_length_prevalence':prevalence,
            'passes_predeclared_gate':accepted,'default_promotion':False,'new_fits':0,
            'new_encodings':0,'vlm_calls':0,'elapsed_sec':time.perf_counter()-start}
    write_json_atomic(out/'metrics.json',result)
    print('OUTPUT='+str(out));print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostic',type=Path,required=True);args=ap.parse_args()
    run(checked_output(args.diagnostic))
