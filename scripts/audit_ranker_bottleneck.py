"""Training-source OOF diagnosis only: reuse cached candidates and saved models."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import multiscale_ranker_study as prior
from scripts.localizer_study import read,rows
from scripts.sampling_probe import checked_output
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.retrieval.candidate_ranker import rank_candidates,source_equal_weights
from queryseek.eval.qvhighlights import temporal_iou
from queryseek.tools.export import new_job_id

PARENT=ROOT/'outputs/multiscale_ranker/job_20261015_132144_9118b9'

def overlaps(windows,gt):
    return np.array([[temporal_iou(a,b,c,d) for c,d in gt] for a,b,*_ in windows],dtype=float)

def event_rows(windows,scores,gt,selected):
    ov=overlaps(windows,gt); order=np.argsort(-scores,kind='stable')
    short_order=order[(windows[order,1]-windows[order,0])<=10.]
    chosen=overlaps(selected,gt) if selected else np.empty((0,len(gt)))
    result=[]
    for j,(a,b) in enumerate(gt):
        good=ov[:,j]>=.5
        def rank(indices):
            hit=np.flatnonzero(good[indices])
            return int(hit[0]+1) if len(hit) else None
        result.append({'duration':b-a,'pool_hit':bool(good.any()),'raw_rank':rank(order),
                       'within_short_rank':rank(short_order),'top5_hit':bool((chosen[:,j]>=.5).any())})
    return result

def matched_redundancy(selected,gt):
    if not selected:return {'matched':0,'distinct_events':0,'duplicate_matches':0}
    ov=overlaps(selected,gt); good=ov.max(axis=1)>=.5
    targets=ov.argmax(axis=1)[good]
    return {'matched':int(good.sum()),'distinct_events':len(set(targets.tolist())),
            'duplicate_matches':int(good.sum())-len(set(targets.tolist()))}

def main():
    started=time.perf_counter(); p=prior.verify(PARENT)
    saved=read(PARENT/'metrics.json')
    for name,h in saved['artifact_sha256'].items():
        if sha256_file(PARENT/name)!=h:raise ValueError('Parent changed: '+name)
    out=checked_output(ROOT/'outputs/ranker_bottleneck'/new_job_id());out.mkdir(parents=True)
    items=[r for r in read(PARENT/'candidate_manifest.json') if r['split']=='train']
    qids={r['qid'] for r in items}
    truth={r['entry']['qid']:r['entry'] for r in rows(ROOT/p['expanded_data']/'feature_records.jsonl')
           if r['split']=='train'}
    folds=read(PARENT/'folds.json')
    with np.load(PARENT/'candidate_tables.npz',allow_pickle=False) as z:
        arrays={k:z[k] for q in qids for k in (f'union_w_{q}',f'union_x_{q}',f'union_y_{q}')}
    scores={}
    with threadpool_limits(limits=4):
        for i,fold in enumerate(folds):
            train=set(fold['train_qids']);test=set(fold['test_qids'])
            lookup={r['qid']:r['source'] for r in items}
            assert not {lookup[q] for q in train}&{lookup[q] for q in test}
            m=joblib.load(PARENT/'models'/f'oof_full_{i}.joblib')
            for q in test:scores[q]=m.predict(arrays[f'union_x_{q}'])
    np.savez(out/'oof_scores.npz',**{str(q):s for q,s in scores.items()})
    historical=read(PARENT/'oof_full_predictions.json')
    weights=source_equal_weights([r['source'] for r in items],[len(arrays[f'union_w_{r["qid"]}']) for r in items])
    offset=0;mass={'short':0.,'middle':0.,'long':0.}; events=[];redundancy=[];positive=0.;total=0.
    for r in items:
        q=r['qid'];w=arrays[f'union_w_{q}'];s=scores[q];gt=truth[q]['relevant_windows']
        pp=rank_candidates(w,s,5,.9)
        assert [list(x) for x in pp]==historical[str(q)],'Original OOF not reproduced'
        yy=arrays[f'union_y_{q}'];ww=weights[offset:offset+len(w)];offset+=len(w)
        ov=overlaps(w,gt); assignments=ov.argmax(axis=1); lengths=np.array([b-a for a,b in gt])
        for name,lo,hi in [('short',0,10),('middle',10,30),('long',30,float('inf'))]:
            mask=(yy>0)&(lengths[assignments]>lo)&(lengths[assignments]<=hi)
            mass[name]+=float(ww[mask].sum())
        positive+=float(ww[yy>0].sum());total+=float(ww.sum())
        events += [{'qid':q,**e} for e in event_rows(w,s,gt,pp)]
        redundancy.append(matched_redundancy(pp,gt))
    summary={}
    for name,lo,hi in [('short',0,10),('middle',10,30),('long',30,float('inf'))]:
        rr=[r for r in events if lo<r['duration']<=hi]
        ranks=[r['raw_rank'] for r in rr if r['raw_rank'] is not None]
        summary[name]={'events':len(rr),'positive_weight_share':mass[name]/positive,
                       'pool_hits':sum(r['pool_hit'] for r in rr),'top5_hits':sum(r['top5_hit'] for r in rr),
                       'raw_top5_hits':sum(r['raw_rank'] is not None and r['raw_rank']<=5 for r in rr),
                       'within_short_top5_hits':sum(r['within_short_rank'] is not None and r['within_short_rank']<=5 for r in rr),
                       'median_raw_rank':float(np.median(ranks)) if ranks else None}
    result={'scope':'posthoc training-source OOF diagnosis, not independent validation',
            'parent':str(PARENT.relative_to(ROOT)),'queries':len(items),'sources':len({r['source'] for r in items}),
            'by_event_length':summary,'foreground_weight_fraction':positive/total,
            'top5_redundancy':{k:sum(r[k] for r in redundancy) for k in redundancy[0]},
            'queries_with_duplicate_matches':sum(r['duplicate_matches']>0 for r in redundancy),
            'new_fits':0,'new_encodings':0,'downloads':0,'vlm_calls':0,'prior_oof_exact':True,
            'elapsed_sec':time.perf_counter()-started,'source_sha256':sha256_file(__file__),
            'input_sha256':{str(f.relative_to(ROOT)):sha256_file(f) for f in
              [PARENT/'candidate_tables.npz',PARENT/'folds.json',PARENT/'metrics.json']},
            'score_sha256':sha256_file(out/'oof_scores.npz')}
    write_json_atomic(out/'events.json',events);write_json_atomic(out/'metrics.json',result)
    print('OUTPUT='+str(out));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
