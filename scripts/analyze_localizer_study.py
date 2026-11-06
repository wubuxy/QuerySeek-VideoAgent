"""Read-only replay, candidate diagnosis and source-cluster paired estimates.

Writes only a new analysis directory; no API calls, image encoding or training.
Internal validation selected checkpoints: confidence intervals do not remove
selection bias and must not be presented as independent generalization claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from scripts.localizer_study import verify, load_records, make_model, variant_control, evaluate, read, rows
from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output
from queryseek.retrieval.univtg_adapter import study_input
from queryseek.retrieval.localizer import decode_spans
from queryseek.retrieval.curve_spans import curve_spans
from queryseek.eval.qvhighlights import compute_ap, temporal_iou, eval_mr_brief
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.tools.export import new_job_id


def window_score_margin(scores,times,windows):
    """Posthoc benchmark-window separation, not certified semantic negatives."""
    scores=np.asarray(scores,dtype=float); times=np.asarray(times,dtype=float)
    if scores.shape!=times.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid diagnostic score timeline')
    mask=np.zeros(len(times),dtype=bool)
    for a,b in windows: mask |= (times>=a)&(times<b)
    if not mask.any() or mask.all(): return None
    return float(scores[mask].mean()-scores[~mask].mean())


def margins_summary(values):
    good=[x for x in values if x is not None]
    return {'eligible_queries':len(good),'planned_queries':len(values),
            'positive_margins':sum(x>0 for x in good),
            'mean_margin':float(np.mean(good)) if good else None,
            'scope':'posthoc labeled-window versus unannotated-window, not action accuracy'}


def per_query(predictions, gt):
    result = {}
    for qid, entry in gt.items():
        windows = [tuple(w) for w in entry['relevant_windows']]
        pp = predictions.get(qid,[])
        hit = max((temporal_iou(pp[0][0],pp[0][1],a,b) for a,b in windows),default=0.) if pp else 0.
        result[qid] = {'source':source_video_id(entry['vid']),
                      'ap':float(compute_ap(windows,pp).mean()),'r1_05':float(hit>=.5)}
    return result


def paired_clusters(first, second, seed=12, resamples=5000):
    if set(first) != set(second): raise ValueError('Paired query sets differ')
    groups = defaultdict(list)
    for qid, row in first.items():
        if row['source'] != second[qid]['source']: raise ValueError('Source identity mismatch')
        groups[row['source']].append(qid)
    sums = np.array([[sum(second[q]['ap']-first[q]['ap'] for q in ids),
                      sum(second[q]['r1_05']-first[q]['r1_05'] for q in ids),len(ids)]
                     for source,ids in sorted(groups.items())])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0,len(sums),(resamples,len(sums)))
    totals = sums[draws].sum(axis=1)
    delta = sums[:,:2].sum(axis=0)/sums[:,2].sum()*100
    intervals = np.quantile(totals[:,:2]/totals[:,2,None]*100,[.025,.975],axis=0)
    return {'mAP_delta_pp':float(delta[0]),'mAP_ci95_pp':intervals[:,0].tolist(),
            'r1_05_delta_pp':float(delta[1]),'r1_05_ci95_pp':intervals[:,1].tolist(),
            'source_clusters':len(groups),'queries':len(first),'resamples':resamples,'seed':seed,
            'scope':'conditional internal validation, checkpoint selection bias remains'}


def candidate_diagnostic(pool, predictions, gt):
    query_hits=0; short_total=0; short_pool=0; short_top5=0; top5_queries=0
    for qid,entry in gt.items():
        def hit(windows,a,b):
            return any(temporal_iou(s,e,a,b)>=.5 for s,e,*_ in windows)
        ws = entry['relevant_windows']
        query_hits += any(hit(pool.get(qid,[]),a,b) for a,b in ws)
        top5_queries += any(hit(predictions.get(qid,[]),a,b) for a,b in ws)
        for a,b in ws:
            if b-a <= 10:
                short_total += 1
                short_pool += hit(pool.get(qid,[]),a,b)
                short_top5 += hit(predictions.get(qid,[]),a,b)
    return {'all_candidate_query_oracle_hits':query_hits,'top5_query_oracle_hits':top5_queries,
            'queries':len(gt),'short_gt':short_total,'short_pool_hits':short_pool,'short_top5_hits':short_top5}


def analyze(run,out):
    protocol = verify(run); config = protocol['config']; cfg = protocol['feature_config']
    if read(run/'status.json')['phase'] != 'complete': raise ValueError('Completed study required')
    report = read(run/'metrics.json')
    torch.set_num_threads(config['torch_threads']); torch.use_deterministic_algorithms(True)
    records = load_records(run,protocol); val = [r for r in records if r['split']=='validation']
    gt = {e['qid']:e for e in rows(run/'validation.jsonl')}
    def prediction_file(path): return {int(q):v for q,v in read(path).items()}
    baseline = prediction_file(run/'e4_predictions.json')
    if dict(eval_mr_brief(baseline,gt)) != report['e4_same_features']: raise ValueError('E4 metric replay mismatch')
    baseline_rows = per_query(baseline,gt)
    baseline_pool = {r['entry']['qid']:[(w.start_sec,w.end_sec,w.score) for w in
                      curve_spans(r['times'],r['images']@r['query_embedding'],r['duration_sec'])] for r in val}
    result = {'independent':False,'new_vlm_calls':0,'new_image_encodings':0,'training_updates':0,
              'e4_diagnostic':candidate_diagnostic(baseline_pool,baseline,gt),'models':{},'scale_pairs':{},
              'source_hashes':{'protocol.json':sha256_file(run/'protocol.json'),
                              'metrics.json':sha256_file(run/'metrics.json'),
                              'feature_records.jsonl':sha256_file(run/'feature_records.jsonl')}}
    result['analyzer_sha256']=sha256_file(__file__)
    result['posthoc_clip_window_margin']=margins_summary([window_score_margin(
            r['images']@r['query_embedding'],r['times'],r['entry']['relevant_windows']) for r in val])
    query_rows = {}
    for n in config['train_scales']:
        for variant in config['variants']:
            directory = run/'models'/str(n)/variant
            checkpoint = directory/'selected.pt'; metrics = read(directory/'metrics.json')
            if sha256_file(checkpoint) != metrics['checkpoint_sha256']: raise ValueError('Checkpoint corruption')
            saved = torch.load(checkpoint,weights_only=True)
            if saved['config'] != config or saved['variant'] != variant: raise ValueError('Checkpoint identity mismatch')
            model = make_model(variant,val[0]['images'].shape[1],config)
            model.load_state_dict(saved['state_dict']); model.eval()
            replay, pp, diagnostic = evaluate(model,val,variant_control(variant),cfg,gt)
            original = prediction_file(directory/'predictions.json')
            if replay != metrics['validation_metrics'] or set(pp)!=set(original):
                raise ValueError('Model metric replay mismatch')
            for qid in pp:
                if len(pp[qid]) != len(original[qid]) or not np.allclose(pp[qid],original[qid],atol=1e-6,rtol=1e-6):
                    raise ValueError('Prediction replay mismatch')
            pool = {}; logit_margins=[]; projection_margins=[]
            with torch.inference_mode():
                for record in val:
                    logits, distances = model(study_input(record,variant_control(variant)))
                    logit_margins.append(window_score_margin(logits.cpu().numpy(),record['times'],record['entry']['relevant_windows']))
                    if variant=='univtg':
                        x=study_input(record); dim=model.embedding_dim
                        video=torch.cat([x[:,:dim],x[:,-2:]],-1).unsqueeze(0)
                        query=x[:1,dim:2*dim].unsqueeze(0)
                        upstream=model.network(query,torch.ones((1,1)),video,torch.ones(video.shape[:2]))
                        projection_margins.append(window_score_margin(upstream['saliency_scores'][0].numpy(),record['times'],record['entry']['relevant_windows']))
                    windows = decode_spans(record['times'],record['duration_sec'],logits,distances,
                                           len(record['times']),1.,cfg['min_span_sec'])
                    pool[record['entry']['qid']] = [(w.start_sec,w.end_sec,w.score) for w in windows]
            key = f'{n}/{variant}'
            query_rows[key] = per_query(pp,gt)
            result['models'][key] = {'metrics':replay,'diagnostic':candidate_diagnostic(pool,pp,gt),
                 'paired_vs_e4':paired_clusters(baseline_rows,query_rows[key],config['bootstrap_seed'],config['bootstrap_resamples']),
                 'parameter_count':metrics['parameter_count'],'selected_epoch':metrics['selected_epoch'],
                 'prediction_replay_exact_within_tolerance':True}
            result['models'][key]['posthoc_relevance_logit_margin']=margins_summary(logit_margins)
            if projection_margins:
                result['models'][key]['posthoc_projected_cosine_margin']=margins_summary(projection_margins)
            result['source_hashes'][checkpoint.relative_to(run).as_posix()] = sha256_file(checkpoint)
    if len(config['train_scales'])==2:
        a,b = config['train_scales']
        for variant in config['variants']:
            result['scale_pairs'][variant] = paired_clusters(query_rows[f'{a}/{variant}'],query_rows[f'{b}/{variant}'],
                                                       config['bootstrap_seed'],config['bootstrap_resamples'])
    out = checked_output(out); out.mkdir(parents=True,exist_ok=False)
    write_json_atomic(out/'diagnostic.json',result)
    verify(run)
    print('OUTPUT='+str(out),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    args = parser.parse_args(); run = checked_output(args.run)
    analyze(run,run/'analysis'/new_job_id())
