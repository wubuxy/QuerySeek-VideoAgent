"""Posthoc decoder interchange; frozen checkpoints, unchanged E4 curve rules.

This is development diagnosis on already observed internal validation, not
independent evaluation, a new training ablation, or an automatic method choice.
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from scripts.localizer_study import verify, read, rows, load_records, make_model, variant_control
from scripts.analyze_localizer_study import per_query, paired_clusters, candidate_diagnostic
from queryseek.retrieval.univtg_adapter import study_input
from queryseek.retrieval.curve_spans import curve_spans, select_curve_windows
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.cache import write_json_atomic
from queryseek.tools.export import new_job_id
from scripts.sampling_probe import checked_output


def curve_predictions(record,scores):
    pool=curve_spans(record['times'],np.asarray(scores),record['duration_sec'])
    selected=select_curve_windows(pool,5,.9)
    convert=lambda windows:[(w.start_sec,w.end_sec,w.score) for w in windows]
    return convert(selected),convert(pool)


def replay(run,out):
    protocol=verify(run); config=protocol['config']
    if read(run/'status.json')['phase']!='complete': raise ValueError('Completed study required')
    out=checked_output(out); out.mkdir(parents=True,exist_ok=False)
    write_json_atomic(out/'protocol.json',{
        'posthoc':True,'independent':False,'internal_validation_previously_observed':True,
        'hypothesis':'relevance scores may separate labeled points while point boundary decoding/ranking fails',
        'checkpoints':'reuse exactly the already selected point-decoder checkpoints, no curve-based epoch selection',
        'curve_rules':'original quantiles .5/.7/.85, smoothing .25/.5/.25, duration support, max120sec, NMS.9/top5',
        'scales':config['train_scales'],'variants':config['variants'],
        'new_training_updates':0,'new_image_encodings':0,'new_vlm_calls':0,
        'source_sha256':{'script':sha256_file(__file__),'curve_rules':sha256_file(ROOT/'queryseek/retrieval/curve_spans.py'),
                         'study_protocol':sha256_file(run/'protocol.json'),'study_metrics':sha256_file(run/'metrics.json')},
        'decision':'diagnostic only, default E4 unchanged'})
    torch.set_num_threads(config['torch_threads']); torch.use_deterministic_algorithms(True)
    records=load_records(run,protocol); validation=[r for r in records if r['split']=='validation']
    gt={e['qid']:e for e in rows(run/'validation.jsonl')}
    convert=lambda path:{int(q):v for q,v in read(path).items()}
    baseline=convert(run/'e4_predictions.json'); baseline_rows=per_query(baseline,gt)
    result={'posthoc':True,'independent':False,'e4_metrics':dict(eval_mr_brief(baseline,gt)),
            'new_training_updates':0,'new_image_encodings':0,'new_vlm_calls':0,'models':{}}
    for n in config['train_scales']:
        for variant in config['variants']:
            folder=run/'models'/str(n)/variant
            metrics=read(folder/'metrics.json'); checkpoint=folder/'selected.pt'
            if sha256_file(checkpoint)!=metrics['checkpoint_sha256']: raise ValueError('Checkpoint changed')
            model=make_model(variant,validation[0]['images'].shape[1],config)
            model.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict']); model.eval()
            predictions={}; pools={}
            with torch.inference_mode():
                for record in validation:
                    logits,_=model(study_input(record,variant_control(variant)))
                    predictions[record['entry']['qid']],pools[record['entry']['qid']]=curve_predictions(record,logits.numpy())
            for qid in gt: predictions.setdefault(qid,[])
            original=convert(folder/'predictions.json')
            curve_rows=per_query(predictions,gt)
            key=f'{n}/{variant}'
            result['models'][key]={'point_metrics':metrics['validation_metrics'],
                 'curve_metrics':dict(eval_mr_brief(predictions,gt)),
                 'curve_diagnostic':candidate_diagnostic(pools,predictions,gt),
                 'paired_curve_vs_point':paired_clusters(per_query(original,gt),curve_rows),
                 'paired_curve_vs_e4':paired_clusters(baseline_rows,curve_rows),
                 'checkpoint_sha256':sha256_file(checkpoint),'selected_epoch':metrics['selected_epoch']}
            write_json_atomic(out/(str(n)+'_'+variant+'_predictions.json'),predictions)
            write_json_atomic(out/'metrics.json',result)
            print(key,'point_mAP',metrics['validation_metrics']['MR-full-mAP'],
                  'curve_mAP',result['models'][key]['curve_metrics']['MR-full-mAP'],flush=True)
    verify(run)
    print('OUTPUT='+str(out),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    args=parser.parse_args(); run=checked_output(args.run)
    replay(run,run/'decoder_interchange'/new_job_id())
