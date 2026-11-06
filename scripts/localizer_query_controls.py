"""Frozen retraining controls; reuse features and full-model last checkpoints.

No encoder, downloads, API calls or validation-driven checkpoint selection.
All inference candidates are constructed without access to ground truth.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.localizer import supervision, localization_loss, decode_spans, validate_timeline
from queryseek.retrieval.univtg_adapter import study_input
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.tools.export import new_job_id
from scripts.localizer_study import read, rows, verify, load_records, make_model, parameter_count, study_mock_examples
from scripts.make_qvhighlights_subset import source_video_id
from scripts.replay_localizer_curves import curve_predictions
from scripts.analyze_localizer_study import per_query, paired_clusters, candidate_diagnostic
from scripts.sampling_probe import checked_output


def file_predictions(path):
    return {int(q):v for q,v in read(path).items()}


def audit_study(study):
    """Check actual bytes, timelines, selections and metric reconstruction."""
    protocol = verify(study)
    if read(study/'status.json')['phase'] != 'complete':
        raise ValueError('Completed study required')
    for name, sha in protocol['source_sha256'].items():
        if sha256_file(study/'source_snapshot'/name) != sha:
            raise ValueError('Frozen source snapshot changed')
    records = load_records(study, protocol)
    videos = {}; visual = {}
    for r in records:
        videos[r['video']] = r['video_sha256']
        visual[r['images_path']] = r
        if not np.isclose(np.linalg.norm(r['query_embedding']), 1., atol=1e-5):
            raise ValueError('Query normalization changed')
    for path, sha in videos.items():
        if sha256_file(path) != sha:
            raise ValueError('Video bytes changed')
    for r in visual.values():
        validate_timeline(r['times'], r['duration_sec'])
        if not np.allclose(np.linalg.norm(r['images'], axis=1), 1., atol=1e-5):
            raise ValueError('Visual normalization changed')
        meta = read(Path(r['images_path']).parent/'feature_meta.json')
        signature = meta['signature']
        if signature['video_sha256'] != r['video_sha256']:
            raise ValueError('Feature/video identity changed')
    gt = {e['qid']:e for e in rows(study/'validation.jsonl')}
    report = read(study/'metrics.json')
    if dict(eval_mr_brief(file_predictions(study/'e4_predictions.json'), gt)) != report['e4_same_features']:
        raise ValueError('E4 metric mismatch')
    fits = 0
    for scale, sr in report['scales'].items():
        for variant, result in sr['variants'].items():
            folder = study/'models'/scale/variant
            if sha256_file(folder/'selected.pt') != result['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed')
            history = read(folder/'history.json')
            if len(history) != 80 or [r['epoch'] for r in history] != list(range(1,81)):
                raise ValueError('Training budget mismatch')
            eligible = [r for r in history if 'validation_metrics' in r]
            best = max(eligible, key=lambda r:r['validation_metrics']['MR-full-mAP'])
            if best['epoch'] != result['selected_epoch']:
                raise ValueError('Historical checkpoint rule mismatch')
            for filename, expected_epoch in [('selected.pt', result['selected_epoch']), ('last.pt',80)]:
                checkpoint = torch.load(folder/filename, weights_only=True)
                if checkpoint['epoch'] != expected_epoch:
                    raise ValueError('Checkpoint epoch mismatch')
            if dict(eval_mr_brief(file_predictions(folder/'predictions.json'),gt)) != result['validation_metrics']:
                raise ValueError('Saved prediction metric mismatch')
            fits += 1
    train_sources = {source_video_id(r['entry']['vid']) for r in records if r['split']=='train'}
    val_sources = {source_video_id(r['entry']['vid']) for r in records if r['split']=='validation'}
    if train_sources & val_sources:
        raise ValueError('Source leakage')
    return {'fits_checked':fits, 'feature_queries':len(records), 'video_crops':len(videos),
            'unique_frames':sum(len(r['times']) for r in visual.values()),
            'train_sources':len(train_sources),'validation_sources':len(val_sources),
            'source_overlap':0,'e4_metrics_reconstructed':True,
            'sha_normalization_pts_checked':True}, records


def frozen_inventory(study, protocol):
    inventory = {str(p.relative_to(ROOT)):sha256_file(p) for p in study.rglob('*') if p.is_file()}
    inventory.update(protocol['preserved_artifacts_sha256'])
    inventory['data/qvhighlights_fresh80/selection.json'] = protocol['fresh_sha256']
    return inventory


def freeze(config, out):
    study = checked_output(ROOT/config['study_run'])
    audit, _ = audit_study(study)
    parent = verify(study)
    if config['scale'] not in parent['config']['train_scales'] or parent['config']['epochs'] != 80:
        raise ValueError('Wrong source scale or training budget')
    out = checked_output(out); out.mkdir(parents=True, exist_ok=False)
    source_files = ['scripts/localizer_query_controls.py', 'configs/localizer_query_controls_v1.json',
                    'scripts/replay_localizer_curves.py', 'scripts/analyze_localizer_study.py',
                    'queryseek/retrieval/curve_spans.py', 'queryseek/retrieval/localizer.py',
                    'queryseek/retrieval/univtg_adapter.py', 'scripts/localizer_study.py',
                    'queryseek/eval/qvhighlights.py']
    code = {}
    for name in source_files:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target); code[name] = sha256_file(target)
    protocol = {'config':config, 'training_config':parent['config'], 'feature_config':parent['feature_config'],
                'source_sha256':code,'preserved_sha256':frozen_inventory(study,parent),
                'study_protocol_sha256':sha256_file(study/'protocol.json'),
                'checkpoint_selected_on_validation':False,'internal_validation_previously_observed':True,
                'independent':False,'mock':False,'pretrained_localizer':False,
                'new_image_encodings':0,'vlm_calls':0,'audit':audit}
    write_json_atomic(out/'protocol.json',protocol)
    write_json_atomic(out/'status.json',{'phase':'frozen','training_started':False})
    print('AUDIT='+json.dumps(audit),flush=True)
    print('OUTPUT='+str(out),flush=True)
    return out


def verify_controls(out):
    protocol = read(out/'protocol.json')
    for group in ('source_sha256','preserved_sha256'):
        for name, sha in protocol[group].items():
            if sha256_file(ROOT/name) != sha:
                raise ValueError('Frozen input/code changed: '+name)
    for name,sha in protocol['source_sha256'].items():
        if sha256_file(out/'source_snapshot'/name) != sha:
            raise ValueError('Control source snapshot changed: '+name)
    verify(checked_output(ROOT/protocol['config']['study_run']))
    return protocol


def infer(model, records, control, cfg):
    """GT-free inference. Entry/query labels are not passed to the model/decoders."""
    model.eval(); point = {}; curve = {}; pools = {}
    with torch.inference_mode():
        for r in records:
            x = study_input(r,control)
            logits, distances = model(x)
            if not torch.isfinite(logits).all() or not torch.isfinite(distances).all():
                raise ValueError('Nonfinite inference')
            qid = r['entry']['qid']
            curve[qid], pools[qid] = curve_predictions(r,logits.numpy())
            windows = decode_spans(r['times'],r['duration_sec'],logits,distances,
                                  cfg['top_k'],cfg['nms_iou'],cfg['min_span_sec'])
            point[qid] = [(w.start_sec,w.end_sec,w.score) for w in windows]
    return {'curve':curve,'point':point}, pools


def fit_control(train, variant, control, config, cfg, out, deadline):
    """Matched initialization/order/loss/optimizer, no validation observations."""
    if control not in ('no_query','position_only'):
        raise ValueError('Only retraining controls are authorized')
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(config['head_seed'])
    model = make_model(variant,train[0]['images'].shape[1],config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    prepared = [(study_input(r,control),*supervision(r['times'],r['duration_sec'],r['entry']['relevant_windows'])) for r in train]
    counts = Counter(source_video_id(r['entry']['vid']) for r in train)
    weights = [1/counts[source_video_id(r['entry']['vid'])] for r in train]
    rng = np.random.default_rng(config['head_seed']); history = []; started = time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        if time.perf_counter() > deadline:
            raise TimeoutError('Frozen wall-time cap exceeded; no automatic retry')
        model.train(); losses = []
        for idx in rng.permutation(len(train)):
            x,y,d = prepared[idx]; optimizer.zero_grad(set_to_none=True)
            loss,_ = localization_loss(*model(x),y,d,cfg['loss_weights'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            (loss*weights[idx]).backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Missing/nonfinite gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip']); optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch':epoch,'train_query_mean_loss':float(np.mean(losses))})
        write_json_atomic(out/'history.json',history)
        if epoch % 10 == 0: print(variant,control,epoch,'loss',history[-1]['train_query_mean_loss'],flush=True)
    torch.save({'state_dict':model.state_dict(),'epoch':config['epochs'],'variant':variant,
                'control':control,'config':config,'embedding_dim':train[0]['images'].shape[1]},out/'last.pt')
    restored = make_model(variant,train[0]['images'].shape[1],config)
    restored.load_state_dict(torch.load(out/'last.pt',weights_only=True)['state_dict'])
    model.eval(); restored.eval()
    with torch.inference_mode():
        x = study_input(train[0],control)
        exact = all(torch.equal(a,b) for a,b in zip(model(x),restored(x)))
    if not exact: raise ValueError('Checkpoint reload mismatch')
    result = {'checkpoint_sha256':sha256_file(out/'last.pt'),'trained_epochs':config['epochs'],
              'checkpoint_rule':'last epoch, no validation selection','checkpoint_reload_exact':exact,
              'parameter_count':parameter_count(model),'training_sec':time.perf_counter()-started,
              'train_sources':len(counts),'train_queries':len(train),'control':control}
    write_json_atomic(out/'fit_metrics.json',result)
    return result


def run_controls(out):
    protocol = verify_controls(out); task = protocol['config']; config = protocol['training_config']; cfg = protocol['feature_config']
    torch.set_num_threads(config['torch_threads']); torch.use_deterministic_algorithms(True)
    study = checked_output(ROOT/task['study_run']); records = load_records(study,verify(study))
    ids = {e['qid'] for e in rows(study/f'train_{task["scale"]}.jsonl')}
    train = [r for r in records if r['split']=='train' and r['entry']['qid'] in ids]
    val = [r for r in records if r['split']=='validation']
    gt = {e['qid']:e for e in rows(study/'validation.jsonl')}
    started = time.perf_counter(); deadline = started+task['cost_cap_sec']
    write_json_atomic(out/'status.json',{'phase':'training','training_started':True})
    result = {'mock':False,'independent':False,'checkpoint_selected_on_validation':False,
              'internal_validation_previously_observed':True,'new_image_encodings':0,'vlm_calls':0,
              'train_sources':len({source_video_id(r['entry']['vid']) for r in train}),
              'train_queries':len(train),'validation_queries':len(gt),'validation_available_queries':len(val),
              'e4_metrics':dict(eval_mr_brief(file_predictions(study/'e4_predictions.json'),gt)),
              'arms':{},'contrasts':{}}
    arm_predictions = {}
    try:
        for variant in task['architectures']:
            for control in ['full']+task['controls']:
                key = variant+'_'+control; folder = out/'models'/key
                if control == 'full':
                    checkpoint_path = study/'models'/str(task['scale'])/variant/'last.pt'
                    fit = {'new_training':False,'trained_epochs':80,'checkpoint_sha256':sha256_file(checkpoint_path)}
                else:
                    if (folder/'fit_metrics.json').exists():
                        fit = read(folder/'fit_metrics.json')
                        if sha256_file(folder/'last.pt') != fit['checkpoint_sha256']: raise ValueError('Control checkpoint changed')
                    else:
                        fit = fit_control(train,variant,control,config,cfg,folder,deadline)
                    checkpoint_path = folder/'last.pt'
                checkpoint = torch.load(checkpoint_path,weights_only=True)
                if checkpoint['epoch'] != config['epochs']: raise ValueError('Wrong checkpoint epoch')
                model = make_model(variant,train[0]['images'].shape[1],config)
                model.load_state_dict(checkpoint['state_dict']); model.eval()
                predictions, pools = infer(model,val,control,cfg)
                arm = {'fit':fit,'validation_metrics':{},'candidate_diagnostic':{}}
                for decoder, pp in predictions.items():
                    for qid in gt: pp.setdefault(qid,[])
                    arm['validation_metrics'][decoder] = dict(eval_mr_brief(pp,gt))
                    write_json_atomic(out/(key+'_'+decoder+'_predictions.json'),pp)
                arm['candidate_diagnostic']['curve'] = candidate_diagnostic(pools,predictions['curve'],gt)
                result['arms'][key] = arm; arm_predictions[key] = predictions
                write_json_atomic(out/'metrics.json',result)
                print(key,'curve_mAP',arm['validation_metrics']['curve']['MR-full-mAP'],
                      'point_mAP',arm['validation_metrics']['point']['MR-full-mAP'],flush=True)
        e4 = file_predictions(study/'e4_predictions.json')
        for variant in task['architectures']:
            full = arm_predictions[variant+'_full']
            for decoder in ('curve','point'):
                full_rows = per_query(full[decoder],gt)
                for control in task['controls']:
                    result['contrasts'][variant+'/'+decoder+'/full_minus_'+control] = paired_clusters(
                        per_query(arm_predictions[variant+'_'+control][decoder],gt),full_rows,
                        config['bootstrap_seed'],config['bootstrap_resamples'])
                result['contrasts'][variant+'/'+decoder+'/full_minus_e4'] = paired_clusters(per_query(e4,gt),full_rows,
                    config['bootstrap_seed'],config['bootstrap_resamples'])
        result['training_and_evaluation_wall_sec'] = time.perf_counter()-started
        verify_controls(out)
        result['all_frozen_inputs_unchanged_after_run'] = True
        write_json_atomic(out/'metrics.json',result)
        write_json_atomic(out/'status.json',{'phase':'complete','independent':False,'new_fits':4})
    except Exception as exc:
        write_json_atomic(out/'status.json',{'phase':'failed','error_type':type(exc).__name__,'reason':str(exc)})
        raise
    print('OUTPUT='+str(out),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['freeze','run','mock'])
    parser.add_argument('--run',type=Path)
    parser.add_argument('--config',default='configs/localizer_query_controls_v1.json')
    parser.add_argument('--video',default='assets/demo/sample.mp4')
    args = parser.parse_args()
    if args.phase == 'freeze':
        freeze(read(ROOT/args.config),ROOT/'outputs/localizer_query_controls'/new_job_id())
    elif args.phase == 'run':
        if args.run is None: parser.error('--run required')
        run_controls(checked_output(args.run))
    else:
        config = {**read(ROOT/'configs/localizer_study_v2.json'),'epochs':2,'torch_threads':1,'univtg_hidden':8}
        cfg = read(ROOT/config['pilot_config']); torch.set_num_threads(1)
        records = study_mock_examples(ROOT/args.video,cfg)
        out = checked_output(ROOT/'outputs/localizer_query_controls_mock'/new_job_id()); out.mkdir(parents=True)
        report = {'mock':True,'scope':'synthetic mechanics only, not real accuracy','fits':{}}
        for variant in ('matched_mlp','matched_conv'):
            for control in ('no_query','position_only'):
                key = variant+'_'+control
                report['fits'][key] = fit_control(records[:4],variant,control,config,cfg,out/key,time.perf_counter()+60)
        write_json_atomic(out/'metrics.json',report); print('OUTPUT='+str(out),flush=True)


if __name__ == '__main__': main()
