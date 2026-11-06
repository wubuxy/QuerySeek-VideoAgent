"""Frozen nested source-scale experiment, zero VLM; each phase is resumable."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.localizer import LocalizationHead, supervision, localization_loss, decode_spans
from queryseek.retrieval.univtg_adapter import UniVTGHead, study_input
from queryseek.retrieval.curve_spans import curve_spans, select_curve_windows
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.tools.export import new_job_id
from scripts.localizer_pilot import read, rows, acquire, check_video, features, model_weights_sha, mock_examples
from scripts.make_qvhighlights_subset import source_video_id
from scripts.prepare_localizer_split import valid_localization_rows
from scripts.sampling_probe import checked_output


def nested_selection(train, validation, pilot_train, pilot_val, config, excluded):
    """Select by source identities only, retain every query in the frozen crop."""
    pools = {'train': train, 'validation': validation}
    pilots = {'train': pilot_train, 'validation': pilot_val}
    groups = {}
    for split, entries in pools.items():
        g = defaultdict(list)
        for entry in entries:
            g[source_video_id(entry['vid'])].append(entry)
        groups[split] = g
    if set(groups['train']) & set(groups['validation']) or (set(groups['train'])|set(groups['validation'])) & set(excluded):
        raise ValueError('Source isolation failed')
    seed = config['source_seed']
    def order(value):
        return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    chosen = {}
    for split in pools:
        anchors = {source_video_id(e['vid']): e['vid'] for e in pilots[split]}
        source_order = sorted(anchors, key=order)+sorted(set(groups[split])-set(anchors), key=order)
        n = max(config['train_scales']) if split == 'train' else config['validation_sources']
        if n > len(source_order):
            raise ValueError('Insufficient source groups')
        selected = []
        for source in source_order[:n]:
            crop = anchors.get(source) or min({e['vid'] for e in groups[split][source]}, key=order)
            selected += sorted([e for e in groups[split][source] if e['vid'] == crop], key=lambda e:e['qid'])
        chosen[split] = (source_order[:n], selected)
    result = {'validation': chosen['validation'][1]}
    for n in config['train_scales']:
        allowed = set(chosen['train'][0][:n])
        result[f'train_{n}'] = [e for e in chosen['train'][1] if source_video_id(e['vid']) in allowed]
    return result


def snapshot(run):
    files = ['configs/localizer_study_v1.json', 'configs/localizer_pilot_v1.json',
             'scripts/localizer_study.py', 'scripts/localizer_pilot.py',
             'queryseek/retrieval/localizer.py', 'queryseek/retrieval/univtg_adapter.py',
             'queryseek/ingest/frames.py', 'queryseek/embeddings/openclip_encoder.py',
             'queryseek/eval/qvhighlights.py']
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/'queryseek/vendor/univtg').glob('*.py')]
    files += ['queryseek/vendor/univtg/LICENSE']
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/'configs').glob('localizer_study_v2*.json')]
    hashes = {}
    for name in files:
        target = run/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        hashes[name] = sha256_file(ROOT/name)
    return hashes


def freeze(config, run):
    data = checked_output(ROOT/config['data_run'])
    prior = read(data/'protocol.json')
    fresh = ROOT/'data/qvhighlights_fresh80/selection.json'
    fresh_sha = prior['identity_input_hashes']['fresh/qvhighlights_fresh80/selection.json']
    if sha256_file(fresh) != fresh_sha:
        raise ValueError('Frozen fresh identity selection changed')
    selected = nested_selection(rows(data/'train.jsonl'), rows(data/'internal_validation.jsonl'),
                                rows(data/'pilot_train.jsonl'), rows(data/'pilot_validation.jsonl'),
                                config, read(data/'excluded_sources.json'))
    pilot = read(ROOT/config['pilot_config'])
    run = checked_output(run); run.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rr in selected.items():
        _, bad = valid_localization_rows(rr)
        if bad:
            raise ValueError('Invalid official rows in selected set')
        path = run/(name+'.jsonl')
        path.write_text(''.join(json.dumps(e)+'\n' for e in rr), encoding='utf-8')
        hashes[path.name] = sha256_file(path)
    shutil.copyfile(data/'DATA_LICENSE.txt', run/'DATA_LICENSE.txt')
    summary = {name: {'queries': len(rr), 'sources': len({source_video_id(e['vid']) for e in rr}),
                      'crops': len({e['vid'] for e in rr})} for name, rr in selected.items()}
    protocol = {'config': config, 'feature_config': pilot, 'fresh_sha256': fresh_sha,
                'selection_sha256': hashes, 'source_sha256': snapshot(run), 'scope': config['scope'],
                'data_sha256': {name: sha256_file(data/name) for name in
                  ['protocol.json', 'train.jsonl', 'internal_validation.jsonl', 'excluded_sources.json']},
                'summary': summary, 'upstream_commit': '32659ac7aeba21742a63274f30eba785fc57e247',
                'sampling': 'pipeline2, 1fps, source PTS, CLIP ViT-B32/openai QuickGELU',
                'paper_reproduction': False, 'pretrained_localizer_weights': False,
                'max_crops': 320, 'max_video_bytes_total': 320*config['max_video_bytes']}
    preserved = ['outputs/baseline_integrity_probe/job_20261013_120027_d26ea7/metrics.json',
                 'outputs/baseline_integrity_probe/job_20261013_120027_d26ea7/records.jsonl',
                 'outputs/localizer_pilot/job_20261013_152613_a2429c/metrics.json',
                 'outputs/localizer_pilot/job_20261013_152613_a2429c/frame_mlp.pt',
                 'outputs/localizer_pilot/job_20261013_152613_a2429c/temporal_conv.pt']
    protocol['preserved_artifacts_sha256'] = {name:sha256_file(ROOT/name) for name in preserved}
    write_json_atomic(run/'protocol.json', protocol)
    write_json_atomic(run/'status.json', {'phase': 'frozen', 'summary': summary, 'training_started': False})
    print(json.dumps(summary), flush=True)
    return run


def verify(run, verify_code=True):
    protocol = read(run/'protocol.json')
    if sha256_file(ROOT/'data/qvhighlights_fresh80/selection.json') != protocol['fresh_sha256']:
        raise ValueError('Independent selection changed')
    for name, sha in protocol['selection_sha256'].items():
        if sha256_file(run/name) != sha:
            raise ValueError('Frozen selection changed')
    for name, sha in protocol.get('preserved_artifacts_sha256',{}).items():
        if sha256_file(ROOT/name) != sha:
            raise ValueError('Historical artifact changed')
    if verify_code:
        for name, sha in protocol['source_sha256'].items():
            if sha256_file(ROOT/name) != sha:
                raise ValueError('Experiment code changed after freeze: '+name)
    return protocol


def all_entries(run, protocol):
    return [('train', e) for e in rows(run/f'train_{max(protocol["config"]["train_scales"])}.jsonl')]+[
            ('validation', e) for e in rows(run/'validation.jsonl')]


def usable_entries(run, protocol):
    available = {r['entry']['vid']:r for r in rows(run/'availability.jsonl')}
    return [(split,e) for split,e in all_entries(run,protocol) if available[e['vid']]['available']]


def prepare(run):
    from huggingface_hub import HfApi
    protocol = verify(run)
    cfg = protocol['feature_config']; config = protocol['config']
    entries = all_entries(run, protocol)
    by_crop = {}
    for split, entry in entries:
        by_crop.setdefault(entry['vid'], []).append((split, entry))
    mirror_path = run/'mirror_index.json'
    if not mirror_path.exists():
        # Inventory is read only after the identity selection was frozen.
        selected = {}
        tree = HfApi(token=False).list_repo_tree(cfg['mirror_repo'], repo_type='dataset',
                revision=cfg['mirror_revision'], path_in_repo='FlashVTGvis/qvhighlight/videos', recursive=True)
        sizes = {}
        for item in tree:
            name = Path(item.path).name
            if name.endswith('.mp4') and name[:-4] in by_crop:
                if name in selected:
                    raise ValueError('Ambiguous mirrored crop identity')
                selected[name] = item.path; sizes[name] = item.size
        missing = sorted(set(by_crop)-{name[:-4] for name in selected})
        write_json_atomic(mirror_path, {'repo_id': cfg['mirror_repo'], 'revision': cfg['mirror_revision'],
                                       'selected': selected, 'sizes': sizes, 'missing': missing})
        if missing or any(n > config['max_video_bytes'] for n in sizes.values()):
            raise ValueError('Frozen crop unavailable or exceeds byte cap; no substitution')
    cfg = {**cfg, 'mirror_index': mirror_path.relative_to(ROOT).as_posix()}
    previous = {r['entry']['vid']:r for r in rows(run/'availability.jsonl')} if (run/'availability.jsonl').exists() else {}
    def one(vid):
        started = time.perf_counter(); split, entry = by_crop[vid][0]
        result = {'split': split, 'entry': entry, 'available': False}
        try:
            path, provenance = acquire(entry, cfg)
            result.update(check_video(entry, path, cfg))
            for _, other in by_crop[vid][1:]:
                if any(not 0 <= a < b <= result['duration_sec'] for a,b in other['relevant_windows']):
                    raise ValueError('A query exceeds actual visual duration')
            result.update(available=True, provenance=provenance)
        except Exception as exc:
            result['error_type'] = type(exc).__name__
            if isinstance(exc, ValueError): result['reason'] = str(exc)
        result['elapsed_sec'] = time.perf_counter()-started
        return result
    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        # Successful immutable local crops are reused; retries are explicit phase resumes.
        todo = [vid for vid in by_crop if not previous.get(vid, {}).get('available')]
        for future in as_completed([pool.submit(one, vid) for vid in todo]):
            result = future.result(); previous[result['entry']['vid']] = result
            write_json_atomic(run/'availability.json', list(previous.values()))
            (run/'availability.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in previous.values()), encoding='utf-8')
            print('prepared', len(previous), '/', len(by_crop), result['entry']['vid'], result['available'], flush=True)
    failures = [r['entry']['vid'] for r in previous.values() if not r['available']]
    write_json_atomic(run/'status.json', {'phase': 'prepared' if not failures else 'availability_failed',
             'planned_crops': len(by_crop), 'checked': len(previous), 'failures': failures,
             'bytes': sum(r.get('bytes', 0) for r in previous.values()), 'training_started': False})
    rejects = [r for r in previous.values() if not r['available']]
    quality_only = all(r.get('reason') in ('Ground truth exceeds actual visual duration; no clipping allowed',
               'Actual duration differs from annotation by over frozen tolerance',
               'A query exceeds actual visual duration',
               'Nonzero stream origin needs separate source-time audit') for r in rejects)
    if len(previous) != len(by_crop) or (failures and not (config.get('quality_rejections_allowed') and quality_only)):
        raise ValueError('Frozen availability gate failed; all identities retained')
    if failures:
        write_json_atomic(run/'status.json',{'phase':'prepared_with_quality_rejections','planned_crops':len(by_crop),
                    'available_crops':len(previous)-len(failures),'failures':failures,'training_started':False})


def encode(run):
    from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
    protocol = verify(run)
    cfg = protocol['feature_config']
    available = {r['entry']['vid']: r for r in rows(run/'availability.jsonl')}
    if read(run/'status.json')['phase'] not in ('prepared', 'prepared_with_quality_rejections', 'encoded', 'training', 'complete'):
        raise ValueError('Availability gate required')
    torch.set_num_threads(protocol['config']['torch_threads'])
    encoder = OpenCLIPEmbedding(cfg['model_name'], cfg['pretrained'], cfg['device'],
                               cfg['max_side'], cfg['force_quick_gelu'])
    weights_sha = model_weights_sha(encoder)
    previous = {r['entry']['qid']: r for r in rows(run/'feature_records.jsonl')} if (run/'feature_records.jsonl').exists() else {}
    started = time.perf_counter(); new_frames = 0
    for split, entry in usable_entries(run, protocol):
        qid = entry['qid']
        if qid in previous:
            continue
        record = {**available[entry['vid']], 'split': split, 'entry': entry}
        if sha256_file(record['video']) != record['video_sha256']:
            raise ValueError('Prepared video changed')
        item = features(record, cfg, encoder, weights_sha)
        new_frames += 0 if item['feature_cache_hit'] else item['n_frames']
        # Persist cache paths/SHAs so training needs no encoder or image I/O.
        images_path = ROOT/'outputs/localizer_feature_cache'/item['cache_key']/'features.npz'
        ts = {'query': entry['query'], 'model_id': encoder.model_id, 'weights_sha256': weights_sha,
              'open_clip_version': read(images_path.parent/'feature_meta.json')['signature']['open_clip_version']}
        key = hashlib.sha256(json.dumps(ts, sort_keys=True).encode()).hexdigest()
        text_path = ROOT/'outputs/localizer_text_cache'/(key+'.npy')
        metadata = {k:v for k,v in item.items() if k not in ('images', 'query_embedding', 'times')}
        metadata.update(images_path=str(images_path), images_sha256=sha256_file(images_path),
                        text_path=str(text_path), text_sha256=sha256_file(text_path))
        previous[qid] = metadata
        with (run/'feature_records.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(metadata)+'\n')
        print('encoded', len(previous), '/', len(usable_entries(run, protocol)), 'new_frames', new_frames, flush=True)
    write_json_atomic(run/'features_summary.json', {'queries': len(previous), 'new_frames_this_invocation': new_frames,
                'elapsed_sec_this_invocation': time.perf_counter()-started, 'encoder_training': False, 'vlm_calls': 0,
                'feature_records_sha256': sha256_file(run/'feature_records.jsonl'), 'weights_sha256': weights_sha})
    write_json_atomic(run/'status.json', {'phase': 'encoded', 'queries': len(previous), 'training_started': False})


def load_records(run, protocol):
    features_report = read(run/'features_summary.json')
    if sha256_file(run/'feature_records.jsonl') != features_report['feature_records_sha256']:
        raise ValueError('Feature manifest changed')
    expected = {(split, e['qid']):e for split,e in usable_entries(run, protocol)}
    metadata = rows(run/'feature_records.jsonl')
    if len(metadata) != len(expected) or {(r['split'],r['entry']['qid']):r['entry'] for r in metadata} != expected:
        raise ValueError('Features differ from frozen query identities')
    records = []; visual = {}
    for item in metadata:
        path = item['images_path']; text_path = item['text_path']
        if path not in visual:
            if sha256_file(path) != item['images_sha256']:
                raise ValueError('Visual feature corruption')
            with np.load(path, allow_pickle=False) as data:
                visual[path] = (data['images'], data['times'])
        if sha256_file(text_path) != item['text_sha256']:
            raise ValueError('Query feature corruption')
        v, t = visual[path]
        records.append({**item, 'images': v, 'times': t, 'query_embedding': np.load(text_path, allow_pickle=False)})
    return records


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model(variant, embedding_dim, config):
    if variant.startswith('univtg'):
        return UniVTGHead(embedding_dim, config['univtg_hidden'])
    architecture = {'matched_mlp':'frame_mlp', 'matched_conv':'temporal_conv'}[variant]
    target = parameter_count(UniVTGHead(embedding_dim, config['univtg_hidden']))
    # Find capacity without drawing randomly initialized models for every width.
    def count(h):
        project = (embedding_dim*3+4)*h
        context = (2*h*h+2*h) if architecture == 'frame_mlp' else (6*h*h+2*h)
        return project+context+3*h+3
    hidden = min(range(8, 513), key=lambda h: abs(count(h)-target))
    return LocalizationHead(embedding_dim, hidden, architecture)


def variant_control(variant):
    return 'no_query' if variant == 'univtg_no_query' else 'position_only' if variant == 'univtg_position_only' else 'full'


def study_mock_examples(video, feature_config):
    records = mock_examples(video, feature_config)
    for record in records:
        record['entry']['vid'] = f'synthetic{record["entry"]["qid"]}_0.0_30.0'
    return records


def evaluate(model, records, control, cfg, full_gt=None):
    model.eval(); predictions = {}
    with torch.inference_mode():
        for record in records:
            logits, distances = model(study_input(record, control))
            windows = decode_spans(record['times'], record['duration_sec'], logits, distances,
                                    cfg['top_k'], cfg['nms_iou'], cfg['min_span_sec'])
            predictions[record['entry']['qid']] = [(w.start_sec,w.end_sec,w.score) for w in windows]
    gt = full_gt or {r['entry']['qid']:r['entry'] for r in records}
    for qid in gt: predictions.setdefault(qid,[])
    metrics = dict(eval_mr_brief(predictions, gt))
    from queryseek.tools.validate import iou
    from queryseek.schemas import TimeWindow
    oracle = 0
    for record in records:
        positives = [TimeWindow(start_sec=a,end_sec=b) for a,b in record['entry']['relevant_windows']]
        if any(iou(TimeWindow(start_sec=a,end_sec=b), positive)>=.5
               for a,b,_ in predictions[record['entry']['qid']] for positive in positives): oracle += 1
    return metrics, predictions, {'top5_oracle_hits': oracle, 'queries':len(gt),'available_queries':len(records)}


def fit_study(train, validation, variant, config, feature_cfg, out, validation_gt=None):
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config['head_seed'])
    model = make_model(variant, train[0]['images'].shape[1], config)
    control = variant_control(variant)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=config['learning_rate'], weight_decay=config['weight_decay'])
    prepared = [(study_input(r, control), *supervision(r['times'],r['duration_sec'],r['entry']['relevant_windows'])) for r in train]
    source_counts = Counter(source_video_id(r['entry']['vid']) for r in train)
    weights = [1/source_counts[source_video_id(r['entry']['vid'])] for r in train]
    rng = np.random.default_rng(config['head_seed']); history = []; best = -1; selected = None
    started = time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        model.train(); losses = []
        for idx in rng.permutation(len(train)):
            x,y,d = prepared[idx]; optimizer.zero_grad(set_to_none=True)
            loss,_ = localization_loss(*model(x),y,d,feature_cfg['loss_weights'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            (loss*weights[idx]).backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise ValueError('Missing/nonfinite trainable parameter gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip']); optimizer.step()
            losses.append(float(loss.detach()))
        row = {'epoch':epoch, 'train_query_mean_loss':float(np.mean(losses))}
        if epoch % config['validation_every'] == 0:
            metrics,_,_ = evaluate(model, validation, control, feature_cfg,validation_gt)
            row['validation_metrics'] = metrics
            score = metrics['MR-full-mAP']
            if score > best:
                best = score; selected = epoch
                torch.save({'state_dict':model.state_dict(), 'variant':variant, 'epoch':epoch,
                            'config':config, 'embedding_dim':train[0]['images'].shape[1]},out/'selected.pt')
            print('train',len(source_counts),variant,epoch,'loss',row['train_query_mean_loss'],'val_mAP',score,flush=True)
        history.append(row); write_json_atomic(out/'history.json', history)
    torch.save({'state_dict':model.state_dict(), 'epoch':config['epochs']},out/'last.pt')
    checkpoint = torch.load(out/'selected.pt', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    metrics,predictions,diagnostic = evaluate(model, validation, control, feature_cfg,validation_gt)
    train_metrics,_,train_diagnostic = evaluate(model, train, control, feature_cfg)
    restored = make_model(variant,train[0]['images'].shape[1],config)
    restored.load_state_dict(checkpoint['state_dict']); restored.eval(); model.eval()
    with torch.inference_mode():
        x = study_input(validation[0],control)
        exact = all(torch.equal(a,b) for a,b in zip(model(x),restored(x)))
    if not exact: raise ValueError('Checkpoint reload mismatch')
    write_json_atomic(out/'predictions.json', predictions)
    result = {'validation_metrics':metrics, 'train_metrics':train_metrics,
              'validation_diagnostic':diagnostic,'train_diagnostic':train_diagnostic,
              'selected_epoch':selected,'trained_epochs':config['epochs'], 'parameter_count':parameter_count(model),
              'checkpoint_reload_exact':exact, 'checkpoint_sha256':sha256_file(out/'selected.pt'),
              'training_and_evaluation_sec':time.perf_counter()-started,
              'control':control, 'checkpoint_selected_on_reported_validation':True}
    write_json_atomic(out/'metrics.json',result)
    return result


def train(run):
    protocol = verify(run); config = protocol['config']; cfg = protocol['feature_config']
    torch.set_num_threads(config['torch_threads']); torch.use_deterministic_algorithms(True)
    records = load_records(run,protocol)
    validation = [r for r in records if r['split']=='validation']
    gt = {e['qid']:e for e in rows(run/'validation.jsonl')}
    e4 = {r['entry']['qid']:[(w.start_sec,w.end_sec,w.score) for w in select_curve_windows(
         curve_spans(r['times'],r['images']@r['query_embedding'],r['duration_sec']),5,.9)] for r in validation}
    for qid in gt: e4.setdefault(qid,[])
    write_json_atomic(run/'e4_predictions.json',e4)
    report = {'mock':False,'independent':False,'vlm_calls':0,'encoder_training':False,
              'pretrained_localizer_weights':False,'validation_queries':len(gt),
              'validation_available_queries':len(validation),'validation_coverage':len(validation)/len(gt),
              'validation_sources':len({source_video_id(e['vid']) for e in gt.values()}),
              'e4_same_features':dict(eval_mr_brief(e4,gt)), 'scales':{}}
    write_json_atomic(run/'status.json',{'phase':'training','training_started':True})
    for n in config['train_scales']:
        ids = {e['qid'] for e in rows(run/f'train_{n}.jsonl')}
        rr = [r for r in records if r['split']=='train' and r['entry']['qid'] in ids]
        scale = {'train_queries':len(rr),'planned_train_sources':n,
                 'train_sources':len({source_video_id(r['entry']['vid']) for r in rr}),'variants':{}}
        for variant in config['variants']:
            out = run/'models'/str(n)/variant
            # Completed immutable fits are reused when explicitly resuming.
            scale['variants'][variant] = read(out/'metrics.json') if (out/'metrics.json').exists() else fit_study(rr,validation,variant,config,cfg,out,gt)
            report['scales'][str(n)] = scale
            write_json_atomic(run/'metrics.json',report)
    verify(run)
    write_json_atomic(run/'status.json',{'phase':'complete','training_started':True,'independent':False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['freeze','prepare','encode','train','mock'])
    parser.add_argument('--run',type=Path)
    parser.add_argument('--config',default='configs/localizer_study_v2.json')
    parser.add_argument('--video',default='assets/demo/sample.mp4')
    args = parser.parse_args()
    if args.phase == 'mock':
        config = read(ROOT/args.config)
        config = {**config,'epochs':2,'validation_every':1,'univtg_hidden':8,'torch_threads':1}
        cfg = read(ROOT/config['pilot_config'])
        torch.set_num_threads(1)
        rr = study_mock_examples(ROOT/args.video,cfg)
        run = checked_output(ROOT/'outputs/localizer_study_mock'/new_job_id()); run.mkdir(parents=True)
        report = {'mock':True,'scope':'synthetic mechanical test, not accuracy','variants':{}}
        for variant in config['variants']:
            report['variants'][variant] = fit_study(rr[:4],rr[4:],variant,config,cfg,run/variant)
        write_json_atomic(run/'metrics.json',report)
    elif args.phase == 'freeze':
        run = freeze(read(ROOT/args.config),ROOT/'outputs/localizer_study'/new_job_id())
    else:
        if args.run is None: parser.error('--run is required')
        run = checked_output(args.run)
        {'prepare':prepare,'encode':encode,'train':train}[args.phase](run)
    print('OUTPUT='+str(run),flush=True)


if __name__ == '__main__': main()
