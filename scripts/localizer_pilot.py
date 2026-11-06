"""Frozen official-train localization pilot; zero VLM and no fresh evaluation.

--prepare acquires/probes exactly the frozen pilot. --train requires a prepared
run, extracts cached 1fps features and trains both heads at fixed last epoch.
--mock uses only the standard generated synthetic fixture, never real accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from queryseek import PIPELINE_VERSION
from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import asset_from_probe, probe, sha256_file
from queryseek.ingest.frames import extract_uniform_frames, save_frames_index
from queryseek.retrieval.localizer import (
    LocalizationHead, condition_features, decode_spans, localization_loss, supervision,
)
from queryseek.retrieval.curve_spans import curve_spans, select_curve_windows
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.tools.export import new_job_id
from scripts.make_qvhighlights_subset import source_video_id
from scripts.prepare_localizer_split import valid_localization_rows
from scripts.sampling_probe import checked_output, optional_version


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def frozen_entries(config):
    root = checked_output(ROOT/config['data_run'])
    train, validation = rows(root/'pilot_train.jsonl'), rows(root/'pilot_validation.jsonl')
    good, bad = valid_localization_rows(train+validation)
    if bad or len(good) != 32 or len(train) != 24 or len(validation) != 8:
        raise ValueError('Frozen pilot must contain valid original 24+8 records')
    a, b = ({source_video_id(e['vid']) for e in subset} for subset in (train, validation))
    excluded = set(read(root/'excluded_sources.json'))
    if len(a) != 24 or len(b) != 8 or a&b or (a|b)&excluded:
        raise ValueError('Frozen pilot source isolation violation')
    return [(split, e) for split, rr in [('train', train), ('validation', validation)] for e in rr]


def check_video(entry, path, config):
    asset = asset_from_probe(entry['vid'], path)
    info = probe(path)
    stream = next(s for s in info['streams'] if s.get('codec_type') == 'video')
    # Do not silently use a longer audio stream as playable visual time.
    visual_duration = float(stream.get('duration') or asset.duration_sec)
    duration = min(visual_duration, asset.duration_sec)
    start = float(stream.get('start_time') or info.get('format', {}).get('start_time') or 0.)
    if abs(start) > .1:
        raise ValueError('Nonzero stream origin needs separate source-time audit')
    asset.duration_sec = duration
    difference = duration-float(entry['duration'])
    if abs(difference) > config['duration_reject_sec']:
        raise ValueError('Actual duration differs from annotation by over frozen tolerance')
    if any(not 0 <= float(a) < float(b) <= duration for a, b in entry['relevant_windows']):
        raise ValueError('Ground truth exceeds actual visual duration; no clipping allowed')
    if path.stat().st_size > config['max_video_bytes']:
        raise ValueError('Video exceeds fixed per-file budget')
    return {'video': asset.path, 'video_sha256': asset.sha256, 'duration_sec': duration,
            'annotation_duration_sec': entry['duration'], 'duration_difference_sec': difference,
            'duration_warning': abs(difference) > config['duration_warning_sec'],
            'stream_start_sec': start, 'width': asset.width, 'height': asset.height,
            'bytes': path.stat().st_size}


def acquire(entry, config):
    from huggingface_hub import hf_hub_download
    mirror = read(ROOT/config['mirror_index'])
    if mirror['repo_id'] != config['mirror_repo'] or mirror['revision'] != config['mirror_revision']:
        raise ValueError('Pinned mirror provenance mismatch')
    remote = mirror['selected'].get(entry['vid']+'.mp4')
    if (not remote or not remote.startswith('FlashVTGvis/qvhighlight/videos/')
            or '..' in Path(remote).parts or Path(remote).name != entry['vid']+'.mp4'):
        raise ValueError('Missing/unsafe selected mirror path')
    folder = checked_output(ROOT/'outputs/localizer_videos'/config['mirror_revision'])
    folder.mkdir(parents=True, exist_ok=True)
    target = checked_output(folder/(entry['vid']+'.mp4'))
    hit = target.is_file()
    if not hit:
        cached = hf_hub_download(repo_id=config['mirror_repo'], repo_type='dataset',
                                 revision=config['mirror_revision'], filename=remote,
                                 cache_dir=str(checked_output(ROOT/'outputs/localizer_download_cache')),
                                 token=False)
        if Path(cached).stat().st_size > config['max_video_bytes']:
            raise ValueError('Mirror video exceeds file budget')
        shutil.copyfile(cached, target)
    return target, {'repo': config['mirror_repo'], 'revision': config['mirror_revision'],
                    'remote': remote, 'cache_hit': hit,
                    'identity_validation': 'pinned mirror filename/metadata; not independently frame-certified'}


def snapshot(output, config, extra):
    files = ['configs/localizer_pilot_v1.json', 'scripts/localizer_pilot.py',
             'queryseek/retrieval/localizer.py', 'queryseek/ingest/frames.py',
             'queryseek/embeddings/openclip_encoder.py']
    copied = output/'source_snapshot'
    copied.mkdir()
    hashes = {}
    for name in files:
        source = ROOT/name
        target = copied/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = sha256_file(source)
    write_json_atomic(output/'protocol.json', {'config': config, 'pipeline_version': PIPELINE_VERSION,
                      'source_sha256': hashes, 'versions': {m: optional_version(m) for m in
                        ['torch', 'open-clip-torch', 'huggingface-hub', 'numpy']}, **extra})


def prepare(config, output, downloader=acquire):
    output = checked_output(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = frozen_entries(config)
    data_root = ROOT/config['data_run']
    snapshot(output, config, {'mock': False, 'purpose': 'availability and true-duration checks only',
             'data_sha256': {f: sha256_file(data_root/f) for f in
                 ['protocol.json', 'pilot_train.jsonl', 'pilot_validation.jsonl', 'excluded_sources.json']}})
    completed = []
    def one(split, entry):
        start = time.perf_counter()
        record = {'split': split, 'entry': entry, 'available': False}
        try:
            path, provenance = downloader(entry, config)
            record.update(check_video(entry, path, config))
            record.update(available=True, provenance=provenance)
        except Exception as exc:
            # Network errors can contain signed URLs; store type only.
            record['error_type'] = type(exc).__name__
            if isinstance(exc, ValueError):
                record['validation_reason'] = str(exc)
        record['elapsed_sec'] = time.perf_counter()-start
        return record
    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        futures = [pool.submit(one, split, entry) for split, entry in entries]
        for future in as_completed(futures):
            record = future.result()
            completed.append(record)
            with (output/'availability.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(record, ensure_ascii=False)+'\n')
            write_json_atomic(output/'metrics.json', {'planned': 32, 'checked': len(completed),
                 'available': sum(x['available'] for x in completed),
                 'failures': [{'vid': x['entry']['vid'], 'error_type': x['error_type']}
                              for x in completed if not x['available']],
                 'training_started': False, 'feature_encodings': 0, 'vlm_calls': 0})
            print('prepared', len(completed), '/32', record['entry']['vid'], record['available'], flush=True)
    return completed


def model_weights_sha(encoder):
    h = hashlib.sha256()
    for name, tensor in sorted(encoder._model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def features(record, config, encoder, weights_sha):
    start = time.perf_counter()
    signature = {'pipeline_version': PIPELINE_VERSION, 'video_sha256': record['video_sha256'],
                 'duration_sec': record['duration_sec'], 'fps': config['fps'], 'max_side': config['max_side'],
                 'model_id': encoder.model_id, 'weights_sha256': weights_sha,
                 'preprocess': repr(encoder._preprocess), 'extractor_sha256': sha256_file(ROOT/'queryseek/ingest/frames.py'),
                 'open_clip_version': optional_version('open-clip-torch'), 'jpeg_quality': 3}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = checked_output(ROOT/'outputs/localizer_feature_cache'/key)
    metadata = folder/'feature_meta.json'
    path = folder/'features.npz'
    hit = path.is_file() and metadata.is_file()
    if hit:
        meta = read(metadata)
        if meta['signature'] != signature or meta['feature_sha256'] != sha256_file(path):
            raise ValueError('Frozen feature cache mismatch')
        with np.load(path, allow_pickle=False) as data:
            images, times = data['images'], data['times']
    else:
        folder.mkdir(parents=True, exist_ok=True)
        ff = extract_uniform_frames(record['video'], folder/'frames', config['fps'],
                                    record['duration_sec'], config['max_side'])
        times = np.array([f.timestamp_sec for f in ff], np.float64)
        images = encoder.encode_images([f.image_path for f in ff])
        save_frames_index(ff, folder/'frames.jsonl')
        np.savez(path, images=images, times=times)
        write_json_atomic(metadata, {'signature': signature, 'feature_sha256': sha256_file(path)})
    query = record['entry']['query']
    text_signature = {'query': query, 'model_id': encoder.model_id, 'weights_sha256': weights_sha,
                      'open_clip_version': optional_version('open-clip-torch')}
    text_key = hashlib.sha256(json.dumps(text_signature, sort_keys=True).encode()).hexdigest()
    text_path = checked_output(ROOT/'outputs/localizer_text_cache'/(text_key+'.npy'))
    text_meta = text_path.with_suffix('.json')
    if text_path.is_file() and text_meta.is_file():
        tm = read(text_meta)
        if tm['signature'] != text_signature or tm['sha256'] != sha256_file(text_path):
            raise ValueError('Frozen text cache mismatch')
        text = np.load(text_path, allow_pickle=False)
    else:
        text = encoder.encode_texts([query])[0]
        text_path.parent.mkdir(parents=True, exist_ok=True); np.save(text_path, text)
        write_json_atomic(text_meta, {'signature': text_signature, 'sha256': sha256_file(text_path)})
    supervision(times, record['duration_sec'], record['entry']['relevant_windows'])
    return {**record, 'images': images, 'query_embedding': text, 'times': times,
            'cache_key': key, 'feature_cache_hit': hit, 'n_frames': len(times),
            'feature_elapsed_sec': time.perf_counter()-start}


def example_tensors(record, drop_query=False, reverse=False):
    images = record['images'][::-1].copy() if reverse else record['images']
    conditioned = condition_features(images, record['query_embedding'], record['times'],
                                      record['duration_sec'], drop_query)
    relevance, offsets = supervision(record['times'], record['duration_sec'],
                                     record['entry']['relevant_windows'])
    return conditioned, relevance, offsets


def prediction(model, record, config, drop_query=False, reverse=False):
    with torch.inference_mode():
        images = record['images'][::-1].copy() if reverse else record['images']
        conditioned = condition_features(images, record['query_embedding'], record['times'],
                                         record['duration_sec'], drop_query)
        logits, distances = model(conditioned)
    spans = decode_spans(record['times'], record['duration_sec'], logits, distances,
                         config['top_k'], config['nms_iou'], config['min_span_sec'])
    return [(w.start_sec, w.end_sec, w.score) for w in spans], logits


def fit(records, config, architecture, output):
    torch.manual_seed(config['seed'])
    model = LocalizationHead(records[0]['images'].shape[1], config['hidden'], architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    train = [r for r in records if r['split'] == 'train']
    validation = [r for r in records if r['split'] == 'validation']
    prepared = [example_tensors(r) for r in train]
    def train_loss():
        model.eval()
        with torch.inference_mode():
            return float(np.mean([float(localization_loss(*model(x), y, d, config['loss_weights'])[0])
                                  for x, y, d in prepared]))
    initial = train_loss(); history = []; rng = np.random.default_rng(config['seed'])
    start = time.perf_counter()
    all_grads_finite = True
    for epoch in range(config['epochs']):
        model.train(); epoch_losses = []
        for i in rng.permutation(len(prepared)):
            x, y, d = prepared[i]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = localization_loss(*model(x), y, d, config['loss_weights'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            good = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            all_grads_finite &= bool(good)
            if not good:
                raise RuntimeError('Missing/nonfinite model gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step(); epoch_losses.append(float(loss.detach()))
        history.append({'epoch': epoch+1, 'train_update_loss': float(np.mean(epoch_losses))})
        if (epoch+1) % 20 == 0:
            print(architecture, 'epoch', epoch+1, 'train loss', history[-1]['train_update_loss'], flush=True)
    elapsed = time.perf_counter()-start
    final = train_loss(); model.eval()
    checkpoint = output/(architecture+'.pt')
    torch.save({'state_dict': model.state_dict(), 'config': config, 'architecture': architecture,
                'embedding_dim': records[0]['images'].shape[1], 'epoch': config['epochs']}, checkpoint)
    reloaded = LocalizationHead(records[0]['images'].shape[1], config['hidden'], architecture)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict']); reloaded.eval()
    probe_x = prepared[0][0]
    with torch.inference_mode():
        equal = all(torch.equal(a, b) for a, b in zip(model(probe_x), reloaded(probe_x)))
    predictions = {}
    interventions = {}
    gt = {r['entry']['qid']: r['entry'] for r in validation}
    for arm in ['original', 'query_zeroed', 'frames_reversed']:
        pp = {}; changes = []
        for r in validation:
            spans, logits = prediction(model, r, config, arm == 'query_zeroed', arm == 'frames_reversed')
            qid = r['entry']['qid']; pp[qid] = spans
            if arm != 'original':
                _, original = prediction(model, r, config)
                changes.append(float((logits-original).abs().mean()))
        predictions[arm] = pp
        interventions[arm] = {'validation_metrics': dict(eval_mr_brief(pp, gt)),
                              'mean_absolute_logit_change': float(np.mean(changes)) if changes else 0.}
    result = {'architecture': architecture, 'parameter_count': sum(p.numel() for p in model.parameters()),
              'initial_train_loss': initial, 'final_train_loss': final, 'train_loss_decreased': final < initial,
              'all_gradients_finite': all_grads_finite, 'checkpoint_reload_exact': equal,
              'epochs': config['epochs'], 'training_sec': elapsed, 'interventions': interventions,
              'checkpoint_sha256': sha256_file(checkpoint), 'scope': 'mechanical pilot; not generalization evidence'}
    write_json_atomic(output/(architecture+'_history.json'), history)
    write_json_atomic(output/(architecture+'_predictions.json'), predictions)
    write_json_atomic(output/(architecture+'_metrics.json'), result)
    if not (all_grads_finite and equal and final < initial):
        raise RuntimeError('Pilot mechanical gate failed; metrics retained')
    return result


def train(config, prepared_run, output, mock=False, mock_records=None):
    output = checked_output(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config['torch_threads']); torch.use_deterministic_algorithms(True)
    snapshot(output, config, {'mock': mock, 'purpose': 'fixed last-epoch mechanical head comparison',
             'prepared_run': str(prepared_run), 'prepared_sha256': None if mock else
                 sha256_file(Path(prepared_run)/'availability.jsonl')})
    if mock:
        records = mock_records
    else:
        available = rows(Path(prepared_run)/'availability.jsonl')
        if read(Path(prepared_run)/'protocol.json')['config'] != config:
            raise ValueError('Preparation protocol differs from training configuration')
        expected = frozen_entries(config)
        if len(available) != 32 or any(not r['available'] for r in available):
            raise ValueError('All frozen 32 sources must pass availability; no silent subset/replacement')
        if sorted(expected, key=lambda p: p[1]['qid']) != sorted(
                ((r['split'], r['entry']) for r in available), key=lambda p: p[1]['qid']):
            raise ValueError('Prepared entries differ from frozen source pilot')
        encoder = OpenCLIPEmbedding(config['model_name'], config['pretrained'], config['device'],
                                    config['max_side'], config['force_quick_gelu'])
        weights = model_weights_sha(encoder)
        records = []
        for record in sorted(available, key=lambda r: r['entry']['qid']):
            if sha256_file(record['video']) != record['video_sha256']:
                raise ValueError('Prepared video changed')
            item = features(record, config, encoder, weights); records.append(item)
            print('features', len(records), '/32', item['n_frames'], 'hit', item['feature_cache_hit'], flush=True)
            with (output/'feature_records.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps({k: v for k, v in item.items() if k not in
                           ['images', 'query_embedding', 'times']}, ensure_ascii=False)+'\n')
        del encoder
    if not records or not {'train', 'validation'} <= {r['split'] for r in records}:
        raise ValueError('Both train and validation required')
    gt = {r['entry']['qid']: r['entry'] for r in records if r['split'] == 'validation'}
    e4 = {r['entry']['qid']: [(w.start_sec, w.end_sec, w.score) for w in select_curve_windows(
                curve_spans(r['times'], r['images']@r['query_embedding'], r['duration_sec']), 5, .9)]
          for r in records if r['split'] == 'validation'}
    metrics = {'mock': mock, 'independent': False, 'vlm_calls': 0,
               'scope': 'mechanical pilot only; internal 8 sources not independent evaluation',
               'n_train': sum(r['split'] == 'train' for r in records),
               'n_validation': sum(r['split'] == 'validation' for r in records),
               'frames': sum(len(r['times']) for r in records), 'encoder_training': False,
               'e4_same_features': dict(eval_mr_brief(e4, gt)), 'heads': {}}
    write_json_atomic(output/'e4_predictions.json', e4)
    for architecture in config['architectures']:
        metrics['heads'][architecture] = fit(records, config, architecture, output)
        write_json_atomic(output/'metrics.json', metrics)
    return metrics


def mock_examples(video, config):
    """Synthetic features/labels exercise learning mechanics, never real accuracy."""
    asset = asset_from_probe('synthetic_pilot', video)
    folder = checked_output(ROOT/'outputs/localizer_mock_frames'/asset.sha256)
    frames = extract_uniform_frames(video, folder, 1., asset.duration_sec, 128)
    times = np.array([f.timestamp_sec for f in frames], np.float64)
    rng = np.random.default_rng(123); records = []
    for i in range(6):
        q = np.zeros(8, np.float32); q[i % 2] = 1
        a, b = 2.+i, 6.+i
        images = rng.normal(0, .02, (len(times), 8)).astype(np.float32)
        images[:, i % 2] += ((times >= a)&(times < b)).astype(np.float32)
        records.append({'split': 'train' if i < 4 else 'validation', 'images': images,
                        'query_embedding': q, 'times': times, 'duration_sec': asset.duration_sec,
                        'entry': {'qid': i, 'query': 'synthetic mechanics', 'relevant_windows': [[a, b]]}})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/localizer_pilot_v1.json')
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--train', metavar='PREPARED_RUN')
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--video', default='assets/demo/sample.mp4')
    args = ap.parse_args(); config = read(args.config)
    out = checked_output(ROOT/'outputs/localizer_pilot'/new_job_id())
    print('output', out, flush=True)
    if args.mock:
        train(config, None, out, True, mock_examples(args.video, config))
    elif args.prepare:
        prepare(config, out)
    elif args.train:
        train(config, checked_output(Path(args.train)), out)
    else:
        ap.error('Choose --prepare, --train PREPARED_RUN or --mock')
    print('complete', out, flush=True)


if __name__ == '__main__':
    main()
