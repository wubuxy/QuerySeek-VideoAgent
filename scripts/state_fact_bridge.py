"""Query-free visual facts bridge on two frozen same-object state cases.

Reuses the exact source images. At most eight real requests, no retries. Raw
images and credentials are never written to request logs or experiment records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.event_composition_probe import phase_diagnosis
from scripts.mechanism_probe import midpoint_times, snapshot
from scripts.sampling_probe import checked_output, extra_features
from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
from queryseek.env import ensure_dotenv
from queryseek.hints import hint_at
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.ingest.frames import load_frames_index
from queryseek.ingest.pipeline import VideoIndex
from queryseek.models.compatible_api import _image_data_url, _parse_json_strict
from queryseek.models.factory import build_vlm
from queryseek.retrieval.event_composition import ordered_event_margin
from queryseek.tools.export import new_job_id

PROMPT = (
    'Describe visible facts in each supplied image independently, in English. '
    'Mention the actor, objects, visible condition or shape, and visible manipulation when supported. '
    'Do not infer unseen motion, causal relationships, or facts from adjacent images. '
    'Presentation order may be shuffled. A timestamp, when supplied, is only source metadata. '
    'Use one short sentence per image, at most 20 words; say uncertain when necessary. '
    'Return JSON {"observations": [{"frame_id": "provided ID", "fact": "visible facts"}]} '
    'with exactly one observation for every provided frame ID, no additional keys.\n'
)


def opaque_ids(row):
    return [hashlib.sha256(f"{row['video_sha256']}:{t:.9f}".encode()).hexdigest()[:12]
            for t in row['actual_times_sec']]


def caption_layout(row, group, seed):
    times = row['actual_times_sec']
    ids = opaque_ids(row)
    order = np.random.default_rng(seed).permutation(len(times)).tolist() if group.startswith('shuffle') else list(range(len(times)))
    lines = [{'frame_id': ids[i], **({'source_time_sec': float(times[i])} if group.endswith('source_time') else {})}
             for i in order]
    return ids, order, PROMPT + json.dumps(lines)


def validate_facts(obj, ids):
    facts = obj.get('observations') if isinstance(obj, dict) else None
    if not isinstance(facts, list) or len(facts) != len(ids):
        raise ValueError('Facts must cover every supplied image')
    by_id = {}
    for item in facts:
        if (not isinstance(item, dict) or set(item) != {'frame_id', 'fact'}
                or item['frame_id'] not in ids or item['frame_id'] in by_id
                or not isinstance(item['fact'], str) or not 0 < len(item['fact'].strip()) <= 600):
            raise ValueError('Invalid, duplicate, or unsupported fact entry')
        by_id[item['frame_id']] = item['fact'].strip()
    return [by_id[i] for i in ids]


def locate_source_frames(row):
    # Read metadata instead of reconstructing a mock-hints-dependent cache key.
    matches = []
    for meta_path in (ROOT/'outputs/sampling_feature_cache').glob('*/sampling_meta.json'):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta['video_sha256'] == row['video_sha256'] and meta['model_id'] == row['model_id']
                and meta['times_sec'] == row['requested_times_sec']):
            frames = load_frames_index(meta_path.parent/'frames.jsonl')
            if [f.timestamp_sec for f in frames] == row['actual_times_sec']:
                matches.append(frames)
    if len(matches) != 1:
        raise ValueError('Source frame identity must match exactly one feature cache')
    return matches[0]


def bridge_summary(records, intended, errors, attempts, presentations, usage, mock):
    grouped = {}
    for group in sorted({g for _, g in intended}):
        rows = [r for r in records if r['group'] == group]
        grouped[group] = {
            'planned': sum(g == group for _, g in intended), 'completed': len(rows),
            'source_order_positive': sum(r['source_order']['margin'] > 1e-12 for r in rows),
            'both_phase_positive': sum(r['phase_diagnosis']['a_vs_b_in_a_phase'] > 1e-12
                                       and r['phase_diagnosis']['b_vs_a_in_b_phase'] > 1e-12 for r in rows)}
    return {'mock': mock, 'independent': False, 'oracle_only': True, 'planned_groups': len(intended),
            'completed': len(records), 'coverage': len(records)/len(intended), 'errors': errors,
            'actual_vlm_calls': attempts, 'actual_image_presentations': presentations,
            'usage': usage, 'groups': grouped, 'new_image_encodings': 0,
            'actual_bill': 'TODO / unverified',
            'interpretation': 'two selected state cases; text-text cosine is not calibrated against image-text cosine'}


def run_bridge(manifest, rows, cfg, out, mock=False, frame_map=None, hints=None, embedder=None, vlm=None):
    groups = manifest['groups']
    if (len(set(manifest['qids'])) != len(manifest['qids']) or len(set(groups)) != len(groups)
            or set(groups) != {'original_no_time', 'original_source_time', 'shuffle_no_time', 'shuffle_source_time'}
            or len(manifest['qids'])*len(groups) > min(8, manifest['max_calls']) or manifest['max_retries'] != 0):
        raise ValueError('Frozen eight-call bound and four matched groups required')
    chosen = [next(r for r in rows if r['qid'] == qid) for qid in manifest['qids']]
    if any(len(r['actual_times_sec']) != manifest['images_per_call'] for r in chosen):
        raise ValueError('Equal frozen image budgets required')
    out = checked_output(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'manifest': manifest, 'mock': mock, 'pipeline_version': PIPELINE_VERSION,
                'config': cfg.model_dump(mode='json'), 'caption_prompt': PROMPT,
                'oracle_only': True, 'no_query_in_caption_request': True}
    if not mock:
        source = ROOT/manifest['source_run']
        protocol['source_hashes'] = {name: sha256_file(source/name) for name in ('protocol.json', 'records.jsonl', 'metrics.json')}
        protocol['fresh_selection_sha256'] = sha256_file(ROOT/'data/qvhighlights_fresh80/selection.json')
    protocol['source_sha256'] = snapshot(out, [ROOT/'scripts/state_fact_bridge.py',
        ROOT/'configs/state_fact_bridge_v1.json', ROOT/'configs/event_chains_v1.json',
        ROOT/'queryseek/retrieval/event_composition.py', ROOT/'queryseek/models/compatible_api.py',
        ROOT/'queryseek/embeddings/openclip_encoder.py', ROOT/'docs/EVAL_EVENT_COMPOSITION.md'])
    if embedder is None:
        embedder = MockEmbedding() if mock else OpenCLIPEmbedding(max_side=cfg.video.max_frame_side_px)
    owned_vlm = vlm is None and not mock
    if owned_vlm:
        vlm = build_vlm(cfg, use_mock=False)
    if not mock and not hasattr(vlm, '_chat'):
        raise RuntimeError('Real API key required; no silent mock fallback')
    protocol['caption_model'] = 'mock-hints' if mock else vlm.model
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    intended = [(r['qid'], g) for r in chosen for g in groups]
    records, errors = [], []
    attempts = presentations = 0
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    try:
        for row in chosen:
            frames = frame_map[row['qid']] if frame_map is not None else locate_source_frames(row)
            if ([f.timestamp_sec for f in frames] != row['actual_times_sec']
                    or any(not Path(f.image_path).is_file() for f in frames)):
                raise ValueError('Source images/time mismatch')
            events = embedder.encode_texts([row['chain']['event_a'], row['chain']['event_b']])
            for group in groups:
                ids, order, prompt = caption_layout(row, group, manifest['shuffle_seed'])
                signature = {'video_sha256': row['video_sha256'], 'pipeline_version': PIPELINE_VERSION,
                             'image_sha256': [sha256_file(f.image_path) for f in frames],
                             'prompt': prompt, 'image_order': order, 'model': protocol['caption_model'],
                             'temperature': cfg.vlm.temperature, 'max_output_tokens': manifest['max_output_tokens'],
                             'enable_thinking': manifest['enable_thinking'], 'mock': mock}
                key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
                cached = checked_output(ROOT/'outputs/state_fact_cache'/f'{key}.json')
                cache_hit = cached.is_file()
                try:
                    if cache_hit:
                        saved = json.loads(cached.read_text(encoding='utf-8'))
                        if saved['signature'] != signature:
                            raise ValueError('Caption cache mismatch')
                        obj = saved['response']
                    elif mock:
                        obj = {'observations': [{'frame_id': ids[i], 'fact': hint_at(hints or [], frames[i].timestamp_sec) or 'unknown visible objects'}
                                                for i in order]}
                    else:
                        if attempts >= manifest['max_calls']:
                            raise RuntimeError('Hard request budget exhausted')
                        attempts += 1
                        presentations += len(frames)
                        vlm.last_usage = {k: 0 for k in usage}
                        parts = [{'type': 'text', 'text': prompt}] + [
                            {'type': 'image_url', 'image_url': {'url': _image_data_url(frames[i].image_path)}} for i in order]
                        try:
                            obj = _parse_json_strict(vlm._chat(parts, max_retries=0))
                        finally:
                            for k in usage:
                                usage[k] += int(vlm.last_usage.get(k, 0))
                    facts = validate_facts(obj, ids)
                    if not cache_hit:
                        cached.parent.mkdir(parents=True, exist_ok=True)
                        cached.write_text(json.dumps({'signature': signature, 'response': obj}), encoding='utf-8')
                    fact_vectors = embedder.encode_texts(facts)
                    scores = fact_vectors.astype(np.float64) @ events.astype(np.float64).T
                    result = {'qid': row['qid'], 'group': group, 'caption_model': protocol['caption_model'],
                              'frame_ids_source_order': ids, 'displayed_frame_positions': order,
                              'facts_source_order': facts, 'actual_times_sec': row['actual_times_sec'],
                              'source_order': ordered_event_margin(scores),
                              'display_order': ordered_event_margin(scores[order]),
                              'phase_diagnosis': phase_diagnosis(scores, row['actual_times_sec'], row['chain']),
                              'per_frame_event_scores': scores.tolist(), 'cache_hit': cache_hit}
                    serialized = json.dumps(result)
                    with (out/'records.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(serialized+'\n')
                    records.append(result)
                    print(f"qid={row['qid']} {group} margin={result['source_order']['margin']:+.5f} cache={cache_hit}", flush=True)
                except Exception as exc:
                    errors.append({'qid': row['qid'], 'group': group, 'error_type': type(exc).__name__})
                    print(f"FAILED qid={row['qid']} {group}: {type(exc).__name__}", flush=True)
                report = bridge_summary(records, intended, errors, attempts, presentations, usage, mock)
                (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        if owned_vlm:
            vlm.close()
    if not mock and sha256_file(ROOT/'data/qvhighlights_fresh80/selection.json') != protocol['fresh_selection_sha256']:
        raise RuntimeError('Independent selection changed')
    print('OUTPUT='+str(out), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    if not args.mock:
        ensure_dotenv()
    cfg = Config.load(ROOT/'configs/qvhighlights_v2.yaml', apply_env=False).resolved()
    manifest = json.loads((ROOT/'configs/state_fact_bridge_v1.json').read_text(encoding='utf-8'))
    cfg.vlm.max_retries = 0
    cfg.vlm.max_output_tokens = manifest['max_output_tokens']
    cfg.vlm.read_timeout_sec = manifest['read_timeout_sec']
    cfg.vlm.enable_thinking = manifest['enable_thinking']
    cfg.vlm.max_frames_per_call = manifest['images_per_call']
    out = ROOT/'outputs/state_fact_bridge'/new_job_id()
    cfg.vlm.req_log_dir = str(checked_output(out/'requests'))
    frame_map, hints = None, None
    if args.mock:
        asset = asset_from_probe('sample', ROOT/'assets/demo/sample.mp4')
        hints = json.loads((ROOT/'assets/demo/sample.labels.json').read_text(encoding='utf-8'))['segments']
        chain = {'event_a': 'intro welcome', 'event_b': 'architecture',
                 'event_a_window': [0., 12.], 'event_b_window': [12., 24.]}
        requested = midpoint_times((0., 24.), asset.duration_sec, 16)
        frames, _, _, _, _ = extra_features(VideoIndex(asset, out, content_hints=hints), requested, MockEmbedding(), cfg, True)
        frame_map = {0: frames}
        rows = [{'qid': 0, 'video_sha256': asset.sha256, 'chain': chain,
                 'requested_times_sec': requested, 'actual_times_sec': [f.timestamp_sec for f in frames]}]
        manifest['qids'] = [0]
    else:
        rows = [json.loads(line) for line in (ROOT/manifest['source_run']/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    report = run_bridge(manifest, rows, cfg, out, args.mock, frame_map, hints)
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
