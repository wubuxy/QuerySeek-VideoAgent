"""Matched image-budget caption binding diagnosis; never an online strategy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.event_composition_probe import phase_diagnosis
from scripts.mechanism_probe import midpoint_times, snapshot
from scripts.sampling_probe import checked_output, extra_features
from scripts.state_fact_bridge import PROMPT, locate_source_frames, opaque_ids, validate_facts
from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
from queryseek.env import ensure_dotenv
from queryseek.hints import hint_at
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.ingest.pipeline import VideoIndex
from queryseek.models.compatible_api import _image_data_url, _parse_json_strict
from queryseek.models.factory import build_vlm
from queryseek.retrieval.event_composition import ordered_event_margin
from queryseek.tools.export import new_job_id

ARMS = ('single', 'batch', 'interleaved')


def make_plan(manifest):
    if (manifest['arms'] != list(ARMS) or len(set(manifest['qids'])) != len(manifest['qids'])
            or not manifest['qids'] or manifest['frames_per_case'] != 16
            or manifest['max_retries'] != 0 or not 1 <= manifest['workers'] <= 2):
        raise ValueError('Unique cases and frozen three-arm/budget protocol required')
    plan = []
    for qid in manifest['qids']:
        for arm in ARMS:
            subsets = [[i] for i in range(16)] if arm == 'single' else [list(range(16))]
            for positions in subsets:
                plan.append({'qid': qid, 'arm': arm, 'positions': positions})
    if len(plan) > manifest['max_calls'] or manifest['max_calls'] > 36:
        raise ValueError('Hard 36-call limit exceeded')
    rng = np.random.default_rng(manifest['request_schedule_seed'])
    return [plan[i] for i in rng.permutation(len(plan))]


def prompt_parts(ids, paths, arm, image_encoder=_image_data_url):
    if len(ids) != len(paths) or not ids or len(set(ids)) != len(ids) or arm not in ARMS:
        raise ValueError('Aligned unique opaque identifiers required')
    if arm == 'single' and len(ids) != 1:
        raise ValueError('Single-image arm must contain exactly one image')
    if arm == 'interleaved':
        parts = [{'type': 'text', 'text': PROMPT}]
        for frame_id, path in zip(ids, paths):
            parts += [{'type': 'text', 'text': json.dumps({'frame_id': frame_id})},
                      {'type': 'image_url', 'image_url': {'url': image_encoder(path)}}]
        return parts
    return [{'type': 'text', 'text': PROMPT + json.dumps([{'frame_id': i} for i in ids])}] + [
        {'type': 'image_url', 'image_url': {'url': image_encoder(path)}} for path in paths]


def run_request(job, row, frames, cfg, manifest, model, out, deadline, mock=False, hints=None, factory=build_vlm):
    positions = job['positions']
    ids = [opaque_ids(row)[i] for i in positions]
    paths = [frames[i].image_path for i in positions]
    limit = manifest['single_output_tokens'] if job['arm'] == 'single' else manifest['multi_output_tokens']
    signature = {'version': manifest['version'], 'pipeline_version': PIPELINE_VERSION,
                 'video_sha256': row['video_sha256'], 'image_sha256': [sha256_file(p) for p in paths],
                 'actual_times_sec': [row['actual_times_sec'][i] for i in positions],
                 'frame_ids': ids, 'arm': job['arm'], 'prompt': PROMPT, 'layout_version': 'opaque-id-placement-v1',
                 'model': model, 'max_output_tokens': limit, 'temperature': manifest['temperature'],
                 'enable_thinking': manifest['enable_thinking'], 'mock': mock}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = checked_output(ROOT/'outputs/caption_binding_cache'/f'{key}.json')
    base = {**job, 'cache_key': key, 'actual_calls': 0, 'image_presentations': 0,
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
    started = time.perf_counter()
    client = None
    try:
        if cache.is_file():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved['signature'] != signature:
                raise ValueError('Caption cache signature mismatch')
            obj = saved['response']
            hit = True
        elif mock:
            obj = {'observations': [{'frame_id': frame_id,
                                    'fact': hint_at(hints or [], frames[i].timestamp_sec) or 'Unknown visible objects.'}
                                   for frame_id, i in zip(ids, positions)]}
            hit = False
        else:
            if time.perf_counter() >= deadline:
                raise TimeoutError('Request start budget exceeded')
            request_cfg = cfg.model_copy(deep=True)
            request_cfg.vlm.max_retries = 0
            request_cfg.vlm.max_output_tokens = limit
            request_cfg.vlm.enable_thinking = manifest['enable_thinking']
            request_cfg.vlm.temperature = manifest['temperature']
            request_cfg.vlm.read_timeout_sec = manifest['read_timeout_sec']
            request_cfg.vlm.req_log_dir = str(checked_output(out/'requests'/key))
            client = factory(request_cfg, use_mock=False)
            if not hasattr(client, '_chat') or client.model != model:
                raise RuntimeError('Real configured model required; no silent fallback')
            parts = prompt_parts(ids, paths, job['arm'])
            base['actual_calls'] = 1
            base['image_presentations'] = len(ids)
            try:
                obj = _parse_json_strict(client._chat(parts, max_retries=0))
            finally:
                base['usage'] = dict(client.last_usage)
            hit = False
        facts = validate_facts(obj, ids)
        if not hit:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({'signature': signature, 'response': obj}), encoding='utf-8')
        return {**base, 'completed': True, 'facts': facts, 'frame_ids': ids, 'cache_hit': hit,
                'elapsed_sec': time.perf_counter() - started}
    except Exception as exc:
        # Record only whitelisted internal validation categories, never an API
        # exception body which could echo a sensitive request/response payload.
        validation_kind = {'Facts must cover every supplied image': 'count_or_container',
                           'Invalid, duplicate, or unsupported fact entry': 'entry_identity_or_fields',
                           'Caption cache signature mismatch': 'cache_signature'}.get(str(exc))
        return {**base, 'completed': False, 'error_type': type(exc).__name__,
                'validation_kind': validation_kind,
                'elapsed_sec': time.perf_counter() - started}
    finally:
        if client is not None and hasattr(client, 'close'):
            client.close()


def summarize_requests(results, plan, mock):
    usage = {k: sum(r['usage'].get(k, 0) for r in results)
             for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    arms = {}
    for arm in ARMS:
        planned = [j for j in plan if j['arm'] == arm]
        done = [r for r in results if r['arm'] == arm]
        arms[arm] = {'planned_calls': len(planned), 'completed_calls': sum(r['completed'] for r in done),
                     'planned_frames': sum(len(j['positions']) for j in planned),
                     'completed_frames': sum(len(r['positions']) for r in done if r['completed']),
                     'actual_calls': sum(r['actual_calls'] for r in done),
                     'image_presentations': sum(r['image_presentations'] for r in done),
                     'usage': {k: sum(r['usage'].get(k, 0) for r in done) for k in usage}}
    return {'mock': mock, 'independent': False, 'oracle_only': True, 'planned_calls': len(plan),
            'completed_calls': sum(r['completed'] for r in results),
            'coverage': sum(r['completed'] for r in results)/len(plan),
            'actual_vlm_calls': sum(r['actual_calls'] for r in results),
            'actual_image_presentations': sum(r['image_presentations'] for r in results),
            'usage': usage, 'arms': arms, 'new_image_encodings': 0,
            'errors': [{k: r[k] for k in ('qid','arm','positions','error_type')} for r in results if not r['completed']],
            'quality_review': 'TODO / not yet reviewed', 'actual_bill': 'TODO / unverified'}


def collate(results, rows):
    assembled = {(r['qid'], arm): [None]*16 for r in rows for arm in ARMS}
    for result in results:
        if not result['completed']:
            continue
        target = assembled[result['qid'], result['arm']]
        for position, fact in zip(result['positions'], result['facts']):
            if target[position] is not None:
                raise ValueError('Duplicate frame facts in one arm')
            target[position] = fact
    return assembled


def write_review(out, assembled, rows, frame_map, truth):
    truth_by_q = {c['qid']: c['states_source_order'] for c in truth['cases']}
    items, binding = [], {}
    for row in rows:
        for arm in ARMS:
            for position, fact in enumerate(assembled[row['qid'], arm]):
                rid = hashlib.sha256(f'{out.name}:{row["qid"]}:{arm}:{position}'.encode()).hexdigest()[:16]
                items.append({'review_id': rid, 'qid': row['qid'], 'source_position': position,
                              'source_time_sec': row['actual_times_sec'][position],
                              'image_path': frame_map[row['qid']][position].image_path,
                              'expected_state': truth_by_q[row['qid']][position], 'fact': fact})
                binding[rid] = {'qid': row['qid'], 'arm': arm, 'source_position': position}
    order = np.random.default_rng(273123).permutation(len(items))
    (out/'review_items.jsonl').write_text(''.join(json.dumps(items[i])+'\n' for i in order), encoding='utf-8')
    (out/'review_binding.json').write_text(json.dumps(binding, indent=2), encoding='utf-8')


def apply_review(run, ratings):
    run = checked_output(run)
    items = {r['review_id']: r for r in (json.loads(line) for line in (run/'review_items.jsonl').read_text().splitlines())}
    binding = json.loads((run/'review_binding.json').read_text())
    if ratings['review_items_sha256'] != sha256_file(run/'review_items.jsonl'):
        raise ValueError('Review items changed')
    entries = ratings['ratings']
    if len(entries) != len(items) or {r['review_id'] for r in entries} != set(items):
        raise ValueError('Every planned frame must have one review entry')
    by_id = {r['review_id']: r for r in entries}
    summary = {}
    for arm in ARMS:
        ids = [rid for rid, b in binding.items() if b['arm'] == arm]
        states = [rid for rid in ids if items[rid]['expected_state'] not in ('uncertain','not_target')]
        counters = {label: 0 for label in ('supported','omitted','contradicted','uncertain','missing')}
        for rid in ids:
            entry = by_id[rid]
            if (entry['state_label'] not in counters or type(entry['wrong_image_claim']) is not bool
                    or not isinstance(entry['reason'], str) or not entry['reason'].strip()
                    or (entry['state_label'] == 'missing') != (items[rid]['fact'] is None)):
                raise ValueError('Invalid review category, identity judgement, or missing caption status')
            if rid in states:
                counters[entry['state_label']] += 1
        summary[arm] = {'planned_frames': len(ids), 'state_denominator': len(states), **counters,
                        'wrong_image_claims': sum(by_id[rid]['wrong_image_claim'] for rid in ids),
                        'state_recovery_rate': counters['supported']/len(states) if states else None}
    result = {'reviewer': ratings['reviewer'], 'independent_human_adjudication': False,
              'arms': summary, 'interpretation': 'selected cases; states scored by one Agent, frames are not independent source-video samples'}
    (run/'review_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (run/'quality_ratings.json').write_text(json.dumps(ratings, indent=2), encoding='utf-8')
    if (run/'metrics.json').is_file():
        request_metrics = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
        request_metrics['quality_review'] = 'review_metrics.json'
        (run/'metrics.json').write_text(json.dumps(request_metrics,indent=2),encoding='utf-8')
    return result


def run_probe(manifest, rows, cfg, out, truth, mock=False, frame_map=None, hints=None, factory=build_vlm, text_encoder=None):
    plan = make_plan(manifest)
    selected = [next(r for r in rows if r['qid'] == qid) for qid in manifest['qids']]
    if any(len(r['actual_times_sec']) != 16 for r in selected):
        raise ValueError('All arms must use the exact sixteen source observations')
    frame_map = frame_map or {r['qid']: locate_source_frames(r) for r in selected}
    truth_by_q = {c['qid']: c['states_source_order'] for c in truth['cases']}
    for row in selected:
        if (len(truth_by_q.get(row['qid'], [])) != 16
                or [f.timestamp_sec for f in frame_map[row['qid']]] != row['actual_times_sec']):
            raise ValueError('Frozen reference states and exact source PTS required')
    model = 'mock-hints' if mock else os.environ.get('VLM_MODEL', '')
    if not mock and (not os.environ.get('VLM_API_KEY') or not model):
        raise RuntimeError('Real API environment required; use --mock explicitly')
    out = checked_output(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'manifest': manifest, 'truth': truth, 'mock': mock, 'pipeline_version': PIPELINE_VERSION,
                'config': cfg.model_dump(mode='json'), 'model': model, 'request_plan': plan, 'prompt': PROMPT}
    if not mock:
        source = ROOT/manifest['source_run']
        protocol['source_hashes'] = {n: sha256_file(source/n) for n in ('protocol.json','records.jsonl','metrics.json')}
        protocol['archived_fact_hashes'] = {n: sha256_file(ROOT/manifest['archived_fact_run']/n)
                                          for n in ('protocol.json','records.jsonl','metrics.json')}
        protocol['fresh_selection_sha256'] = sha256_file(ROOT/'data/qvhighlights_fresh80/selection.json')
    protocol['source_sha256'] = snapshot(out, [ROOT/'scripts/caption_binding_probe.py',
        ROOT/'scripts/state_fact_bridge.py', ROOT/'configs/caption_binding_v1.json',
        ROOT/'configs/caption_state_truth_v1.json', ROOT/'configs/state_minimal_pairs_v1.json',
        ROOT/'queryseek/models/compatible_api.py', ROOT/'queryseek/embeddings/openclip_encoder.py',
        ROOT/'queryseek/retrieval/event_composition.py', ROOT/'scripts/event_composition_probe.py',
        ROOT/'scripts/sampling_probe.py', ROOT/'queryseek/config.py', ROOT/'pyproject.toml',
        ROOT/'docs/EVAL_CAPTION_BINDING.md'])
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    by_q = {r['qid']: r for r in selected}
    results = []
    deadline = time.perf_counter() + manifest['start_budget_sec']
    with ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
        futures = [pool.submit(run_request, job, by_q[job['qid']], frame_map[job['qid']], cfg,
                               manifest, model, out, deadline, mock, hints, factory) for job in plan]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with (out/'requests_records.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(result)+'\n')
            report = summarize_requests(results, plan, mock)
            (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f"{len(results)}/{len(plan)} qid={result['qid']} {result['arm']} complete={result['completed']}", flush=True)
    assembled = collate(results, selected)
    write_review(out, assembled, selected, frame_map, truth)
    if text_encoder is None:
        text_encoder = MockEmbedding() if mock else OpenCLIPEmbedding()
    pairs = {p['qid']: [p['a'], p['b']] for p in json.loads((ROOT/'configs/state_minimal_pairs_v1.json').read_text())['pairs']} if not mock else {}
    clips = []
    for row in selected:
        vectors = text_encoder.encode_texts(pairs.get(row['qid'], [row['chain']['event_a'], row['chain']['event_b']]))
        for arm in ARMS:
            facts = assembled[row['qid'], arm]
            complete = all(f is not None for f in facts)
            result = {'qid': row['qid'], 'arm': arm, 'complete': complete, 'facts_source_order': facts,
                      'actual_times_sec': row['actual_times_sec']}
            if complete:
                scores = text_encoder.encode_texts(facts).astype(np.float64) @ vectors.astype(np.float64).T
                result.update({'order': ordered_event_margin(scores),
                               'phases': phase_diagnosis(scores, row['actual_times_sec'], row['chain'])})
            clips.append(result)
    (out/'clips.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in clips), encoding='utf-8')
    if not mock and sha256_file(ROOT/'data/qvhighlights_fresh80/selection.json') != protocol['fresh_selection_sha256']:
        raise RuntimeError('Independent selection changed')
    print('OUTPUT='+str(out), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--analyze', type=Path)
    parser.add_argument('--ratings', type=Path)
    args = parser.parse_args()
    if args.analyze:
        if not args.ratings:
            parser.error('--ratings required for review analysis')
        print(json.dumps(apply_review(args.analyze, json.loads(args.ratings.read_text(encoding='utf-8'))), indent=2))
        return
    import torch
    torch.set_num_threads(4)
    if not args.mock:
        ensure_dotenv()
    cfg = Config.load(ROOT/'configs/qvhighlights_v2.yaml', apply_env=False).resolved()
    manifest = json.loads((ROOT/'configs/caption_binding_v1.json').read_text())
    truth = json.loads((ROOT/'configs/caption_state_truth_v1.json').read_text())
    out = ROOT/'outputs/caption_binding'/new_job_id()
    frame_map, hints = None, None
    if args.mock:
        asset = asset_from_probe('sample', ROOT/'assets/demo/sample.mp4')
        hints = json.loads((ROOT/'assets/demo/sample.labels.json').read_text(encoding='utf-8'))['segments']
        # Keep the same sixteen-observation budget as the real comparison.
        times = midpoint_times((0.,24.), asset.duration_sec, 16)
        frames, _, _, _, _ = extra_features(VideoIndex(asset, out, content_hints=hints), times, MockEmbedding(), cfg, True)
        rows = [{'qid': 0, 'video_sha256': asset.sha256, 'actual_times_sec': [f.timestamp_sec for f in frames],
                 'chain': {'event_a': 'intro welcome', 'event_b': 'architecture',
                           'event_a_window': [0.,12.], 'event_b_window': [12.,24.]}}]
        manifest['qids'] = [0]
        frame_map = {0: frames}
        truth = {'cases': [{'qid': 0, 'states_source_order': ['uncertain']*16}], 'reviewer': 'mock workflow only'}
    else:
        rows = [json.loads(line) for line in (ROOT/manifest['source_run']/'records.jsonl').read_text().splitlines()]
    result = run_probe(manifest, rows, cfg, out, truth, args.mock, frame_map, hints)
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
