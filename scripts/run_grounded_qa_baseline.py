"""One-call whole-video baseline, with source-PTS frames and explicit costs.

Consumes only requests.json; never loads evaluator labels. --mock runs the same
media path without model weights or API calls. Real inference reuses our tested
Qwen3-VL NF4 backend and requires a separately prepared local model snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from queryseek import PIPELINE_VERSION
from queryseek.eval.grounded_qa import checked_intervals, finite_number, uniform_times
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.ingest.frames import extract_scene_frames, load_frames_index, save_frames_index
from queryseek.schemas import Scene
from queryseek.tools.paths import resolved_path_is_within

PROMPT = ('Answer the multiple-choice question using the supplied video frames. '
          'Return one JSON object with answer (one choice letter) and intervals '
          '(one to five [start_sec, end_sec] evidence intervals). Times are source '
          'video seconds; require 0 <= start_sec < end_sec <= duration_sec. '
          'Identify evidence that supports your answer.\n')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def checked_output(path):
    path = Path(path).resolve()
    if not resolved_path_is_within(path, (ROOT / 'outputs').resolve()):
        raise ValueError('Outputs must remain under outputs/')
    return path


def check_requests(rows):
    allowed = {'qid', 'video_id', 'question', 'choices', 'duration_sec', 'video_path'}
    seen = set()
    if not isinstance(rows, list) or not rows:
        raise ValueError('Empty requests')
    for row in rows:
        if set(row) != allowed:
            raise ValueError('Inference requests must contain only whitelisted fields')
        if not isinstance(row['qid'], str) or row['qid'] in seen or not row['qid']:
            raise ValueError('Invalid/duplicate qid')
        if (not isinstance(row['question'], str) or not row['question'].strip()
                or not isinstance(row['choices'], list) or not 2 <= len(row['choices']) <= 26
                or any(not isinstance(c, str) or not c.strip() for c in row['choices'])
                or finite_number(row['duration_sec']) <= 0):
            raise ValueError('Invalid question, choices or duration')
        seen.add(row['qid'])
    return rows


def verify_model_snapshot(model_path, expected_revision):
    """Use a locally supplied full-file manifest, checked before allocating GPU."""
    path = Path(model_path)
    manifest = json.loads((path / 'QUERYSEEK_SNAPSHOT.json').read_text('utf-8'))
    if manifest.get('revision') != expected_revision or not manifest.get('files'):
        raise ValueError('Model revision or manifest missing')
    disk_files = {p.relative_to(path).as_posix() for p in path.rglob('*')
                  if p.is_file() and '.cache' not in p.relative_to(path).parts
                  and p.name != 'QUERYSEEK_SNAPSHOT.json'}
    if disk_files != set(manifest['files']):
        raise ValueError('Model manifest must cover all snapshot files')
    for name, expected in manifest['files'].items():
        target = path / name
        if not resolved_path_is_within(target.resolve(), path.resolve()) or sha256_file(target) != expected:
            raise ValueError('Model snapshot file hash mismatch')
    return sha256_file(path / 'QUERYSEEK_SNAPSHOT.json')


def sampled_frames(request, budget, max_side=512):
    started = time.perf_counter()
    asset = asset_from_probe(request['video_id'], request['video_path'])
    if abs(asset.duration_sec - request['duration_sec']) > 1.:
        raise ValueError('Media duration differs from frozen annotation by more than one second')
    times = uniform_times(asset.duration_sec, budget)
    signature = {'video_sha256': asset.sha256, 'pipeline_version': PIPELINE_VERSION,
                 'extractor': 'extract_scene_frames-source-pts-v3', 'budget': budget,
                 'max_side': max_side, 'requested_times_sec': times, 'jpeg_quality': 3}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = checked_output(ROOT / 'outputs/grounded_qa_frame_cache' / key)
    manifest_path = cache / 'manifest.json'
    hit = manifest_path.exists()
    if hit:
        manifest = json.loads(manifest_path.read_text('utf-8'))
        if manifest['signature'] != signature:
            raise ValueError('Frame cache signature mismatch')
        frames = load_frames_index(cache / 'frames.jsonl')
        if [sha256_file(f.image_path) for f in frames] != manifest['image_sha256']:
            raise ValueError('Frame cache content mismatch')
    else:
        scenes = [Scene(scene_id=f'qa_{i:04d}', start_sec=t-.0001, end_sec=t+.0001)
                  for i, t in enumerate(times)]
        frames = extract_scene_frames(asset.path, cache, scenes, max_side=max_side)
        for i, frame in enumerate(frames):
            frame.frame_id = f'qa_{i:04d}'
    if (len(frames) != budget or any(not 0 <= f.timestamp_sec < asset.duration_sec for f in frames)
            or len({f.timestamp_sec for f in frames}) != budget):
        raise ValueError('Incomplete or invalid source-PTS frame sample')
    if not hit:
        save_frames_index(frames, cache / 'frames.jsonl')
        write(manifest_path, {'signature': signature,
                             'image_sha256': [sha256_file(f.image_path) for f in frames]})
    return asset, frames, {'cache_hit': hit, 'video_sha256': asset.sha256,
                            'cache_key': key, 'elapsed_sec': time.perf_counter()-started,
                            'new_frames_extracted': 0 if hit else len(frames)}


def run(requests_path, output, config_path, *, budget=16, mock=False, model_path=None):
    requests = check_requests(json.loads(Path(requests_path).read_text('utf-8')))
    config = json.loads(Path(config_path).read_text('utf-8'))
    if budget not in config['frame_budgets']:
        raise ValueError('Budget not in frozen protocol')
    if not mock and (model_path is None or not Path(model_path).is_dir()):
        raise ValueError('Real inference requires a local fixed model snapshot')
    model_manifest_sha = verify_model_snapshot(model_path, config['model_revision']) if not mock else None
    output = checked_output(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'run_manifest.json', {
        'mock': mock, 'method': 'full_video_uniform', 'frame_budget': budget,
        'requests_sha256': sha256_file(requests_path), 'config_sha256': sha256_file(config_path),
        'runner_sha256': sha256_file(Path(__file__)), 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
        'model_path': str(model_path) if model_path else None,
        'model_manifest_sha256': model_manifest_sha,
        'model_revision_requested': config['model_revision'], 'accuracy_evaluated': False})
    backend = None
    load_start = time.perf_counter()
    if not mock:
        from scripts.train_agent_sft import QuantizedBackend
        cfg = {'seed': 12, 'model_path': str(Path(model_path).resolve()), 'min_pixels': 65536,
               'max_pixels': config['max_pixels_per_image'], 'attention': 'sdpa',
               'max_new_tokens': config['max_output_tokens'], 'max_seconds_per_generation': 120,
               'max_consecutive_errors': 3}
        backend = QuantizedBackend(cfg)
    model_load_sec = time.perf_counter()-load_start
    records = []
    for request in requests:
        started = time.perf_counter()
        record = {'qid': request['qid'], 'video_id': request['video_id'], 'mock': mock,
                  'status': 'error', 'frame_budget': budget, 'generation_attempted': False}
        previous_calls = len(backend.calls) if backend else 0
        try:
            asset, frames, extraction = sampled_frames(request, budget)
            record.update(preprocessing=extraction, source_times_sec=[f.timestamp_sec for f in frames])
            prompt_input = {k: request[k] for k in ['question', 'choices']}
            prompt_input['duration_sec'] = asset.duration_sec
            prompt_input['choice_letters'] = [chr(65+i) for i in range(len(request['choices']))]
            schema = {'type': 'object', 'additionalProperties': False,
                      'required': ['answer', 'intervals'], 'properties': {
                          'answer': {'type': 'string', 'enum': prompt_input['choice_letters']},
                          'intervals': {'type': 'array', 'minItems': 1, 'maxItems': 5,
                                        'items': {'type': 'array', 'minItems': 2, 'maxItems': 2,
                                                  'items': {'type': 'number'}}}}}
            if mock:
                prediction = {'answer': 'A', 'intervals': [[frames[0].timestamp_sec,
                                                           frames[-1].timestamp_sec]]}
            else:
                record['generation_attempted'] = True
                prediction = backend(PROMPT + json.dumps(prompt_input, ensure_ascii=False), frames, schema)
            if (prediction.get('answer') not in prompt_input['choice_letters']
                    or not isinstance(prediction.get('intervals'), list)
                    or not 1 <= len(prediction['intervals']) <= 5):
                raise ValueError('Invalid structured prediction')
            checked_intervals(prediction['intervals'], request['duration_sec'])
            record.update(answer=prediction['answer'], intervals=prediction['intervals'], status='ok')
        except Exception as exc:
            record['error_type'] = type(exc).__name__
        record['model_calls'] = backend.calls[previous_calls:] if backend else []
        record['elapsed_sec'] = time.perf_counter()-started
        records.append(record)
        with (output / 'predictions.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    calls = [c for row in records for c in row['model_calls']]
    summary = {'scope': 'Mock media-flow validation' if mock else 'Frozen development baseline',
               'scheduled': len(requests), 'completed': sum(r['status'] == 'ok' for r in records),
               'failed': sum(r['status'] != 'ok' for r in records), 'mock': mock,
               'model_load_sec': model_load_sec, 'vlm_calls': len(calls),
               'image_presentations': sum(c['images'] for c in calls),
               'input_tokens': sum(c['input_tokens'] for c in calls),
               'output_tokens': sum(c['output_tokens'] for c in calls), 'accuracy': None}
    write(output / 'run_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/grounded_qa_r1_v1.json')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.requests, args.output, args.config, budget=args.frames,
                         mock=args.mock, model_path=args.model_path), indent=2))


if __name__ == '__main__':
    main()
