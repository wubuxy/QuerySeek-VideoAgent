"""Frozen input diagnostics; prepare privileged crops separately from inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from queryseek import PIPELINE_VERSION
from queryseek.eval.grounded_qa import checked_intervals, merged_intervals
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.ingest.frames import extract_scene_frames, load_frames_index, save_frames_index
from queryseek.schemas import Scene, VideoFrame
from scripts import run_grounded_qa_baseline as baseline

QA_PROMPT = 'Answer the multiple-choice question. Return one JSON object with answer (one choice letter).\n'


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def clue_times(intervals, duration, count):
    intervals = merged_intervals(checked_intervals(intervals, duration))
    if not isinstance(count, int) or isinstance(count, bool) or count < len(intervals):
        raise ValueError('Insufficient clue sampling budget')
    total = sum(b-a for a, b in intervals)
    result = []
    for i in range(count):
        offset = (i+.5)*total/count
        for a, b in intervals:
            if offset < b-a:
                result.append(a+offset)
                break
            offset -= b-a
    return result


def source_pts_metadata(times, duration):
    """Internal integer timebase transport, explicitly not a physical FPS claim."""
    if len(times) < 4 or len(times) % 2 or any(not 0 <= t < duration for t in times):
        raise ValueError('Need an even, nonempty source-time sequence')
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Source PTS must increase strictly')
    ticks = [round(t*1_000_000) for t in times]
    if any(abs(i/1_000_000-t) > .00000051 for i, t in zip(ticks, times)):
        raise ValueError('Timestamp transport mismatch')
    return {'fps': 1_000_000., 'frames_indices': ticks,
            'total_num_frames': round(duration*1_000_000), 'duration': duration}


def expected_video_timestamps(times):
    return [(a+b)/2 for a, b in zip(times[::2], times[1::2])]


def oracle_frames(request, intervals, count):
    started = time.perf_counter()
    asset = asset_from_probe(request['video_id'], request['video_path'])
    if abs(asset.duration_sec-request['duration_sec']) > .001:
        raise ValueError('Use the frozen probed duration')
    times = clue_times(intervals, asset.duration_sec, count)
    signature = {'video_sha256': asset.sha256, 'pipeline_version': PIPELINE_VERSION,
                 'extractor': 'extract_scene_frames-source-pts-v3', 'max_side': 1024,
                 'requested_times_sec': times, 'jpeg_quality': 3, 'scope': 'oracle-only'}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = baseline.checked_output(ROOT/'outputs/grounded_qa_oracle_cache'/key)
    manifest_path = cache/'manifest.json'
    hit = manifest_path.exists()
    if hit:
        manifest = json.loads(manifest_path.read_text('utf-8'))
        if manifest['signature'] != signature:
            raise ValueError('Oracle cache signature mismatch')
        frames = load_frames_index(cache/'frames.jsonl')
        if [sha256_file(f.image_path) for f in frames] != manifest['image_sha256']:
            raise ValueError('Oracle cache image mismatch')
    else:
        scenes = [Scene(scene_id=f'oracle_{i:04d}', start_sec=t-.0001, end_sec=t+.0001)
                  for i, t in enumerate(times)]
        frames = extract_scene_frames(asset.path, cache, scenes, max_side=1024)
        for i, frame in enumerate(frames):
            frame.frame_id = f'oracle_{i:04d}'
    if (len(frames) != count or len({f.timestamp_sec for f in frames}) != count
            or any(not any(a <= f.timestamp_sec < b for a, b in intervals) for f in frames)):
        raise ValueError('Incomplete/out-of-clue oracle sample')
    if not hit:
        save_frames_index(frames, cache/'frames.jsonl')
        write(manifest_path, {'signature': signature, 'image_sha256': [sha256_file(f.image_path) for f in frames]})
    return frames, {'elapsed_sec': time.perf_counter()-started, 'cache_hit': hit,
                    'new_frames_extracted': 0 if hit else len(frames), 'video_sha256': asset.sha256}


def prepare(requests_path, labels_path, config_path, output):
    config = json.loads(Path(config_path).read_text('utf-8'))
    if (sha256_file(requests_path) != config['requests_sha256']
            or sha256_file(labels_path) != config['labels_sha256']):
        raise ValueError('Frozen input hash mismatch')
    requests = baseline.check_requests(json.loads(Path(requests_path).read_text('utf-8')))
    labels = {r['qid']: r for r in json.loads(Path(labels_path).read_text('utf-8'))}
    if set(labels) != {r['qid'] for r in requests}:
        raise ValueError('Labels must match frozen requests')
    if len(requests)*len(config['conditions']) != config['max_generations']:
        raise ValueError('Frozen call budget mismatch')
    output = baseline.checked_output(output)
    output.mkdir(parents=True, exist_ok=False)
    records, cache = [], {}
    for condition in config['conditions']:
        for request in requests:
            key = (request['qid'], condition['scope'] == 'oracle_diagnostic', condition['frames'])
            started = time.perf_counter()
            reused = key in cache
            if condition['input'] == 'text':
                frames, cost = [], {'cache_hit': False, 'new_frames_extracted': 0, 'elapsed_sec': 0.}
            elif reused:
                frames, _ = cache[key]
                cost = {'cache_hit': True, 'new_frames_extracted': 0, 'elapsed_sec': time.perf_counter()-started}
            elif condition['scope'] == 'oracle_diagnostic':
                frames, cost = oracle_frames(request, labels[request['qid']]['gt_intervals_sec'], condition['frames'])
                cache[key] = (frames, cost)
            else:
                _, frames, cost = baseline.sampled_frames(request, condition['frames'])
                cache[key] = (frames, cost)
            records.append({'request': request, 'condition': condition['name'],
                            'frames': [f.model_dump() for f in frames],
                            'frame_sha256': [sha256_file(f.image_path) for f in frames],
                            'preprocessing': cost})
            print(json.dumps({'prepared': condition['name'], 'qid': request['qid'], 'frames': len(frames)}), flush=True)
    write(output/'prepared.json', records)
    write(output/'manifest.json', {'config_sha256': sha256_file(config_path),
          'requests_sha256': sha256_file(requests_path), 'labels_sha256': sha256_file(labels_path),
          'prepared_sha256': sha256_file(output/'prepared.json'), 'rows': len(records), 'vlm_calls': 0,
          'new_frames_extracted': sum(r['preprocessing']['new_frames_extracted'] for r in records),
          'preprocessing_sec': sum(r['preprocessing']['elapsed_sec'] for r in records)})


def validate_prepared(rows, config):
    conditions = {c['name']: c for c in config['conditions']}
    seen = set()
    for row in rows:
        if set(row) != {'request', 'condition', 'frames', 'frame_sha256', 'preprocessing'}:
            raise ValueError('Unexpected prepared fields')
        baseline.check_requests([row['request']])
        name = row['condition']
        condition = conditions[name]
        key = (name, row['request']['qid'])
        if key in seen:
            raise ValueError('Duplicate condition/query')
        seen.add(key)
        if len(row['frames']) != condition['frames'] or len(row['frame_sha256']) != condition['frames']:
            raise ValueError('Incorrect prepared frame count')
        for frame, digest in zip(row['frames'], row['frame_sha256']):
            baseline.checked_output(frame['image_path'])
            if sha256_file(frame['image_path']) != digest:
                raise ValueError('Prepared image hash mismatch')
        if condition['input'] != 'text':
            source_pts_metadata([f['timestamp_sec'] for f in row['frames']], row['request']['duration_sec'])
    ids = {r['request']['qid'] for r in rows}
    if seen != {(name, qid) for name in conditions for qid in ids} or len(rows) != config['max_generations']:
        raise ValueError('Incomplete frozen experiment matrix')


def prompt_and_schema(request, condition):
    letters = [chr(65+i) for i in range(len(request['choices']))]
    payload = {key: request[key] for key in ['question', 'choices']}
    payload['choice_letters'] = letters
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['answer'],
              'properties': {'answer': {'type': 'string', 'enum': letters}}}
    prompt = QA_PROMPT
    if condition['output'] == 'grounded':
        prompt = baseline.PROMPT
        payload['duration_sec'] = request['duration_sec']
        schema['required'].append('intervals')
        schema['properties']['intervals'] = {'type': 'array', 'minItems': 1, 'maxItems': 5,
            'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}}
    return prompt+json.dumps(payload, ensure_ascii=False), schema


def native_inputs(processor, prompt, frames, pixels, duration):
    """Use explicit PTS metadata and a per-frame pixel budget with HF's video API."""
    from PIL import Image
    import numpy as np
    from transformers.video_utils import VideoMetadata
    arrays = []
    for frame in frames:
        with Image.open(frame.image_path) as image:
            arrays.append(np.array(image.convert('RGB')))
    times = [f.timestamp_sec for f in frames]
    meta = VideoMetadata(**source_pts_metadata(times, duration))
    content = [{'type': 'text', 'text': prompt}, {'type': 'video'}]
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                tokenize=False, add_generation_prompt=True)
    # Qwen3 video smart_resize constrains T*H*W, whereas its image API constrains H*W.
    inputs = processor(text=[text], videos=[np.stack(arrays)], video_metadata=[meta],
        do_sample_frames=False, fps=None, return_tensors='pt',
        size={'shortest_edge': pixels*len(frames), 'longest_edge': pixels*len(frames)})
    decoded = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
    rendered = re.findall(r'<([0-9]+\.[0-9]) seconds>', decoded)
    expected = expected_video_timestamps([i/1_000_000 for i in meta.frames_indices])
    if rendered != [f'{t:.1f}' for t in expected]:
        raise ValueError('Native video timestamp mapping differs from source PTS')
    grids = inputs.video_grid_thw.tolist()
    if sum(t for t, h, w in grids)*2 != len(frames):
        raise ValueError('Unexpected temporal resampling')
    # Integer spatial patch rounding is permitted, but must not shrink with frame count.
    patch_size = processor.video_processor.patch_size
    areas = [h*w*patch_size**2 for t, h, w in grids]
    if any(not .5*pixels <= area <= 1.5*pixels for area in areas):
        raise ValueError('Unexpected per-frame pixel budget')
    return inputs, {'video_grid_thw': grids, 'grouped_source_times_sec': expected,
                    'rendered_source_times_sec': [float(t) for t in rendered],
                    'actual_pixels_per_frame': areas,
                    'visual_tokens': sum(t*h*w//4 for t, h, w in grids)}


class InputBackend:
    """Reuse our NF4 model and constrained decoding; official HF video processing."""
    def __init__(self, config, model_path):
        from scripts.train_agent_sft import QuantizedBackend
        self.base = QuantizedBackend({'seed': config['seed'], 'model_path': str(model_path),
            'min_pixels': 65536, 'max_pixels': 65536, 'attention': 'sdpa',
            'max_new_tokens': config['max_output_tokens'], 'max_seconds_per_generation': 120,
            'max_consecutive_errors': 3})
        self.calls = self.base.calls

    def __call__(self, prompt, frames, schema, condition, duration):
        if condition['input'] == 'images':
            self.base.processor.image_processor.size = {'shortest_edge': condition['pixels'], 'longest_edge': condition['pixels']}
            answer = self.base(prompt, frames, schema)
            record = self.calls[-1]
            record['visual_tokens'] = sum(t*h*w//4 for t, h, w in record['image_grid_thw'])
            record['input_mode'] = 'images'
            return answer
        base = self.base
        if base.errors >= base.cfg['max_consecutive_errors']:
            raise RuntimeError('backend_circuit_open')
        started = time.perf_counter()
        record = {'images': len(frames), 'input_mode': condition['input'], 'input_tokens': 0,
                  'output_tokens': 0, 'visual_tokens': 0, 'status': 'error'}
        try:
            if condition['input'] == 'video':
                inputs, audit = native_inputs(base.processor, prompt, frames, condition['pixels'], duration)
                record.update(audit)
            else:
                content = [{'type': 'text', 'text': prompt}]
                text = base.processor.apply_chat_template([{'role': 'user', 'content': content}],
                       tokenize=False, add_generation_prompt=True)
                inputs = base.processor(text=[text], return_tensors='pt')
            inputs = inputs.to('cuda:0')
            record['input_tokens'] = int(inputs.input_ids.shape[-1])
            record['processor_elapsed_sec'] = time.perf_counter()-started
            base.torch.cuda.reset_peak_memory_stats()
            prefix = base._prefix_constraint(schema)
            record['generation_attempted'] = True
            with base.torch.inference_mode():
                output = base.model.generate(**inputs, do_sample=False,
                    max_new_tokens=base.cfg['max_new_tokens'], max_time=base.cfg['max_seconds_per_generation'],
                    prefix_allowed_tokens_fn=prefix)
            tokens = output[0, inputs.input_ids.shape[-1]:]
            record['output_tokens'] = int(tokens.shape[-1])
            record['peak_allocated_bytes'] = base.torch.cuda.max_memory_allocated()
            result = json.loads(base.processor.tokenizer.decode(tokens, skip_special_tokens=True))
            record['status'] = 'complete'
            base.errors = 0
            return result
        except Exception as exc:
            base.errors += 1
            record['error_type'] = type(exc).__name__
            raise
        finally:
            record['elapsed_sec'] = time.perf_counter()-started
            self.calls.append(record)


def run(prepared, config_path, output, model_path=None, mock=False):
    config = json.loads(Path(config_path).read_text('utf-8'))
    rows = json.loads(Path(prepared).read_text('utf-8'))
    manifest = json.loads(Path(prepared).with_name('manifest.json').read_text('utf-8'))
    if manifest['config_sha256'] != sha256_file(config_path) or manifest['prepared_sha256'] != sha256_file(prepared):
        raise ValueError('Prepared/config hash mismatch')
    validate_prepared(rows, config)
    snapshot = baseline.verify_model_snapshot(model_path, config['model_revision']) if not mock else None
    output = baseline.checked_output(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'run_manifest.json', {'config': config, 'config_sha256': sha256_file(config_path),
          'prepared_sha256': sha256_file(prepared), 'runner_sha256': sha256_file(Path(__file__)),
          'model_manifest_sha256': snapshot, 'mock': mock,
          'input_policy': 'Prepared frames only; no evaluator labels loaded during inference'})
    started = time.perf_counter()
    backend = None if mock else InputBackend(config, model_path)
    load_sec = time.perf_counter()-started
    conditions = {c['name']: c for c in config['conditions']}
    results = []
    for row in rows:
        request = row['request']
        condition = conditions[row['condition']]
        frames = [VideoFrame.model_validate(f) for f in row['frames']]
        prompt, schema = prompt_and_schema(request, condition)
        before = len(backend.calls) if backend else 0
        started = time.perf_counter()
        result = {'condition': condition['name'], 'scope': condition['scope'], 'qid': request['qid'],
                  'video_id': request['video_id'], 'status': 'error', 'mock': mock,
                  'source_times_sec': [f.timestamp_sec for f in frames],
                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            if mock:
                prediction = {'answer': 'A'}
                if condition['output'] == 'grounded':
                    prediction['intervals'] = [[frames[0].timestamp_sec, frames[-1].timestamp_sec]]
            else:
                prediction = backend(prompt, frames, schema, condition, request['duration_sec'])
            if set(prediction) != set(schema['required']) or prediction['answer'] not in schema['properties']['answer']['enum']:
                raise ValueError('Invalid output schema')
            if condition['output'] == 'grounded':
                if not 1 <= len(prediction['intervals']) <= 5:
                    raise ValueError('Invalid interval count')
                checked_intervals(prediction['intervals'], request['duration_sec'])
            result.update(prediction, status='ok')
        except Exception as exc:
            result['error_type'] = type(exc).__name__
        result.update(elapsed_sec=time.perf_counter()-started,
                      model_calls=backend.calls[before:] if backend else [])
        results.append(result)
        with (output/'predictions.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(result, ensure_ascii=False)+'\n')
        print(json.dumps({'finished': len(results), 'condition': result['condition'], 'qid': result['qid'], 'status': result['status']}), flush=True)
    summary = {'scheduled': len(rows), 'completed': sum(r['status'] == 'ok' for r in results),
               'failed': sum(r['status'] != 'ok' for r in results), 'mock': mock, 'model_load_sec': load_sec,
               'backend_calls': len(backend.calls) if backend else 0, 'accuracy': None}
    write(output/'run_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.requests, args.labels, args.config, args.output)
    else:
        if args.labels is not None or args.requests is not None:
            parser.error('Inference accepts prepared observations only')
        print(json.dumps(run(args.prepared, args.config, args.output, args.model_path, args.mock)))


if __name__ == '__main__':
    main()
