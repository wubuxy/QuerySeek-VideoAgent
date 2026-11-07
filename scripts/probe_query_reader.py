"""Frozen question-only plans, oracle interface controls, and candidate-ID answers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from queryseek.agent.tree_observations import SamplingConfig, WindowObservationStore, output_path, write_json
from queryseek.eval.grounded_qa import interval_set_iou
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.retrieval.curve_spans import curve_spans, select_curve_windows
from queryseek.retrieval.query_refinement import select_review_frames
from queryseek.retrieval.query_reader import (METHODS, ORACLE_CONDITIONS, bind_answer, candidates_for,
    letter_token_ids, planner_prompt, query_scores, reader_prompt, token_chunks, validate_plan)
from queryseek.schemas import VideoFrame
from scripts import grounded_qa_input_study as study
from scripts import probe_query_refinement as previous
from scripts.run_grounded_qa_baseline import check_requests, verify_model_snapshot


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text('utf-8').splitlines()]


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False)+'\n')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sources():
    return {**previous.source_hashes(), **{n: sha256_file(ROOT/n) for n in [
        'scripts/probe_query_reader.py', 'queryseek/retrieval/query_reader.py',
        'scripts/run_grounded_qa_baseline.py']}}


def checked(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Frozen input hash mismatch')
    return read(path)


def requests_for(path, config):
    result = check_requests(checked(path, config['requests_sha256']))
    if len(result) != config['sample_count'] or config['max_generations'] != 7*len(result):
        raise ValueError('Frozen generation budget mismatch')
    return result


def begin(output, config_path, mock, dependencies=None):
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.initial.json', {'config_sha256': sha256_file(config_path),
        'source_sha256': sources(), 'mock': mock, 'dependencies': dependencies or {}})
    return output


def finish(output, names, summary):
    write_json(output/'summary.json', summary)
    write_json(output/'manifest.json', {**read(output/'manifest.initial.json'),
        'files': {n: sha256_file(output/n) for n in [*names, 'summary.json']}})
    return summary


def verify_stage(folder, config_path, mock):
    manifest = read(folder/'manifest.json')
    if (manifest['config_sha256'] != sha256_file(config_path) or manifest['mock'] != mock
            or manifest['source_sha256'] != sources()):
        raise ValueError('Stage provenance/mock mismatch')
    for name, value in manifest['files'].items():
        if sha256_file(folder/name) != value:
            raise ValueError('Stage artifact changed')
    return manifest


class ReaderBackend(study.InputBackend):
    def letter(self, prompt, frames, letters, duration, pixels):
        base = self.base
        if base.errors >= base.cfg['max_consecutive_errors']:
            raise RuntimeError('backend_circuit_open')
        started = time.perf_counter()
        record = {'images': len(frames), 'input_mode': 'video', 'input_tokens': 0,
                  'output_tokens': 0, 'visual_tokens': 0, 'status': 'error', 'decoding': 'one_letter_token'}
        try:
            allowed = letter_token_ids(base.processor.tokenizer, letters)
            inputs, audit = study.native_inputs(base.processor, prompt, frames, pixels, duration)
            record.update(audit)
            inputs = inputs.to('cuda:0')
            record['input_tokens'] = int(inputs.input_ids.shape[-1])
            record['processor_elapsed_sec'] = time.perf_counter()-started
            base.torch.cuda.reset_peak_memory_stats()
            record['generation_attempted'] = True
            with base.torch.inference_mode():
                output = base.model.generate(**inputs, do_sample=False, max_new_tokens=1,
                    max_time=base.cfg['max_seconds_per_generation'],
                    prefix_allowed_tokens_fn=lambda batch_id, ids: allowed)
            tokens = output[0, inputs.input_ids.shape[-1]:]
            record['output_tokens'] = int(tokens.shape[-1])
            record['peak_allocated_bytes'] = base.torch.cuda.max_memory_allocated()
            answer = base.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            if answer not in letters or len(tokens) != 1:
                raise ValueError('Invalid letter token')
            record['status'] = 'complete'
            base.errors = 0
            return {'answer': answer}
        except Exception as exc:
            base.errors += 1
            record['error_type'] = type(exc).__name__
            raise
        finally:
            record['elapsed_sec'] = time.perf_counter()-started
            self.calls.append(record)


def load_backend(config, model_path, output, mock):
    model_hash = None if mock else verify_model_snapshot(model_path, config['model_revision'])
    if not mock and model_hash != config['model_manifest_sha256']:
        raise ValueError('Model manifest changed')
    write_json(output/'model.json', {'manifest_sha256': model_hash, 'mock': mock})
    started = time.perf_counter()
    backend = None if mock else ReaderBackend(config, model_path)
    return backend, time.perf_counter()-started


def validate_frames(row, count):
    request = check_requests([row['request']])[0]
    frames = [VideoFrame.model_validate(f) for f in row['frames']]
    if len(frames) != count or len(row['frame_sha256']) != count or count % 2:
        raise ValueError('Frozen frame count mismatch')
    study.source_pts_metadata([f.timestamp_sec for f in frames], request['duration_sec'])
    for frame, expected in zip(frames, row['frame_sha256']):
        output_path(frame.image_path)
        if sha256_file(frame.image_path) != expected:
            raise ValueError('Prepared frame changed')
    return frames


def infer_row(backend, row, config, mock):
    request, condition = row['request'], row['condition']
    frames = validate_frames(row, config['review_frames'])
    prompt, schema, letter_only = reader_prompt(request, condition, row.get('candidates'))
    result = {'qid': request['qid'], 'condition': condition, 'status': 'error', 'mock': mock,
        'prompt_sha256': digest(prompt), 'source_times_sec': [f.timestamp_sec for f in frames],
        'frame_sha256': row['frame_sha256'], 'candidates': row.get('candidates')}
    before = len(backend.calls) if backend else 0
    started = time.perf_counter()
    try:
        if condition in METHODS and not row['candidates']:
            prediction = {'answer': 'abstain', 'candidate_id': 'none'}
            result['deterministic_no_candidate'] = True
        elif mock:
            prediction = {'answer': 'A'}
            if condition in METHODS:
                prediction['candidate_id'] = row['candidates'][0]['candidate_id']
        elif letter_only:
            prediction = backend.letter(prompt, frames, schema['properties']['answer']['enum'],
                                        request['duration_sec'], config['pixels'])
        else:
            prediction = backend(prompt, frames, schema, {'input': 'video', 'pixels': config['pixels']},
                                 request['duration_sec'])
        result['raw_prediction'] = prediction
        if condition in METHODS:
            prediction = bind_answer(prediction, row['candidates'], request['choices'], request['duration_sec'])
        elif set(prediction) != {'answer'} or prediction['answer'] not in schema['properties']['answer']['enum']:
            raise ValueError('Invalid oracle answer')
        result.update(prediction, status='ok')
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    result.update(elapsed_sec=time.perf_counter()-started, model_calls=backend.calls[before:] if backend else [])
    return result


def plan_oracle(requests_path, oracle_prepared, config_path, output, model_path=None, mock=False):
    config = read(config_path)
    requests = requests_for(requests_path, config)
    output = begin(output, config_path, mock)
    backend, load_sec = load_backend(config, model_path, output, mock)
    plans = []
    for request in requests:
        prompt, schema = planner_prompt(request['question'])
        result = {'qid': request['qid'], 'status': 'error', 'mock': mock, 'prompt_sha256': digest(prompt)}
        before = len(backend.calls) if backend else 0
        try:
            plan = ({'queries': ['visible scene'], 'temporal_requirement': 'none', 'answer_focus': 'visible detail'}
                    if mock else backend(prompt, [], schema, {'input': 'text'}, request['duration_sec']))
            result['raw_prediction'] = plan
            result['plan'] = validate_plan(plan)
            result['status'] = 'ok'
        except Exception as exc:
            result['error_type'] = type(exc).__name__
        result['model_calls'] = backend.calls[before:] if backend else []
        plans.append(result)
        append(output/'plans.jsonl', result)
        print(json.dumps({'stage': 'plan', 'finished': len(plans), 'status': result['status']}), flush=True)
    # This freeze occurs before opening any oracle observations or evaluator labels.
    write_json(output/'plans.freeze.json', {'plans_sha256': sha256_file(output/'plans.jsonl'),
        'requests_sha256': sha256_file(requests_path), 'question_only': True})
    oracle = [r for r in checked(oracle_prepared, config['oracle_prepared_sha256']) if r['condition'] == 'oracle_video16']
    request_lookup = {r['qid']: r for r in requests}
    if len(oracle) != len(requests) or {r['request']['qid'] for r in oracle} != set(request_lookup):
        raise ValueError('Oracle identity mismatch')
    prepared, predictions = [], []
    for old in oracle:
        if old['request'] != request_lookup[old['request']['qid']]:
            raise ValueError('Oracle request changed')
        for condition in ORACLE_CONDITIONS:
            row = {**old, 'condition': condition}
            prepared.append(row)
            result = infer_row(backend, row, config, mock)
            predictions.append(result)
            append(output/'oracle_predictions.jsonl', result)
            print(json.dumps({'stage': 'oracle', 'finished': len(predictions), 'status': result['status']}), flush=True)
    write_json(output/'oracle_prepared.json', prepared)
    return finish(output, ['plans.jsonl', 'plans.freeze.json', 'oracle_prepared.json', 'oracle_predictions.jsonl', 'model.json'],
        {'mock': mock, 'plan_scheduled': len(requests), 'plans_ok': sum(p['status']=='ok' for p in plans),
         'oracle_scheduled': len(predictions), 'oracle_ok': sum(p['status']=='ok' for p in predictions),
         'model_load_sec': load_sec, 'backend_calls': len(backend.calls) if backend else 0})


def prepare(requests_path, plan_dir, old_retrieval, config_path, output, clip_manifest=None, weights=None, mock=False):
    config = read(config_path)
    requests = requests_for(requests_path, config)
    verify_stage(plan_dir, config_path, mock)
    plans = {r['qid']: r for r in lines(plan_dir/'plans.jsonl')}
    if set(plans) != {r['qid'] for r in requests}:
        raise ValueError('Plan identities changed')
    old = {}
    if not mock:
        if sha256_file(old_retrieval) != config['old_retrieval_sha256']:
            raise ValueError('Historical retrieval changed')
        old = {r['qid']: r for r in lines(old_retrieval) if r['method']=='dense_uniform'}
    output = begin(output, config_path, mock, {'plans': sha256_file(plan_dir/'manifest.json')})
    started = time.perf_counter()
    encoder, signature, torch = previous.encoder_for(config, clip_manifest, weights, mock)
    load_sec = time.perf_counter()-started
    start_text = time.perf_counter()
    raw_vectors = encoder.encode_texts([r['question'] for r in requests])
    text_sec = time.perf_counter()-start_text
    encoded_texts = len(requests)
    prepared, records, costs = [], [], []
    for request, raw in zip(requests, raw_vectors):
        asset = asset_from_probe(request['video_id'], request['video_path'])
        if abs(asset.duration_sec-request['duration_sec']) > .001:
            raise ValueError('Probed duration changed')
        observation = WindowObservationStore(asset.path, asset.duration_sec, asset.sha256, encoder, signature,
                                             SamplingConfig(**config['sampling']))(0., asset.duration_sec)
        costs.append({'qid': request['qid'], **observation.cost})
        if not mock and (observation.cost['encoded_images'] or observation.cost['new_frames_extracted']):
            raise ValueError('This study requires cached observations only')
        times = [f.timestamp_sec for f in observation.frames]
        for method in METHODS:
            row = {'qid': request['qid'], 'method': method, 'status': 'error', 'mock': mock}
            try:
                if method == 'raw':
                    texts, vectors = [request['question']], [raw]
                elif method == 'token_chunks':
                    texts = ([request['question']] if mock else token_chunks(request['question'], encoder._tokenizer,
                             config['chunk_tokens'], config['max_chunks']))
                    if texts == [request['question']]:
                        vectors = [raw]
                    else:
                        start_text = time.perf_counter(); vectors = encoder.encode_texts(texts)
                        text_sec += time.perf_counter()-start_text; encoded_texts += len(texts)
                else:
                    if plans[request['qid']]['status'] != 'ok':
                        raise ValueError('Planner failure retained')
                    texts = validate_plan(plans[request['qid']]['plan'])['queries']
                    if not mock and any(len(encoder._tokenizer.encode(t))+2 > encoder._tokenizer.context_length for t in texts):
                        raise ValueError('Event query exceeds CLIP context')
                    start_text = time.perf_counter(); vectors = encoder.encode_texts(texts)
                    text_sec += time.perf_counter()-start_text; encoded_texts += len(texts)
                scores = query_scores(observation, vectors)
                windows = select_curve_windows(curve_spans(times, scores, asset.duration_sec),
                                                top_k=config['top_k'], nms_iou=config['nms_iou'])
                frames, fallback = select_review_frames(observation, scores, windows, config['review_frames'])
                if method == 'raw' and not mock:
                    baseline = old[request['qid']]
                    if (times != baseline['observed_source_times_sec']
                            or not np.allclose(scores, baseline['scores'], atol=2e-6, rtol=0)
                            or [f.timestamp_sec for f in frames] != baseline['review_source_times_sec']
                            or [(w.start_sec,w.end_sec) for w in windows] != [(w['start_sec'],w['end_sec']) for w in baseline['windows']]):
                        raise ValueError('Raw baseline replay mismatch')
                    row['raw_replay_max_abs'] = float(np.max(np.abs(scores-np.array(baseline['scores']))))
                candidates = candidates_for(windows, frames, asset.duration_sec, config['candidate_limit'])
                row.update(status='ok', queries=texts, observed_source_times_sec=times, scores=scores.tolist(),
                    windows=[w.model_dump() for w in windows], review_source_times_sec=[f.timestamp_sec for f in frames],
                    review_fallback_frames=fallback, candidates=candidates, evidence_status='unverified',
                    token_lengths=None if mock else [len(encoder._tokenizer.encode(t))+2 for t in texts])
                prepared.append({'request': request, 'condition': method, 'frames': [f.model_dump() for f in frames],
                    'frame_sha256': [sha256_file(f.image_path) for f in frames], 'candidates': candidates})
            except Exception as exc:
                row['error_type'] = type(exc).__name__
            records.append(row); append(output/'retrieval.jsonl', row)
            print(json.dumps({'stage': 'retrieval', 'finished': len(records), 'status': row['status']}), flush=True)
    write_json(output/'prepared.json', prepared)
    return finish(output, ['retrieval.jsonl', 'prepared.json'], {'mock': mock,
        'scheduled': len(requests)*3, 'completed': len(prepared), 'model_load_sec': load_sec,
        'text_encoding_sec': text_sec, 'encoded_texts': encoded_texts, 'physical_observations': costs,
        'new_encoded_images': sum(c['encoded_images'] for c in costs),
        'new_frames_extracted': sum(c['new_frames_extracted'] for c in costs), 'encoder': signature})


def answer(requests_path, prepared_dir, config_path, output, model_path=None, mock=False):
    config = read(config_path)
    requests = requests_for(requests_path, config)
    verify_stage(prepared_dir, config_path, mock)
    prepared = read(prepared_dir/'prepared.json')
    lookup = {(r['request']['qid'], r['condition']): r for r in prepared}
    if len(lookup) != len(prepared):
        raise ValueError('Duplicate prepared row')
    output = begin(output, config_path, mock, {'prepared': sha256_file(prepared_dir/'manifest.json')})
    backend, load_sec = load_backend(config, model_path, output, mock)
    results = []
    for request in requests:
        for method in METHODS:
            row = lookup.get((request['qid'], method))
            if row is None:
                result = {'qid': request['qid'], 'condition': method, 'status': 'error', 'mock': mock,
                          'error_type': 'RetrievalFailure', 'model_calls': []}
            else:
                if row['request'] != request:
                    raise ValueError('Prepared request changed')
                result = infer_row(backend, row, config, mock)
            results.append(result); append(output/'predictions.jsonl', result)
            print(json.dumps({'stage': 'answer', 'finished': len(results), 'status': result['status']}), flush=True)
    return finish(output, ['predictions.jsonl', 'model.json'], {'mock': mock, 'scheduled': len(results),
        'completed': sum(r['status']=='ok' for r in results), 'backend_calls': len(backend.calls) if backend else 0,
        'model_load_sec': load_sec})


def score_bound(prediction, label, duration):
    ok = prediction['status'] == 'ok'
    accepted = ok and not prediction.get('abstained', True)
    correct = accepted and prediction['answer'] == label['right_answer']
    iou = interval_set_iou(prediction['intervals'], label['gt_intervals_sec'], duration) if accepted else 0.
    return {'valid': int(ok), 'accepted': int(accepted), 'correct': int(correct), 'iou': iou,
            'joint_at_05': int(correct and iou >= .5)}


def matrix(rows, ids, field, conditions):
    expected = {(q,c) for q in ids for c in conditions}
    if len(rows) != len(expected) or {(r['qid'],r[field]) for r in rows} != expected:
        raise ValueError('Incomplete/duplicate result matrix')
    return {(r['qid'],r[field]):r for r in rows}


def evaluate(requests_path, labels_path, plan_dir, prepared_dir, run_dir, old_predictions, config_path, output):
    config = read(config_path); requests = requests_for(requests_path, config)
    for folder in [plan_dir, prepared_dir, run_dir]:
        verify_stage(folder, config_path, False)
    labels = checked(labels_path, config['labels_sha256'])
    lookup = {r['qid']:r for r in labels}; ids = [r['qid'] for r in requests]
    if len(lookup) != len(labels) or set(lookup) != set(ids):
        raise ValueError('Label identities mismatch')
    retrieval = matrix(lines(prepared_dir/'retrieval.jsonl'), ids, 'method', METHODS)
    predictions = matrix(lines(run_dir/'predictions.jsonl'), ids, 'condition', METHODS)
    oracle = matrix(lines(plan_dir/'oracle_predictions.jsonl'), ids, 'condition', ORACLE_CONDITIONS)
    if sha256_file(old_predictions) != config['oracle_predictions_sha256']:
        raise ValueError('Old oracle predictions changed')
    old = [p for p in lines(old_predictions) if p['condition']=='oracle_video16']
    matrix(old, ids, 'condition', ['oracle_video16'])
    methods, records, oracle_metrics = {}, {}, {}
    for condition in ['oracle_video16', *ORACLE_CONDITIONS]:
        rows = old if condition == 'oracle_video16' else [oracle[(q,condition)] for q in ids]
        correct_ids = [p['qid'] for p in rows if p['status']=='ok' and p['answer']==lookup[p['qid']]['right_answer']]
        oracle_metrics[condition] = {'n': len(ids), 'correct': len(correct_ids), 'correct_ids': correct_ids,
                                    'valid_outputs': sum(p['status']=='ok' for p in rows), 'reused': condition=='oracle_video16'}
    for method in METHODS:
        scored = []
        for request in requests:
            q, duration = request['qid'], request['duration_sec']
            r, p, label = retrieval[(q,method)], predictions[(q,method)], lookup[q]
            item = {'qid': q, **score_bound(p,label,duration), 'observed_hit': 0, 'review_hit': 0,
                    'rank1_iou': 0., 'top5_best_iou': 0., 'top5_r05': 0}
            if r['status'] == 'ok':
                gt = label['gt_intervals_sec']
                for source, target in [('observed_source_times_sec','observed_hit'),('review_source_times_sec','review_hit')]:
                    item[target] = int(any(a<=t<b for t in r[source] for a,b in gt))
                ious = [interval_set_iou([[w['start_sec'],w['end_sec']]],gt,duration) for w in r['windows']]
                item.update(rank1_iou=ious[0] if ious else 0., top5_best_iou=max(ious,default=0.),
                            top5_r05=int(max(ious,default=0.)>=.5))
            scored.append(item)
        records[method] = scored
        methods[method] = {'n': len(ids), **{k: sum(r[k] for r in scored)/len(ids) for k in scored[0] if k!='qid'}}
        accepted = sum(r['accepted'] for r in scored)
        methods[method]['accepted_correct_fraction'] = sum(r['correct'] for r in scored)/accepted if accepted else None
    plans = lines(plan_dir/'plans.jsonl')
    calls = [c for p in [*plans,*oracle.values(),*predictions.values()] for c in p['model_calls']]
    if len(calls) > config['max_generations']:
        raise ValueError('Generation budget exceeded')
    result = {'scope': 'Eight repeatedly used development sources; no independent confirmation',
        'methods': methods, 'oracle': oracle_metrics, 'records': records,
        'plans_ok': sum(p['status']=='ok' for p in plans),
        'costs': {'backend_calls': len(calls), 'reused_baseline_calls': len(old),
            **{k: sum(c.get(k,0) for c in calls) for k in ['input_tokens','output_tokens','images','elapsed_sec']},
            'peak_allocated_bytes': max((c.get('peak_allocated_bytes',0) for c in calls),default=0),
            'model_load_sec': sum(read(f/'summary.json')['model_load_sec'] for f in [plan_dir,run_dir])},
        'retrieval_costs': read(prepared_dir/'summary.json')}
    output = begin(output, config_path, False, {str(f.name):sha256_file(f/'manifest.json') for f in [plan_dir,prepared_dir,run_dir]})
    write_json(output/'evaluation.json', result)
    finish(output, ['evaluation.json'], {'completed': True, 'new_model_calls': 0})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan-oracle','prepare','answer','evaluate'])
    for name in ['config','output','requests','oracle-prepared','model-path','plan-dir','old-retrieval',
                 'clip-manifest','weights','prepared-dir','run-dir','labels','old-predictions']:
        parser.add_argument('--'+name, type=Path, required=name in ['config','output','requests'])
    parser.add_argument('--mock', action='store_true'); a=parser.parse_args()
    if a.action != 'evaluate' and a.labels is not None:
        parser.error('Only evaluator accepts labels')
    if a.action == 'plan-oracle':
        result=plan_oracle(a.requests,a.oracle_prepared,a.config,a.output,a.model_path,a.mock)
    elif a.action == 'prepare':
        result=prepare(a.requests,a.plan_dir,a.old_retrieval,a.config,a.output,a.clip_manifest,a.weights,a.mock)
    elif a.action == 'answer':
        result=answer(a.requests,a.prepared_dir,a.config,a.output,a.model_path,a.mock)
    else:
        result=evaluate(a.requests,a.labels,a.plan_dir,a.prepared_dir,a.run_dir,a.old_predictions,a.config,a.output)
        result={k:v for k,v in result.items() if k not in ['records','retrieval_costs']}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
