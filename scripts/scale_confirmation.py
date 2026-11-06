"""Prospective source holdout and fixed-model data scaling, reusing frozen code."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import localizer_study as data
from scripts import multiscale_ranker_study as ranker
from scripts.analyze_localizer_study import per_query, paired_clusters, candidate_diagnostic
from scripts.make_qvhighlights_subset import source_video_id
from scripts.sampling_probe import checked_output, optional_version
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.candidate_ranker import candidate_table, training_targets
from queryseek.retrieval.multiscale_candidates import multiscale_table
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.tools.export import new_job_id

CONFIG = 'configs/scale_confirmation_v1.json'
PARENT_MODELS = {'full_256': 'union_full_base', 'full_384': 'union_full_expanded',
                 'structure_384': 'union_structure_expanded'}


def source_set(entries):
    return {source_video_id(e['vid']) for e in entries}


def holdout_partition(entries, old_entries, train_entries, excluded):
    """Identity-only partition; all queries of every selected source stay together."""
    if len({e['qid'] for e in entries}) != len(entries):
        raise ValueError('Duplicate query identity')
    old_sources = source_set(old_entries)
    old = [e for e in entries if source_video_id(e['vid']) in old_sources]
    new = [e for e in entries if source_video_id(e['vid']) not in old_sources]
    if old != old_entries:
        raise ValueError('Old validation prefix changed')
    if source_set(entries) & (source_set(train_entries) | set(excluded)):
        raise ValueError('Holdout source overlap')
    if {e['qid'] for e in entries} & {e['qid'] for e in train_entries}:
        raise ValueError('Holdout query overlap')
    return old, new


def freeze(cfg, out):
    parent = ROOT/cfg['parent_data']; pp = data.verify(parent)
    pr = ROOT/cfg['parent_ranker']; rp = ranker.verify(pr)
    if data.read(pr/'status.json')['phase'] != 'complete':
        raise ValueError('Completed frozen parent required')
    # Verify saved parents before loading any models or extending identities.
    for name, sha in data.read(pr/'metrics.json')['artifact_sha256'].items():
        if sha256_file(pr/name) != sha:
            raise ValueError('Parent artifact changed: '+name)
    c = {**pp['config'], 'version': cfg['version'], 'train_scales': [384, cfg['planned_train_sources']],
         'validation_sources': cfg['old_validation_sources']+cfg['new_validation_sources'],
         'scope': cfg['scope'], 'variants': [], 'checkpoint_selection': cfg['stop_rule']}
    out = data.freeze(c, out)
    p = data.read(out/'protocol.json')
    if sha256_file(out/'train_384.jsonl') != sha256_file(parent/'train_384.jsonl'):
        raise ValueError('Training prefix changed')
    all_train = data.rows(out/f'train_{cfg["planned_train_sources"]}.jsonl')
    excluded = data.read(ROOT/c['data_run']/'excluded_sources.json')
    old, new = holdout_partition(data.rows(out/'validation.jsonl'), data.rows(parent/'validation.jsonl'), all_train, excluded)
    if len(source_set(old)) != cfg['old_validation_sources'] or len(source_set(new)) != cfg['new_validation_sources']:
        raise ValueError('Holdout source count changed')
    for name, entries in [('old_validation', old), ('new_validation', new)]:
        file = out/(name+'.jsonl'); file.write_text(''.join(json.dumps(e)+'\n' for e in entries), encoding='utf-8')
        p['selection_sha256'][file.name] = sha256_file(file)
    preserved = [parent/n for n in ('protocol.json', 'feature_records.jsonl', 'features_summary.json', 'availability.jsonl')]
    preserved += [pr/n for n in ('protocol.json', 'metrics.json', 'candidate_manifest.json')]
    preserved += [pr/'models'/(n+'.joblib') for n in PARENT_MODELS.values()]
    preserved += [pr/('validation_'+n+'_predictions.json') for n in ['e4', *PARENT_MODELS.values()]]
    p['preserved_artifacts_sha256'].update({f.relative_to(ROOT).as_posix(): sha256_file(f) for f in preserved})
    names = [CONFIG, 'scripts/scale_confirmation.py', 'tests/test_scale_confirmation.py',
             'scripts/multiscale_ranker_study.py', 'scripts/analyze_localizer_study.py',
             'scripts/candidate_ranker_study.py', 'queryseek/retrieval/candidate_ranker.py',
             'queryseek/retrieval/multiscale_candidates.py', 'queryseek/retrieval/curve_spans.py',
             'queryseek/tools/paths.py', 'scripts/sampling_probe.py']
    for name in names:
        dst = out/'source_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dst); p['source_sha256'][name] = sha256_file(ROOT/name)
    p.update(confirmation=cfg, ranker_config=rp['config'], model=rp['model'],
             max_crops=cfg['planned_train_sources']+c['validation_sources'],
             max_video_bytes_total=(cfg['planned_train_sources']+c['validation_sources'])*c['max_video_bytes'],
             prior_models=PARENT_MODELS, fresh_query_label_access=False,
             holdout_claim='Previously unused source holdout within official train; not external benchmark',
             inference_label_policy='No holdout training targets; predictions persisted before metric evaluation; labels used only for acquisition validity and evaluation')
    write_json_atomic(out/'protocol.json', p)
    # Resume existing immutable work rather than downloading/encoding the prefix again.
    shutil.copyfile(parent/'availability.jsonl', out/'availability.jsonl')
    shutil.copyfile(parent/'feature_records.jsonl', out/'feature_records.jsonl')
    write_json_atomic(out/'research_status.json', {'phase': 'frozen', 'new_fits': 0})
    print('OUTPUT='+str(out), flush=True)
    return out


def verify(out):
    p = data.verify(out)
    for name, sha in p['source_sha256'].items():
        if sha256_file(out/'source_snapshot'/name) != sha:
            raise ValueError('Snapshot changed: '+name)
    for name, sha in p['data_sha256'].items():
        if sha256_file(ROOT/p['config']['data_run']/name) != sha:
            raise ValueError('Source split metadata changed')
    return p


def build_tables(records, cfg):
    """Evaluation records need qid/video, never relevant_windows."""
    arrays, items = {}, []
    for r in records:
        e = r['entry']; q = e['qid']; scores = r['images']@r['query_embedding']
        for pool in ('e4', 'union'):
            w, x = (candidate_table(r['times'], scores, r['duration_sec']) if pool == 'e4' else
                    multiscale_table(r['times'], scores, r['duration_sec'], cfg['lengths_sec'],
                                     cfg['stride_fraction'], cfg['min_stride_sec']))
            arrays[f'{pool}_w_{q}'] = w; arrays[f'{pool}_x_{q}'] = x
            if r['split'] == 'train' and pool == 'union':
                arrays[f'{pool}_y_{q}'] = training_targets(w, e['relevant_windows'], r['duration_sec'])
        items.append({'qid': q, 'source': source_video_id(e['vid']), 'split': r['split'], 'duration_sec': r['duration_sec']})
    return items, arrays


def complete_denominator(predictions, entries):
    expected = {e['qid'] for e in entries}
    if set(predictions)-expected:
        raise ValueError('Unexpected predicted query')
    return {q: predictions.get(q, []) for q in sorted(expected)}


def predict(out, p, items, arrays, replay=False, parent_models=None):
    cfg = p['ranker_config']; train = [r for r in items if r['split'] == 'train']
    val = [r for r in items if r['split'] == 'validation']
    models = {} if parent_models is None else dict(parent_models)
    if parent_models is None:
        for name, filename in PARENT_MODELS.items():
            models[name] = joblib.load(ROOT/p['confirmation']['parent_ranker']/'models'/(filename+'.joblib'))
    for control in ('full', 'structure'):
        name = control+'_512'
        models[name] = ranker.fit_or_reload(out, name, train, arrays, 'union', control, p, replay)
    result = {'e4': ranker.infer(None, val, arrays, 'e4', 'full', cfg)}
    for name, model in models.items():
        result[name] = ranker.infer(model, val, arrays, 'union', 'structure' if name.startswith('structure') else 'full', cfg)
    return result


def persist_predictions(out, pred, entries, replay):
    result = {}
    for name, pp in pred.items():
        pp = complete_denominator(pp, entries)
        normalized = {str(q): [list(w) for w in windows] for q, windows in pp.items()}
        file = out/(name+'_predictions.json')
        if replay:
            if data.read(file) != normalized:
                raise ValueError('Prediction replay differs: '+name)
        else:
            write_json_atomic(file, normalized)
        result[name] = pp
    return result


def evaluate(out, p, predictions, arrays):
    results = {}
    contrasts = {'expanded_minus_e4': ('e4', 'full_512'), 'data_effect': ('full_384', 'full_512'),
                 'prior_model_minus_e4': ('e4', 'full_384'), 'semantics_effect': ('structure_512', 'full_512')}
    for stage in ('old_validation', 'new_validation'):
        gt = {e['qid']: e for e in data.rows(out/(stage+'.jsonl'))}
        pred = {name: {q: pp[q] for q in gt} for name, pp in predictions.items()}
        metrics = {name: dict(eval_mr_brief(pp, gt)) for name, pp in pred.items()}
        pairs = {}
        for name, (a, b) in contrasts.items():
            pairs[name] = paired_clusters(per_query(pred[a], gt), per_query(pred[b], gt), **p['confirmation']['bootstrap'])
            pairs[name]['scope'] = ('conditional source bootstrap for prospectively frozen holdout; single fit/seed, secondary contrasts unadjusted'
                                     if stage == 'new_validation' else 'previously observed internal validation, descriptive only')
        diagnostic = {}
        for pool in ('e4', 'union'):
            ws = {q: arrays.get(f'{pool}_w_{q}', np.empty((0, 3))).tolist() for q in gt}
            diagnostic[pool] = {**candidate_diagnostic(ws, pred['e4' if pool == 'e4' else 'full_512'], gt),
                                'event_coverage': ranker.coverage(ws, gt)}
        results[stage] = {'metrics': metrics, 'contrasts': pairs, 'diagnostic': diagnostic,
                          'queries': len(gt), 'sources': len(source_set(gt.values())),
                          'available_queries': sum(f'e4_w_{q}' in arrays for q in gt)}
    # Reused models must reproduce the previous validation exactly before new claims.
    parent = ROOT/p['confirmation']['parent_ranker']
    for name, oldname in {'e4': 'e4', **PARENT_MODELS}.items():
        old = data.read(parent/('validation_'+oldname+'_predictions.json'))
        current = {q: [list(w) for w in predictions[name][int(q)]] for q in old}
        if old != current:
            raise ValueError('Previous model/validation not reproduced: '+name)
    return results


def audit(out, p, items, arrays, predictions):
    metadata = data.rows(out/'feature_records.jsonl')
    old = data.rows(ROOT/p['confirmation']['parent_data']/'feature_records.jsonl')
    lookup = {r['entry']['qid']: r for r in metadata}
    if any(lookup[r['entry']['qid']] != r for r in old):
        raise ValueError('Historical feature metadata changed')
    train = data.rows(out/'train_512.jsonl'); validation = data.rows(out/'validation.jsonl')
    excluded = data.read(ROOT/p['config']['data_run']/'excluded_sources.json')
    holdout_partition(validation, data.rows(out/'old_validation.jsonl'), train, excluded)
    if source_set(train) & set(excluded):
        raise ValueError('Excluded training source')
    windows = predicted = 0
    for r in items:
        q = r['qid']; duration = r['duration_sec']
        for pool in ('e4', 'union'):
            ws = arrays[f'{pool}_w_{q}']; windows += len(ws)
            if not np.isfinite(ws).all() or np.any(ws[:, 0]<0) or np.any(ws[:, 0]>=ws[:, 1]) or np.any(ws[:, 1]>duration):
                raise ValueError('Invalid candidate time window')
        if not set(map(tuple, arrays[f'e4_w_{q}'])).issubset(set(map(tuple, arrays[f'union_w_{q}']))):
            raise ValueError('Original E4 candidates lost')
        if r['split'] == 'validation':
            if f'union_y_{q}' in arrays or f'e4_y_{q}' in arrays:
                raise ValueError('Holdout target leakage')
            for pp in predictions.values():
                for a, b, score in pp[q]:
                    if not np.isfinite([a, b, score]).all() or not 0 <= a < b <= duration:
                        raise ValueError('Invalid prediction time window')
                    predicted += 1
    old_paths = {r['images_path'] for r in old}; new_paths = {r['images_path'] for r in metadata}-old_paths
    visual = {r['images_path']: r for r in metadata}
    old_videos = {r['entry']['vid'] for r in old}
    availability = data.rows(out/'availability.jsonl')
    new_downloads = [r for r in availability if r['available'] and r['entry']['vid'] not in old_videos and not r['provenance']['cache_hit']]
    summary = {'passed': True, 'legal_candidate_windows': windows, 'legal_prediction_windows': predicted,
               'old_cache_records_identical': True, 'source_overlap': 0, 'excluded_overlap': 0,
               'fresh_identity_sha256': p['fresh_sha256'], 'fresh_queries_labels_used': False,
               'new_visual_frames': sum(visual[path]['n_frames'] for path in new_paths),
               'total_visual_frames': sum(r['n_frames'] for r in visual.values()),
               'new_visual_caches': len(new_paths), 'new_downloads': len(new_downloads),
               'new_download_bytes': sum(r['bytes'] for r in new_downloads),
               'new_text_cache_paths': len({r['text_path'] for r in metadata}-{r['text_path'] for r in old}),
               'train_sources': len({r['source'] for r in items if r['split']=='train'}),
               'train_queries': sum(r['split']=='train' for r in items),
               'quality_rejections': [{'vid': r['entry']['vid'], 'split': r['split'], 'reason': r.get('reason')} for r in availability if not r['available']],
               'versions': {m: optional_version(m) for m in ['torch', 'open-clip-torch', 'scikit-learn', 'numpy', 'joblib', 'threadpoolctl']}}
    verify(out)
    return summary


def run(out, replay=False):
    started = time.perf_counter(); p = verify(out)
    if replay:
        saved = data.read(out/'metrics.json')
        for name, sha in saved['artifact_sha256'].items():
            if sha256_file(out/name) != sha:
                raise ValueError('Saved artifact changed: '+name)
        items = data.read(out/'candidate_manifest.json')
        with np.load(out/'candidate_tables.npz', allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        # Validate current external feature files even when replay uses saved tables.
        data.load_records(out, p)
    else:
        if data.read(out/'research_status.json')['phase'] != 'frozen':
            raise ValueError('Already started; use replay')
        records = data.load_records(out, p)
        write_json_atomic(out/'research_status.json', {'phase': 'running', 'new_fits': 0})
        with threadpool_limits(limits=p['confirmation']['threads']):
            items, arrays = build_tables(records, p['ranker_config'])
        np.savez(out/'candidate_tables.npz', **arrays)
        write_json_atomic(out/'candidate_manifest.json', items)
    with threadpool_limits(limits=p['confirmation']['threads']):
        pred = predict(out, p, items, arrays, replay)
    pred = persist_predictions(out, pred, data.rows(out/'validation.jsonl'), replay)
    # Only now read held-out labels for metrics. No validation targets enter fit/infer.
    results = evaluate(out, p, pred, arrays)
    validation = audit(out, p, items, arrays, pred)
    if replay:
        if results != saved['results'] or validation != data.read(out/'validation_summary.json'):
            raise ValueError('Metric/audit replay differs')
        result = {'passed': True, 'new_fits': 0, 'models_reloaded': 5, 'predictions_metrics_intervals_exact': True,
                  'elapsed_sec': time.perf_counter()-started}
        write_json_atomic(out/('replay_'+new_job_id()+'.json'), result)
    else:
        write_json_atomic(out/'validation_summary.json', validation)
        result = {'results': results, 'new_fits': 2, 'parent_models_reused': 3, 'vlm_calls': 0,
                  'default_promotion': False, 'fresh_queries_labels_used': False,
                  'scope': p['confirmation']['scope'], 'elapsed_sec': time.perf_counter()-started,
                  'artifact_sha256': {f.relative_to(out).as_posix(): sha256_file(f) for f in out.rglob('*') if f.is_file() and
                     (f.parent.name=='models' or f.name.endswith('_predictions.json') or f.name in
                      ('candidate_tables.npz', 'candidate_manifest.json', 'feature_records.jsonl', 'features_summary.json', 'validation_summary.json'))}}
        write_json_atomic(out/'metrics.json', result)
        write_json_atomic(out/'research_status.json', {'phase': 'complete', 'new_fits': 2})
    print(json.dumps(result if replay else {'elapsed_sec': result['elapsed_sec'], 'results': results}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'prepare', 'encode', 'run', 'replay'])
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.phase == 'freeze':
        freeze(data.read(ROOT/CONFIG), ROOT/'outputs/scale_confirmation'/new_job_id())
        return
    if args.run is None:
        parser.error('--run required')
    out = checked_output(args.run); verify(out)
    started = time.perf_counter()
    if args.phase in ('prepare', 'encode'):
        {'prepare': data.prepare, 'encode': data.encode}[args.phase](out)
        write_json_atomic(out/(args.phase+'_timing_'+new_job_id()+'.json'), {'elapsed_sec': time.perf_counter()-started})
    else:
        run(out, args.phase == 'replay')


if __name__ == '__main__':
    main()
