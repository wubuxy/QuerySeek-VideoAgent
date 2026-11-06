"""Oracle event composition on pre-reviewed development clips, zero VLM.

All event labels/crops are frozen before feature inference. Outputs do not change
the online E4 strategy, baseline predictions or independent evaluation selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.mechanism_probe import midpoint_times, snapshot
from scripts.sampling_probe import checked_output, extra_features
from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.embeddings.base import MockEmbedding
from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
from queryseek.eval.runner import load_dataset
from queryseek.ft.montage import build_montage
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.ingest.pipeline import VideoIndex
from queryseek.retrieval.event_composition import sequence_time_controls
from queryseek.tools.export import new_job_id


def validate_chain(chain, duration):
    window = tuple(chain['window'])
    midpoint_times(window, duration)
    a, b = chain['event_a_window'], chain['event_b_window']
    midpoint_times(a, duration)
    midpoint_times(b, duration)
    if not window[0] <= a[0] < a[1] <= b[0] < b[1] <= window[1]:
        raise ValueError('Visible event phases must be ordered and within the oracle clip')
    if any(not isinstance(chain[k], str) or not chain[k].strip()
           for k in ('event_a', 'event_b', 'context_a', 'context_b')):
        raise ValueError('Nonempty event/context clauses required')
    return window


def phase_diagnosis(scores, times, chain):
    masks = [(np.asarray(times) >= w[0]) & (np.asarray(times) < w[1])
             for w in (chain['event_a_window'], chain['event_b_window'])]
    if any(int(mask.sum()) < 2 for mask in masks):
        raise RuntimeError('Visible event phase has fewer than two actual observations')
    return {'a_frames': int(masks[0].sum()), 'b_frames': int(masks[1].sum()),
            'a_vs_b_in_a_phase': float(np.mean(scores[masks[0], 0] - scores[masks[0], 1])),
            'b_vs_a_in_b_phase': float(np.mean(scores[masks[1], 1] - scores[masks[1], 0]))}


def summarize(rows, manifest, errors, mock):
    expected = len(manifest['chains'])
    epsilon = 1e-12
    margin = lambda r: r['controls']['original']['sequence_axis']['margin']
    context = lambda r: r['controls']['original']['context_sequence_axis']['margin']
    return {'mock': mock, 'oracle_only': True, 'independent': False, 'vlm_calls': 0,
            'reviewed': expected + len(manifest.get('excluded', [])), 'eligible': expected,
            'completed': len(rows), 'coverage': len(rows) / expected, 'errors': errors,
            'new_encoded_frames': sum(r['new_encoded_frames'] for r in rows),
            'full_sentence_positive': sum(r['full_sentence_margin'] > epsilon for r in rows),
            'composed_positive': sum(margin(r) > epsilon for r in rows),
            'composed_ties': sum(abs(margin(r)) <= epsilon for r in rows),
            'new_positive_vs_full': sum(margin(r) > epsilon and r['full_sentence_margin'] <= epsilon for r in rows),
            'lost_positive_vs_full': sum(margin(r) <= epsilon and r['full_sentence_margin'] > epsilon for r in rows),
            'context_positive': sum(context(r) > epsilon for r in rows),
            'both_visible_phases_semantically_positive': sum(r['phase_diagnosis']['a_vs_b_in_a_phase'] > epsilon
                                                            and r['phase_diagnosis']['b_vs_a_in_b_phase'] > epsilon for r in rows),
            'max_source_metadata_reconstruction_delta': max((abs(c['source_pts_axis']['margin'] - margin(r))
                                                            for r in rows for c in r['controls'].values()), default=0.),
            'max_reverse_antisymmetry_error': max((abs(r['controls']['reverse']['sequence_axis']['margin'] + margin(r))
                                                  for r in rows), default=0.),
            'extraction_sec': sum(r['extraction_sec'] for r in rows),
            'encoding_sec': sum(r['encoding_sec'] for r in rows),
            'interpretation': 'constructed small development oracle pairs; counts are not QVHighlights accuracy or full-video retrieval gains'}


def run_probe(manifest, cfg, out, mock=False, samples=None, hints=None):
    if not manifest['chains'] or len({c['qid'] for c in manifest['chains']}) != len(manifest['chains']):
        raise ValueError('Nonempty unique chain qids required')
    count = manifest['frames_per_chain']
    minimum = manifest['min_frames_per_event']
    midpoint_times((0., 1.), 1., count)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1 or count < 2 * minimum:
        raise ValueError('Invalid support budget')
    if samples is None:
        samples = load_dataset(ROOT / 'data/qvhighlights_test/dataset.json')
    by_q = {s.qid: s for s in samples}
    protocol = {'manifest': manifest, 'mock': mock, 'pipeline_version': PIPELINE_VERSION,
                'config': cfg.model_dump(mode='json'), 'vlm_calls': 0, 'oracle_only': True}
    if not mock:
        review = ROOT / manifest['review_run']
        reviewed = [json.loads(line) for line in (review / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
        review_protocol = json.loads((review / 'protocol.json').read_text(encoding='utf-8'))
        covered = {c['qid'] for c in manifest['chains']} | {c['qid'] for c in manifest['excluded']}
        if covered != {r['qid'] for r in reviewed}:
            raise ValueError('Every reviewed query must be retained or explicitly excluded')
        if review_protocol['dataset_sha256'] != sha256_file(ROOT / 'data/qvhighlights_test/dataset.json'):
            raise ValueError('Review dataset changed')
        protocol['review_hashes'] = {name: sha256_file(review / name) for name in ('protocol.json', 'records.jsonl')}
        protocol['fresh_selection_sha256'] = sha256_file(ROOT / 'data/qvhighlights_fresh80/selection.json')
    out = checked_output(out)
    out.mkdir(parents=True, exist_ok=False)
    protocol['source_sha256'] = snapshot(out, [*ROOT.glob('queryseek/**/*.py'),
        ROOT/'scripts/event_composition_probe.py', ROOT/'scripts/review_event_chains.py',
        ROOT/'scripts/sampling_probe.py', ROOT/'scripts/mechanism_probe.py',
        ROOT/'configs/event_visibility_v1.json', ROOT/'configs/event_chains_v1.json',
        ROOT/'configs/qvhighlights_v2.yaml', ROOT/'docs/EVAL_EVENT_COMPOSITION.md'])
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    t0 = time.perf_counter()
    emb = MockEmbedding() if mock else OpenCLIPEmbedding(max_side=cfg.video.max_frame_side_px)
    loading_sec = time.perf_counter() - t0
    if not mock:
        actual = [type(model.transformer.resblocks[0].mlp.gelu).__name__ for model in (emb._model.visual, emb._model)]
        if actual != ['QuickGELU', 'QuickGELU']:
            raise RuntimeError('Matched QuickGELU towers required')
    rows, errors = [], []
    for chain in manifest['chains']:
        try:
            s = by_q[chain['qid']]
            asset = asset_from_probe(s.video_id, s.video_path)
            window = validate_chain(chain, asset.duration_sec)
            times_requested = midpoint_times(window, asset.duration_sec, count)
            index = VideoIndex(asset, out, content_hints=hints or [])
            frames, mat, hit, extraction_sec, encoding_sec = extra_features(index, times_requested, emb, cfg, use_mock=mock)
            times = [f.timestamp_sec for f in frames]
            if len(frames) != count or any(not window[0] <= t < window[1] for t in times):
                raise RuntimeError('Model frame budget/time contract violation')
            a, b = chain['event_a'], chain['event_b']
            sentences = [f'{a} Then {b}', f'{b} Then {a}']
            texts = [a, b, *sentences, chain['context_a'], chain['context_b']]
            vectors = emb.encode_texts(texts)
            scores = mat.astype(np.float64) @ vectors.astype(np.float64).T
            controls = sequence_time_controls(scores[:, :2], scores[:, 4:6], times,
                                               minimum, tuple(manifest['shuffle_seeds']))
            row = {'qid': s.qid, 'video_id': s.video_id, 'video_sha256': asset.sha256,
                   'model_id': emb.model_id, 'original_query': s.query, 'chain': chain,
                   'full_sentence_margin': float(np.mean(scores[:, 2] - scores[:, 3])),
                   'phase_diagnosis': phase_diagnosis(scores[:, :2], times, chain),
                   'controls': controls, 'texts': texts, 'per_frame_text_scores': scores.tolist(),
                   'requested_times_sec': times_requested, 'actual_times_sec': times,
                   'new_encoded_frames': 0 if hit else len(frames), 'cache_hit': hit,
                   'extraction_sec': extraction_sec, 'encoding_sec': encoding_sec}
            serialized = json.dumps(row)
            with (out/'records.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(serialized + '\n')
            rows.append(row)
            (out/f'qid_{s.qid}.jpg').write_bytes(build_montage([Path(f.image_path) for f in frames], cols=4))
            print(f"qid={s.qid} full={row['full_sentence_margin']:+.5f} composed={controls['original']['sequence_axis']['margin']:+.5f} cache={hit}", flush=True)
        except Exception as exc:
            errors.append({'qid': chain['qid'], 'error_type': type(exc).__name__, 'message': str(exc)})
            print(f"FAILED qid={chain['qid']} {type(exc).__name__}: {exc}", flush=True)
        report = summarize(rows, manifest, errors, mock)
        report['model_loading_sec'] = loading_sec
        (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (out/'per_query.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['qid', 'full_sentence_margin', 'composed_margin', 'context_margin', 'a_phase_margin', 'b_phase_margin']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            c = r['controls']['original']
            writer.writerow({'qid': r['qid'], 'full_sentence_margin': r['full_sentence_margin'],
                             'composed_margin': c['sequence_axis']['margin'],
                             'context_margin': c['context_sequence_axis']['margin'],
                             'a_phase_margin': r['phase_diagnosis']['a_vs_b_in_a_phase'],
                             'b_phase_margin': r['phase_diagnosis']['b_vs_a_in_b_phase']})
    if not mock and sha256_file(ROOT/'data/qvhighlights_fresh80/selection.json') != protocol['fresh_selection_sha256']:
        raise RuntimeError('Independent selection changed')
    print('OUTPUT=' + str(out), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    cfg = Config.load(ROOT/'configs/qvhighlights_v2.yaml', apply_env=False).resolved()
    manifest = json.loads((ROOT/'configs/event_chains_v1.json').read_text(encoding='utf-8'))
    samples, hints = None, None
    if args.mock:
        from queryseek.eval.runner import dataset_from_labels
        labels = ROOT/'assets/demo/sample.labels.json'
        samples = dataset_from_labels(labels, ROOT/'assets/demo/sample.mp4', None)[:1]
        hints = json.loads(labels.read_text(encoding='utf-8'))['segments']
        manifest['chains'] = [{'qid': samples[0].qid, 'window': [0., 24.],
                               'event_a_window': [0., 12.], 'event_b_window': [12., 24.],
                               'event_a': 'intro welcome', 'event_b': 'architecture',
                               'context_a': 'intro', 'context_b': 'architecture'}]
        manifest['excluded'] = []
    report = run_probe(manifest, cfg, ROOT/'outputs/event_composition'/new_job_id(), args.mock, samples, hints)
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
