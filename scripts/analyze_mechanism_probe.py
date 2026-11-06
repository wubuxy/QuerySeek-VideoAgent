"""Verify and summarize a completed diagnostic without any model inference."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mechanism_probe import summarize
from scripts.sampling_probe import checked_output, feature_signature
from queryseek.ft.montage import build_montage
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.frames import load_frames_index


def analyze(run, review_qids=()):
    import hashlib
    run = checked_output(run)
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (run / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    rebuilt = summarize(records, protocol['manifest']['samples'], metrics['errors'], metrics['mock'])
    if any(metrics[key] != value for key, value in rebuilt.items()):
        raise ValueError('Metrics do not match recorded evidence')
    if not protocol['mock']:
        source = ROOT / protocol['manifest']['source_run']
        if any(sha256_file(source / name) != value for name, value in protocol['baseline_hashes'].items()):
            raise ValueError('Baseline source artifacts changed')
        if sha256_file(ROOT / 'data/qvhighlights_fresh80/selection.json') != protocol['fresh_selection_sha256']:
            raise ValueError('Independent selection changed')
    out = checked_output(run / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    fields = ['qid', 'stratum', 'failure_stage', 'pool_iou', 'top5_iou', 'top1_iou',
              'pair_eligible', 'full_margin', 'object_margin', 'incremental_margin',
              'coarse_margin', 'text_control_margin', 'query']
    with (out / 'per_query.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            pair = row['pair'] or {}
            writer.writerow({'qid': row['qid'], 'stratum': row['stratum'], 'query': row['query'],
                             **row['structure'], 'pair_eligible': row['pair'] is not None,
                             **{key: pair.get(key) for key in fields[7:11]},
                             'text_control_margin': row['text_control']['margin'] if row['text_control'] else None})
    lines = ['# Offline mechanism diagnosis', '', 'Development/oracle diagnosis; no accuracy gain claim.', '',
             '| Stratum | n | pool hits | top5 hits | top1 hits | pairs | GT beats background | mean cosine margin |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, block in metrics['strata'].items():
        margin = block.get('full_margin', {}).get('delta')
        lines.append(f"| {name} | {block['n_intended']} | {block['pool_hits']} | {block['top5_hits']} | {block['top1_hits']} | {block['pair_eligible']} | {block.get('gt_beats_background', 0)} | {margin if margin is not None else 'N/A'} |")
    lines += ['', 'Counts use IoU >= 0.5. Unannotated backgrounds are not verified negatives.',
              'Longest GT only; other annotated intervals are not pooled into the oracle clip.', '',
              '## Evidence sheets', '', 'Tiles 1–8: GT crop in chronological order; 9–16: background, when eligible.',
              'Only sampled frames are shown; these sheets do not certify an entire action or its absence.']
    for qid in review_qids:
        row = next(r for r in records if r['qid'] == qid)
        index = SimpleNamespace(asset=SimpleNamespace(sha256=row['video_sha256']))
        signature = feature_signature(index, row['model_id'], row['requested_times_sec'],
                                      protocol['config']['video']['max_frame_side_px'])
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        directory = checked_output(ROOT / 'outputs/sampling_feature_cache' / key)
        if json.loads((directory / 'sampling_meta.json').read_text(encoding='utf-8')) != signature:
            raise ValueError('Review cache signature mismatch')
        frames = load_frames_index(directory / 'frames.jsonl')
        if [f.timestamp_sec for f in frames] != row['actual_times_sec']:
            raise ValueError('Review source PTS differ from experiment')
        sheet = out / f'qid_{qid}.jpg'
        sheet.write_bytes(build_montage([Path(f.image_path) for f in frames], cols=4))
        lines += ['', f'### qid={qid}', '', row['query'], '', f'![sampled evidence](qid_{qid}.jpg)', '',
                  'Actual PTS (seconds): ' + ', '.join(f'{f.timestamp_sec:.3f}' for f in frames)]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil_source = out / 'analyze_mechanism_probe.py'
    shutil_source.write_bytes(Path(__file__).read_bytes())
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--review-qids', type=int, nargs='*', default=[])
    args = parser.parse_args()
    print(analyze(args.run, args.review_qids))
