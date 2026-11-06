"""Prepare a frozen development pilot; labels stay in evaluator sidecars.

No model calls, downloads, frame extraction or accuracy claims. Coverage is only
annotation geometry at requested timestamps, not verified visual visibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from queryseek.eval.grounded_qa import load_samples, merged_intervals, uniform_times
from queryseek.tools.paths import resolved_path_is_within


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path, output, video_root=None):
    config = json.loads(config_path.read_text('utf-8'))
    annotation = ROOT / config['annotation']
    selection_path = ROOT / config['selection']
    selection = json.loads(selection_path.read_text('utf-8'))
    actual_sha = digest(annotation)
    if actual_sha != config['annotation_sha256'] or actual_sha != selection['annotation_sha256']:
        raise ValueError('Frozen annotation hash mismatch')
    all_samples = load_samples(json.loads(annotation.read_text('utf-8')))
    by_id = {s.qid: s for s in all_samples}
    chosen = []
    for item in selection['samples']:
        sample = by_id[str(item['qid'])]
        if sample.video_id != item['video_id'] or sample.duration_sec != item['duration']:
            raise ValueError('Frozen selection identity/duration mismatch')
        chosen.append(sample)
    if len({s.video_id for s in chosen}) != len(chosen):
        raise ValueError('Pilot must contain one question per source')
    media_root = Path(video_root) if video_root else ROOT / config['video_root']
    output = output.resolve()
    if not resolved_path_is_within(output, (ROOT / 'outputs').resolve()):
        raise ValueError('Outputs must stay under outputs/')
    if output.exists():
        raise FileExistsError('Choose a new output directory; frozen results are immutable')
    records = []
    requests = []
    labels = []
    for sample in chosen:
        media = media_root / (sample.video_id + '.mp4')
        requests.append({**sample.model_input(), 'video_path': str(media.resolve())})
        labels.append({'qid': sample.qid, 'right_answer': sample.right_answer,
                       'gt_intervals_sec': sample.gt_intervals})
        windows = merged_intervals(sample.gt_intervals)
        for budget in config['frame_budgets']:
            times = uniform_times(sample.duration_sec, budget)
            hits = [sum(start <= t < end for t in times) for start, end in windows]
            records.append({'qid': sample.qid, 'video_id': sample.video_id, 'budget': budget,
                            'requested_times_sec': times, 'clue_count': len(windows),
                            'requested_frames_in_clues': sum(hits), 'any_clue_hit': any(hits),
                            'every_clue_hit': all(hits), 'media_present': media.is_file()})
    duration = [s.duration_sec for s in chosen]
    coverage = {}
    for budget in config['frame_budgets']:
        group = [r for r in records if r['budget'] == budget]
        coverage[str(budget)] = {key: sum(r[key] for r in group) for key in
                                ['any_clue_hit', 'every_clue_hit', 'requested_frames_in_clues']}
    summary = {'scope': 'Frozen development pilot metadata and requested-time geometry only',
               'dataset_rows_validated': len(all_samples), 'pilot_sources': len(chosen),
               'duration_sec': {'min': min(duration), 'median': statistics.median(duration),
                                'max': max(duration), 'sum': sum(duration)},
               'media_present': sum(Path(r['video_path']).is_file() for r in requests),
               'coverage_counts': coverage, 'vlm_calls': 0, 'new_image_encodings': 0,
               'qa_accuracy': None, 'model_inference_executed': False,
               'annotation_sha256': actual_sha, 'selection_sha256': digest(selection_path),
               'config_sha256': digest(config_path), 'source_sha256': {
                   'scripts/prepare_grounded_qa.py': digest(Path(__file__)),
                   'queryseek/eval/grounded_qa.py': digest(ROOT / 'queryseek/eval/grounded_qa.py')}}
    output.mkdir(parents=True)
    for name, data in [('requests.json', requests), ('labels.evaluator_only.json', labels),
                       ('sampling_geometry.json', records), ('preflight.json', summary), ('protocol.json', config)]:
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/grounded_qa_r1_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.output, args.video_root), indent=2))


if __name__ == '__main__':
    main()
