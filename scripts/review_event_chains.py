"""Source-PTS storyboards for Agent visual review, without model inference.

Review is sampled at 1 fps; it does not certify unseen motion or negative labels.
The output is restricted to outputs/ and preserves the earlier diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from queryseek import PIPELINE_VERSION
from queryseek.eval.runner import load_dataset
from queryseek.ft.montage import build_montage
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.ingest.frames import extract_local_frames, load_frames_index, save_frames_index
from scripts.sampling_probe import checked_output


def envelope(windows, duration, margin):
    if not windows or not math.isfinite(margin) or margin < 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Annotated windows and nonnegative margin required')
    start = max(0., min(w.start_sec for w in windows) - margin)
    end = min(duration, max(w.end_sec for w in windows) + margin)
    if not 0 <= start < end <= duration or any(w.end_sec > duration for w in windows):
        raise ValueError('Review envelope violates half-open time contract')
    return float(start), float(end)


def cached_review(asset, window, fps, max_side):
    signature = {'video_sha256': asset.sha256, 'pipeline_version': PIPELINE_VERSION,
                 'extractor': 'local-select-source-pts-v3', 'start_sec': window[0],
                 'end_sec': window[1], 'fps': fps, 'max_side': max_side, 'jpeg_quality': 3}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = checked_output(ROOT / 'outputs/event_review_cache' / key)
    meta = directory / 'review_meta.json'
    hit = meta.is_file()
    if hit:
        if json.loads(meta.read_text(encoding='utf-8')) != signature:
            raise ValueError('Review cache signature mismatch')
        frames = load_frames_index(directory / 'frames.jsonl')
    else:
        frames = extract_local_frames(asset.path, directory / 'frames', *window, fps,
                                      max_side=max_side, tag='review')
        save_frames_index(frames, directory / 'frames.jsonl')
        meta.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    if (not frames or any(not window[0] <= f.timestamp_sec < window[1] for f in frames)
            or len({f.timestamp_sec for f in frames}) != len(frames)
            or any(not Path(f.image_path).is_file() for f in frames)):
        raise RuntimeError('Incomplete/invalid review frames')
    return frames, hit


def review(manifest, out):
    out = checked_output(out)
    out.mkdir(parents=True, exist_ok=False)
    samples = {s.qid: s for s in load_dataset(ROOT / manifest['dataset'])}
    source = ROOT / manifest['source_run']
    source_protocol = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    if sha256_file(ROOT / manifest['dataset']) != source_protocol['dataset_sha256']:
        raise ValueError('Development dataset changed since source diagnosis')
    protocol = {**manifest, 'oracle_only': True, 'reviewer': 'Agent, one reviewer, sampled frames',
                'dataset_sha256': source_protocol['dataset_sha256'], 'vlm_calls': 0,
                'fresh_selection_sha256': sha256_file(ROOT / 'data/qvhighlights_fresh80/selection.json')}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out / 'review_event_chains.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    for qid in manifest['qids']:
        s = samples[qid]
        asset = asset_from_probe(s.video_id, s.video_path)
        window = envelope(s.gt_windows, asset.duration_sec, manifest['pre_post_sec'])
        frames, hit = cached_review(asset, window, manifest['fps'], manifest['max_side'])
        folder = checked_output(out / f'qid_{qid}')
        folder.mkdir()
        pages = []
        for offset in range(0, len(frames), 32):
            chunk = frames[offset:offset + 32]
            # Pad the last page with its last observed frame for rectangular montage.
            # Padding is review layout only; records and model budgets contain no duplicates.
            tiles = chunk + [chunk[-1]] * ((-len(chunk)) % 4)
            page = folder / f'page_{offset // 32:02d}.jpg'
            page.write_bytes(build_montage([Path(f.image_path) for f in tiles], cols=4))
            pages.append({'path': str(page), 'actual_times_sec': [f.timestamp_sec for f in chunk],
                          'padding_tiles': len(tiles) - len(chunk)})
        row = {'qid': qid, 'query': s.query, 'video_id': s.video_id, 'video_sha256': asset.sha256,
               'video_path': s.video_path, 'duration_sec': asset.duration_sec,
               'all_gt': [[w.start_sec, w.end_sec] for w in s.gt_windows],
               'review_window': list(window), 'cache_hit': hit, 'pages': pages,
               'frames': [f.model_dump() for f in frames]}
        rows.append(row)
        with (out / 'records.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        print(f'qid={qid} review={window} frames={len(frames)} cache={hit}', flush=True)
    if sha256_file(ROOT / 'data/qvhighlights_fresh80/selection.json') != protocol['fresh_selection_sha256']:
        raise RuntimeError('Frozen independent selection changed')
    print('OUTPUT=' + str(out), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/event_visibility/review_v1')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/event_visibility_v1.json').read_text(encoding='utf-8'))
    review(manifest, args.output)
