"""Versioned source-PTS and CLIP caches for temporal-tree observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from queryseek import PIPELINE_VERSION
from queryseek.agent.temporal_tree import NodeObservation
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.frames import extract_scene_frames, load_frames_index, save_frames_index
from queryseek.schemas import Scene
from queryseek.tools.paths import resolved_path_is_within

ROOT = Path(__file__).resolve().parents[2]


def output_path(path):
    path = Path(path).resolve()
    if not resolved_path_is_within(path, (ROOT/'outputs').resolve()):
        raise ValueError('Output/cache must remain under outputs/')
    return path


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


@dataclass(frozen=True)
class SamplingConfig:
    fps_cap: float = 1.0
    max_frames: int = 64
    max_side: int = 512
    batch_size: int = 16

    def __post_init__(self):
        if (not math.isfinite(self.fps_cap) or self.fps_cap <= 0
                or any(type(x) is not int for x in [self.max_frames, self.max_side, self.batch_size])
                or self.max_frames < 2 or self.max_side < 16 or self.batch_size < 1):
            raise ValueError('Invalid tree sampling configuration')


class WindowObservationStore:
    def __init__(self, video_path, duration, video_sha256, encoder, encoder_signature, config=None):
        self.video_path, self.duration = str(video_path), float(duration)
        if (not math.isfinite(self.duration) or self.duration <= 0 or len(video_sha256) != 64
                or any(c not in '0123456789abcdef' for c in video_sha256)):
            raise ValueError('Invalid probed media identity')
        if not encoder_signature or not isinstance(encoder_signature, dict):
            raise ValueError('Encoder identity/configuration required')
        self.video_sha256, self.encoder = video_sha256, encoder
        self.encoder_signature = json.loads(json.dumps(encoder_signature))
        self.config = config or SamplingConfig()

    def __call__(self, start, end):
        began = time.perf_counter()
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= self.duration:
            raise ValueError('Invalid observation interval')
        cfg = self.config
        count = min(cfg.max_frames, max(2, math.ceil((end-start)*cfg.fps_cap)))
        times = [float(start+(i+.5)*(end-start)/count) for i in range(count)]
        legacy_signature = {'video_sha256': self.video_sha256, 'pipeline_version': PIPELINE_VERSION,
            'extractor': 'extract_scene_frames-source-pts-v3', 'budget': count,
            'max_side': cfg.max_side, 'requested_times_sec': times, 'jpeg_quality': 3}
        signature = {**legacy_signature, 'interval_sec': [float(start), float(end)],
                     'sampling': asdict(cfg), 'cache_version': 'temporal-tree-frames-v1'}
        cache = output_path(ROOT/'outputs/temporal_tree_frame_cache'/json_hash(signature))
        manifest_path = cache/'manifest.json'
        cache_hit, reused_legacy, extracted = manifest_path.exists(), False, 0
        frame_started = time.perf_counter()
        if cache_hit:
            manifest = json.loads(manifest_path.read_text('utf-8'))
            if manifest['signature'] != signature or sha256_file(cache/'frames.jsonl') != manifest['index_sha256']:
                raise ValueError('Frame cache manifest/index mismatch')
            frames = load_frames_index(cache/'frames.jsonl')
            hashes = [sha256_file(output_path(f.image_path)) for f in frames]
            if hashes != manifest['image_sha256']:
                raise ValueError('Frame cache image mismatch')
        else:
            cache.mkdir(parents=True, exist_ok=True)
            old = output_path(ROOT/'outputs/grounded_qa_frame_cache'/json_hash(legacy_signature))
            if (old/'manifest.json').is_file():
                legacy = json.loads((old/'manifest.json').read_text('utf-8'))
                frames = load_frames_index(old/'frames.jsonl')
                hashes = [sha256_file(output_path(f.image_path)) for f in frames]
                if legacy['signature'] != legacy_signature or hashes != legacy['image_sha256']:
                    raise ValueError('Existing baseline frame cache mismatch')
                reused_legacy = True
            else:
                # Structured intervals only. FFmpeg records real source PTS.
                epsilon = min(.0001, (end-start)/(4*count))
                scenes = [Scene(scene_id=f'tree_{i:04d}', start_sec=t-epsilon, end_sec=t+epsilon)
                          for i, t in enumerate(times)]
                frames = extract_scene_frames(self.video_path, cache, scenes, max_side=cfg.max_side)
                hashes = [sha256_file(f.image_path) for f in frames]
                extracted = len(frames)
            for frame, digest in zip(frames, hashes):
                frame.frame_id = f'pts_{round(frame.timestamp_sec*1e6)}_{digest[:12]}'
        if (len(frames) != count or any(not start <= f.timestamp_sec < end for f in frames)
                or any(a.timestamp_sec >= b.timestamp_sec for a, b in zip(frames, frames[1:]))):
            raise ValueError('Incomplete/nonincreasing/out-of-window source PTS')
        if not cache_hit:
            save_frames_index(frames, cache/'frames.jsonl')
            write_json(manifest_path, {'signature': signature, 'image_sha256': hashes,
                                      'index_sha256': sha256_file(cache/'frames.jsonl')})
        frame_sec = time.perf_counter()-frame_started
        feature_signature = {'frames_signature': signature, 'image_sha256': hashes,
            'source_times_sec': [f.timestamp_sec for f in frames], 'encoder': self.encoder_signature,
            'cache_version': 'temporal-tree-embeddings-v1'}
        features_dir = output_path(ROOT/'outputs/temporal_tree_embedding_cache'/json_hash(feature_signature))
        feature_manifest = features_dir/'manifest.json'
        feature_hit = feature_manifest.exists()
        feature_started = time.perf_counter()
        if feature_hit:
            fm = json.loads(feature_manifest.read_text('utf-8'))
            if fm['signature'] != feature_signature or sha256_file(features_dir/'features.npy') != fm['features_sha256']:
                raise ValueError('Feature cache mismatch')
            vectors = np.load(features_dir/'features.npy', allow_pickle=False)
        else:
            vectors = np.asarray(self.encoder.encode_images([f.image_path for f in frames], batch_size=cfg.batch_size), dtype=np.float32)
        if (vectors.ndim != 2 or vectors.shape[0] != count or not np.isfinite(vectors).all()
                or np.any(np.linalg.norm(vectors, axis=1) <= 0)):
            raise ValueError('Invalid encoder output')
        if not feature_hit:
            features_dir.mkdir(parents=True, exist_ok=True)
            np.save(features_dir/'features.npy', vectors, allow_pickle=False)
            write_json(feature_manifest, {'signature': feature_signature, 'features_sha256': sha256_file(features_dir/'features.npy')})
        return NodeObservation(frames, vectors, {
            'new_frames_extracted': extracted, 'frame_cache_hit': cache_hit,
            'reused_baseline_frames': reused_legacy, 'feature_cache_hit': feature_hit,
            'encoded_images': 0 if feature_hit else count, 'frames': count,
            'frame_seconds': frame_sec, 'feature_seconds': time.perf_counter()-feature_started,
            'elapsed_sec': time.perf_counter()-began, 'frame_manifest_sha256': sha256_file(manifest_path),
            'feature_manifest_sha256': sha256_file(feature_manifest)})


class MockBoundaryEncoder:
    """Deterministic image-content hashes, for plumbing only; no semantic claims."""
    def encode_images(self, paths, batch_size=16):
        from queryseek.embeddings.base import MockEmbedding
        return MockEmbedding().encode_texts([sha256_file(p) for p in paths])
