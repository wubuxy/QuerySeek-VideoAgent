"""Selected boundary helpers from CeeZh/VTS, MIT license.

Copyright (c) 2026 CeeZh. See LICENSE in this directory.
Source: infer/utils/segmentation.py at 184310c5eec2193060a5c95bbda4ad443dae4e25.
Four function bodies copied unchanged; decoding, CLIP service and unrelated
segmentation modes omitted. Source SHA256: d8a2bd7c4f54be62a2b2fb079bd87fabdb394dea90a6441477209e017c1c2dac.
"""
from typing import List, Tuple
import numpy as np

def _compute_frame_deltas(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine distance between each consecutive pair of frame embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    cosine_sim = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return 1.0 - cosine_sim

def _detect_boundaries_topk(deltas: np.ndarray, num_boundaries: int) -> List[int]:
    """Return exactly num_boundaries boundary indices (top-k largest deltas)."""
    num_boundaries = min(num_boundaries, len(deltas))
    if num_boundaries <= 0:
        return []
    indices = np.argpartition(deltas, -num_boundaries)[-num_boundaries:]
    return sorted(indices.tolist())

def _detect_boundaries_threshold(
    deltas: np.ndarray,
    k: float = 1.5,
    min_segments: int = 3,
    max_segments: int = 8,
) -> List[int]:
    """Detect scene boundaries where delta > mean + k * std (z-score)."""
    threshold = deltas.mean() + k * deltas.std()
    boundary_indices = np.where(deltas > threshold)[0].tolist()

    num_segments = len(boundary_indices) + 1
    if num_segments > max_segments:
        boundary_indices = _detect_boundaries_topk(deltas, max_segments - 1)
    elif num_segments < min_segments:
        boundary_indices = _detect_boundaries_topk(deltas, min_segments - 1)

    return boundary_indices

def _boundaries_to_segments(
    timestamps: List[float],
    boundary_indices: List[int],
    start_sec: float,
    end_sec: float,
) -> List[Tuple[float, float]]:
    """Convert boundary frame indices to (start_sec, end_sec) time segments."""
    boundary_times = [
        (timestamps[i] + timestamps[i + 1]) / 2.0 for i in boundary_indices
    ]
    segment_starts = [start_sec] + boundary_times
    segment_ends = boundary_times + [end_sec]
    return list(zip(segment_starts, segment_ends))
