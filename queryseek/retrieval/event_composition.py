"""Frozen-feature event composition for offline representation diagnosis.

These small aggregators are ranking glue, not a trained video/motion encoder.
No GT, text parser, feature model or video extraction belongs in this module.
"""
from __future__ import annotations

import numpy as np


def ordered_event_score(scores, min_frames_per_event=2):
    """Best A-before-B shared split, weighting the two events equally."""
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all()
            or isinstance(min_frames_per_event, bool)
            or not isinstance(min_frames_per_event, int) or min_frames_per_event < 1
            or len(scores) < 2 * min_frames_per_event):
        raise ValueError('Finite [frames, two events] scores and adequate event support required')
    cumulative = np.vstack([np.zeros((1, 2)), np.cumsum(scores, axis=0)])
    splits = np.arange(min_frames_per_event, len(scores) - min_frames_per_event + 1)
    values = (cumulative[splits, 0] / splits
              + (cumulative[-1, 1] - cumulative[splits, 1]) / (len(scores) - splits)) / 2
    best = int(np.argmax(values))
    return {'score': float(values[best]), 'split_position': int(splits[best])}


def ordered_event_margin(scores, min_frames_per_event=2):
    scores = np.asarray(scores, dtype=np.float64)
    forward = ordered_event_score(scores, min_frames_per_event)
    backward = ordered_event_score(scores[:, ::-1], min_frames_per_event)
    return {'margin': forward['score'] - backward['score'], 'forward': forward, 'backward': backward}


def sequence_time_controls(scores, context_scores, timestamps, min_frames_per_event=2,
                           shuffle_seeds=(12, 23, 42, 71, 101)):
    """Separate display-order aggregation from sorting by immutable source PTS.

    Shuffle does not have a binary A/B ordering label; source sorting reconstructs
    original chronology. Neither reconstruction nor reverse antisymmetry measures
    visual motion understanding.
    """
    scores = np.asarray(scores, dtype=np.float64)
    context = np.asarray(context_scores, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if (context.shape != scores.shape or len(times) != len(scores) or not np.isfinite(times).all()
            or not np.isfinite(context).all() or np.any(times < 0)
            or len(np.unique(times)) != len(times) or np.any(np.diff(times) <= 0)):
        raise ValueError('Aligned finite chronological source observations required')
    permutations = {'original': np.arange(len(scores)), 'reverse': np.arange(len(scores))[::-1]}
    permutations.update({f'shuffle_{seed}': np.random.default_rng(seed).permutation(len(scores))
                         for seed in shuffle_seeds})
    out = {}
    for name, permutation in permutations.items():
        source_sort = np.argsort(times[permutation], kind='stable')
        displayed = scores[permutation]
        out[name] = {'source_frame_positions': permutation.tolist(),
                     'displayed_source_times_sec': times[permutation].tolist(),
                     'sequence_axis': ordered_event_margin(displayed, min_frames_per_event),
                     'source_pts_axis': ordered_event_margin(displayed[source_sort], min_frames_per_event),
                     'context_sequence_axis': ordered_event_margin(context[permutation], min_frames_per_event),
                     'unordered_score': float(np.mean(displayed)),
                     'unordered_order_margin': 0.,
                     'source_pts_consumed': True,
                     'source_time_label': 'original A before B',
                     'display_order_label': 'A before B' if name == 'original' else
                                            'B before A' if name == 'reverse' else 'no binary label'}
    return out
