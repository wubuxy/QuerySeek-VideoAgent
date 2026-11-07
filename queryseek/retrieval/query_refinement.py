"""Query-guided observation allocation; CLIP scores are retrieval cues, not proof."""
from __future__ import annotations

import math
import numpy as np

from queryseek.agent.temporal_tree import NodeObservation, SplitConfig, uniform_partition, visual_change_split
from queryseek.eval.grounded_qa import checked_intervals
from queryseek.retrieval.curve_spans import curve_spans, select_curve_windows


METHODS = ('dense_uniform', 'uniform_refine', 'scene_refine')


def relevance(observation, query_vector):
    features, query = np.asarray(observation.embeddings), np.asarray(query_vector)
    if (features.ndim != 2 or features.shape[0] != len(observation.frames)
            or query.shape != (features.shape[1],) or not np.isfinite(features).all()
            or not np.isfinite(query).all() or np.linalg.norm(query) <= 0
            or np.any(np.linalg.norm(features, axis=1) <= 0)):
        raise ValueError('Invalid query/image features')
    return (features/np.linalg.norm(features, axis=1, keepdims=True)) @ (query/np.linalg.norm(query))


def rank_regions(times, scores, spans, duration, pool=3):
    checked_intervals(spans, duration)
    if (type(pool) is not int or pool < 1 or len(times) != len(scores)
            or not np.isfinite(scores).all() or any(not 0 <= t < duration for t in times)):
        raise ValueError('Invalid region ranking input')
    rows=[]
    for i,(a,b) in enumerate(spans):
        values=sorted((float(s) for t,s in zip(times,scores) if a <= t < b),reverse=True)
        rows.append({'region_index':i,'window_sec':[float(a),float(b)],
                     'score':float(np.mean(values[:pool])) if values else None,'coarse_frames':len(values)})
    return sorted(rows,key=lambda r:(r['score'] is None,-(r['score'] or 0.),r['region_index']))


def padded_window(span, duration, padding, min_width=64.):
    (a,b),=checked_intervals([span],duration)
    if not all(math.isfinite(v) and v >= 0 for v in [padding,min_width]):
        raise ValueError('Invalid context width')
    left,right=max(0.,a-padding),min(duration,b+padding)
    width=min(duration,max(right-left,min_width))
    center=(left+right)/2
    left=min(max(0.,center-width/2),duration-width)
    return float(left),float(left+width)


def merge_observations(observations):
    """Deduplicate source-time identities at the microsecond transport precision."""
    by_time={}
    for obs in observations:
        if len(obs.frames) != len(obs.embeddings):
            raise ValueError('Frame/feature mismatch')
        for frame,vector in zip(obs.frames,obs.embeddings):
            by_time.setdefault(round(frame.timestamp_sec*1_000_000),(frame,vector))
    ordered=[by_time[t] for t in sorted(by_time)]
    if not ordered:
        raise ValueError('No observed frames')
    return NodeObservation([r[0] for r in ordered],np.asarray([r[1] for r in ordered]))


def select_review_frames(observation, scores, windows, count=16):
    if type(count) is not int or count < 2 or len(observation.frames) < count:
        raise ValueError('Insufficient review observations')
    chosen=[]
    leading=windows[:2]
    for window in leading:
        indices=[i for i,f in enumerate(observation.frames)
                 if window.start_sec <= f.timestamp_sec < window.end_sec and i not in chosen]
        take=min(len(indices),count//max(1,len(leading)))
        chosen.extend(indices[int((j+.5)*len(indices)/take)] for j in range(take))
    fallback=count-len(chosen)
    ranked=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),observation.frames[i].timestamp_sec))
    for i in ranked:
        if len(chosen)==count:break
        if i not in chosen:chosen.append(i)
    return [observation.frames[i] for i in sorted(chosen)],fallback


def retrieve(method, duration, query_vector, observe, config, root=None):
    """observe(start,end,budget) never receives questions, choices or GT."""
    if method not in METHODS:
        raise ValueError('Unknown retrieval method')
    regions=[]
    if method=='dense_uniform':
        observations=[observe(0.,duration,config['dense_frames'])]
    else:
        root=root or observe(0.,duration,config['root_frames'])
        scene_spans,_=visual_change_split(0.,duration,root,SplitConfig(**config['split']))
        spans=scene_spans if method=='scene_refine' else uniform_partition(0.,duration,len(scene_spans))
        ranked=rank_regions([f.timestamp_sec for f in root.frames],relevance(root,query_vector),
                            spans,duration,config['region_pool'])
        observations=[root]
        for region in ranked[:config['selected_regions']]:
            if region['score'] is None:
                raise ValueError('Selected region has no coarse observations')
            a,b=padded_window(region['window_sec'],duration,duration/config['root_frames'],
                              float(config['local_frames']))
            regions.append({**region,'observation_window_sec':[a,b]})
            observations.append(observe(a,b,config['local_frames']))
    merged=merge_observations(observations)
    scores=relevance(merged,query_vector)
    windows=select_curve_windows(curve_spans([f.timestamp_sec for f in merged.frames],scores,duration),
                                  top_k=config['top_k'],nms_iou=config['nms_iou'])
    review,fallback=select_review_frames(merged,scores,windows,config['review_frames'])
    record={'method':method,'regions':regions,'observed_source_times_sec':[f.timestamp_sec for f in merged.frames],
            'scores':scores.tolist(),'windows':[w.model_dump() for w in windows],
            'review_source_times_sec':[f.timestamp_sec for f in review],
            'review_fallback_frames':fallback,'charged_image_observations':sum(len(o.frames) for o in observations),
            'unique_observations':len(merged.frames),'observation_costs':[o.cost for o in observations],
            'evidence_status':'unverified'}
    return record,review
