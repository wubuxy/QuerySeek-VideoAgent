"""Experimental common-reference span scores and preservation of E4 proposals.

Only scalar observed CLIP scores enter this module; labels are not accepted.
Scores are ranking signals, never calibrated probabilities or event confirmation.
"""

from __future__ import annotations

import math

import numpy as np

from .curve_spans import select_curve_windows
from ..schemas import TimeWindow


def _cells(times, scores, duration_sec):
    t,s=np.asarray(times,dtype=float),np.asarray(scores,dtype=float)
    if (not len(t) or len(t)!=len(s) or not math.isfinite(duration_sec) or duration_sec<=0
            or not np.isfinite(t).all() or not np.isfinite(s).all()
            or np.any(np.diff(t)<=0) or np.any(t<0) or np.any(t>=duration_sec)):
        raise ValueError("Invalid observed curve")
    edges=np.concatenate(([0.],(t[:-1]+t[1:])/2,[duration_sec]))
    return t,s,edges


def time_weighted_mean(times, scores, duration_sec, ranges):
    """Integrate nearest-observation cells; this is an estimate between observations."""
    t,s,edges=_cells(times,scores,duration_sec)
    weights=np.zeros(len(t))
    for a,b in ranges:
        if not math.isfinite(a) or not math.isfinite(b) or not 0<=a<b<=duration_sec:
            raise ValueError("Invalid integration window")
        weights+=np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a))
    return float(weights@s/weights.sum()) if weights.sum()>0 else None


def calibrated_windows(
    original: list[TimeWindow], supplemental: list[TimeWindow],
    coarse_times, coarse_scores, observed_times, observed_scores, duration_sec,
    mode="shared", locked_top1: TimeWindow | None=None,
):
    """Retain all original geometries; compare legacy/shared/local-contrast ranking.

    Shared reference: median/std of coarse scores, never the selectively sampled
    distribution. Support = time-weighted positive z * sqrt(seconds).
    Contrast adds 2 * positive(inside_mean - adjacent_mean) / coarse_std, with
    8-second adjacent context; absent observed context falls back to coarse median.
    Locking top1 is an explicit product guard, not evidence of a better top1 policy.
    """
    if mode not in ("legacy","shared","contrast"):
        raise ValueError("Unknown scoring mode")
    _,base,_=_cells(coarse_times,coarse_scores,duration_sec)
    t,s,_=_cells(observed_times,observed_scores,duration_sec)
    center=float(np.median(base));scale=max(float(np.std(base)),1e-6)
    positive=np.maximum((s-center)/scale,0)
    features=[];ranked=[]
    for origin,windows in (("original",original),("supplemental",supplemental)):
        for window in windows:
            a,b=window.start_sec,window.end_sec
            if not 0<=a<b<=duration_sec or window.score is None or not math.isfinite(window.score):
                raise ValueError("Invalid candidate window/score")
            inside_count=int(np.sum((t>=a)&(t<b)))
            inside=time_weighted_mean(t,s,duration_sec,[(a,b)])
            support=time_weighted_mean(t,positive,duration_sec,[(a,b)])*math.sqrt(b-a)
            ranges=[]
            if a>0:ranges.append((max(0.,a-8),a))
            if b<duration_sec:ranges.append((b,min(duration_sec,b+8)))
            context_count=sum(int(np.sum((t>=x)&(t<y))) for x,y in ranges)
            outside=time_weighted_mean(t,s,duration_sec,ranges) if context_count else center
            contrast=max(0.,(inside-outside)/scale)
            score=window.score if mode=="legacy" else support+(2*contrast if mode=="contrast" else 0)
            # Do not promote a window solely using inferred values between observed frames.
            if not inside_count and mode!="legacy":score=0.
            ranked.append(window.model_copy(update={"score":float(score)}))
            features.append({"origin":origin,"start_sec":a,"end_sec":b,
                             "legacy_score":window.score,"inside_observations":inside_count,
                             "context_observations":context_count,"inside_mean":inside,
                             "context_mean":outside,"support":support,"contrast":contrast,
                             "ranking_score":float(score)})
    if locked_top1 is not None:
        if not any(w.start_sec==locked_top1.start_sec and w.end_sec==locked_top1.end_sec for w in original):
            raise ValueError("Locked anchor must be an original candidate")
        maximum=max((w.score for w in ranked),default=0.)
        for window in ranked:
            if window.start_sec==locked_top1.start_sec and window.end_sec==locked_top1.end_sec:
                window.score=float(maximum+1)
        # The numeric offset explicitly encodes a policy priority, not semantic confidence.
        for feature in features:
            if feature["start_sec"]==locked_top1.start_sec and feature["end_sec"]==locked_top1.end_sec:
                feature["ranking_score"]=float(maximum+1);feature["locked_anchor"]=True
    return ranked,select_curve_windows(ranked,top_k=5,nms_iou=.9),features
