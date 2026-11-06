"""Research time-window glue, preserving E4 and frozen CLIP semantics.

The duration grid is a coverage diagnostic inspired by multiscale localization;
it is neither an implementation nor a reproduction of ActionFormer.
"""
from __future__ import annotations
import numpy as np
from queryseek.retrieval.candidate_ranker import candidate_table,candidate_features


def multiscale_table(times,scores,duration_sec,lengths_sec,stride_fraction=.25,min_stride_sec=1.):
    lengths=np.asarray(lengths_sec,dtype=float)
    if (lengths.ndim!=1 or not len(lengths) or not np.isfinite(lengths).all() or np.any(lengths<=0)
            or not np.isfinite([stride_fraction,min_stride_sec]).all() or not 0<stride_fraction<=1 or min_stride_sec<=0):
        raise ValueError('Invalid multiscale configuration')
    original,_=candidate_table(times,scores,duration_sec)
    # Validate even an empty original pool with the shared strict timeline gate.
    candidate_features(times,scores,duration_sec,[])
    existing={(float(a),float(b)) for a,b,_ in original};extra=[]
    for length in sorted(set(float(min(v,duration_sec)) for v in lengths)|{float(duration_sec)}):
        stride=max(min_stride_sec,length*stride_fraction)
        for start in list(np.arange(0.,max(duration_sec-length,0.),stride))+[max(duration_sec-length,0.)]:
            a=float(start);b=float(min(a+length,duration_sec))
            if (a,b) not in existing:
                extra.append((a,b,0.));existing.add((a,b))
    x=candidate_features(times,scores,duration_sec,extra)
    # Same feature definition for every new scale; no GT or learned boundary.
    for i,(a,b,_) in enumerate(extra):extra[i]=(a,b,float(max(x[i,8],0.)*np.sqrt(b-a)))
    windows=np.concatenate([original,np.asarray(extra,dtype=float).reshape(-1,3)])
    return windows,candidate_features(times,scores,duration_sec,windows)
