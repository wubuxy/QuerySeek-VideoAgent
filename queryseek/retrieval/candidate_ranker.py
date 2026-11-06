"""Research-only candidate quality ranking; reuse E4 geometry and sklearn.

Inference receives only a timeline, query cosine curve and duration. Labels are
accepted exclusively by training_targets. Scores are ranking signals, not an
event-existence probability. No default CLI behavior is changed.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from queryseek.eval.qvhighlights import IOU_THDS, temporal_iou
from queryseek.retrieval.curve_spans import curve_spans, select_curve_windows
from queryseek.schemas import TimeWindow

FEATURE_NAMES = (
    'log_video_seconds', 'log_span_seconds', 'span_fraction', 'start_fraction', 'end_fraction',
    'inside_mean_cos', 'inside_peak_cos', 'inside_std_cos', 'inside_mean_z', 'inside_peak_z',
    'inside_vs_global_z', 'inside_vs_local_background_z', 'late_minus_early_z', 'log_e4_score',
)
STRUCTURE_COLUMNS = 5


def candidate_features(times, scores, duration_sec, windows):
    """Time-support-weighted features on fixed windows, with no labels."""
    t=np.asarray(times,dtype=np.float64);s=np.asarray(scores,dtype=np.float64)
    if t.ndim!=1 or s.shape!=t.shape or not len(t) or not np.isfinite(t).all() or not np.isfinite(s).all():
        raise ValueError('Invalid finite score timeline')
    if not np.isfinite(duration_sec) or duration_sec<=0 or t[0]<0 or t[-1]>=duration_sec or np.any(np.diff(t)<=0):
        raise ValueError('Invalid source-second timeline')
    edges=np.r_[0.,(t[:-1]+t[1:])/2,duration_sec]
    center=float(np.median(s));scale=max(float(np.std(s)),1e-6)
    def weights(a,b):return np.maximum(0.,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a))
    def mean(w):return float(np.average(s,weights=w))
    global_mean=mean(np.diff(edges));features=[]
    for a,b,e4_score in windows:
        if not (np.isfinite([a,b,e4_score]).all() and 0<=a<b<=duration_sec):
            raise ValueError('Invalid float-second candidate')
        w=weights(a,b);mask=w>0
        inside=mean(w);peak=float(s[mask].max())
        std=float(np.sqrt(np.average((s-inside)**2,weights=w)))
        background=weights(max(0.,a-8.),a)+weights(b,min(duration_sec,b+8.))
        bg=mean(background) if background.sum()>0 else global_mean
        mid=(a+b)/2;early=mean(weights(a,mid));late=mean(weights(mid,b))
        features.append([np.log1p(duration_sec),np.log1p(b-a),(b-a)/duration_sec,a/duration_sec,b/duration_sec,
                         inside,peak,std,(inside-center)/scale,(peak-center)/scale,
                         (inside-global_mean)/scale,(inside-bg)/scale,(late-early)/scale,np.log1p(max(e4_score,0.))])
    return np.asarray(features,dtype=np.float64).reshape(-1,len(FEATURE_NAMES))


def candidate_table(times, scores, duration_sec):
    windows=[(w.start_sec,w.end_sec,w.score) for w in curve_spans(times,scores,duration_sec)]
    features=candidate_features(times,scores,duration_sec,windows)
    return np.asarray(windows,dtype=np.float64).reshape(-1,3),features


def training_targets(windows, relevant_windows, duration_sec):
    """Mean .50:.95 threshold utility from best GT IoU; benchmark supervision.

    This is a candidate quality surrogate, not exact listwise AP or certified
    semantic absence. Multiple candidates can target the same labeled moment.
    """
    if not relevant_windows or any(not np.isfinite([a,b]).all() or not 0<=a<b<=duration_sec for a,b in relevant_windows):
        raise ValueError('Invalid training annotations')
    result=[]
    for a,b,*_ in windows:
        if not np.isfinite([a,b]).all() or not 0<=a<b<=duration_sec: raise ValueError('Invalid training candidate')
        overlap=max(temporal_iou(a,b,c,d) for c,d in relevant_windows)
        result.append(float(np.mean(overlap>=np.asarray(IOU_THDS))))
    return np.asarray(result,dtype=np.float64)


def source_equal_weights(sources, candidate_counts):
    if len(sources)!=len(candidate_counts) or not sources or any(n<=0 for n in candidate_counts):
        raise ValueError('Nonempty candidate groups required')
    counts=Counter(sources)
    w=np.concatenate([np.full(n,1/(counts[source]*n)) for source,n in zip(sources,candidate_counts)])
    return w/w.mean()


def model_inputs(features, control):
    features=np.asarray(features,dtype=np.float64)
    if features.ndim!=2 or features.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError('Invalid candidate feature matrix')
    if control=='full':return features
    if control=='structure':return features[:,:STRUCTURE_COLUMNS]
    raise ValueError('Unknown ranking control')


def rank_candidates(windows, scores=None, top_k=5, nms_iou=.9):
    """Change scores only; stable E4 order breaks ties, original geometry stays."""
    windows=np.asarray(windows,dtype=np.float64).reshape(-1,3)
    values=windows[:,2] if scores is None else np.asarray(scores,dtype=np.float64)
    if values.shape!=(len(windows),) or not np.isfinite(values).all(): raise ValueError('Invalid ranking scores')
    result=select_curve_windows([TimeWindow(start_sec=float(a),end_sec=float(b),score=float(score))
                                for (a,b,_),score in zip(windows,values)],top_k,nms_iou)
    return [(w.start_sec,w.end_sec,w.score) for w in result]


def fit_ranker(features, targets, weights, config, control='full'):
    from sklearn.ensemble import HistGradientBoostingRegressor
    x=model_inputs(features,control);y=np.asarray(targets,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if len(x)!=len(y) or w.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('Invalid candidate training arrays')
    if not len(x) or np.any((y<0)|(y>1)):raise ValueError('Invalid candidate training utility')
    model=HistGradientBoostingRegressor(**config)
    model.fit(x,y,sample_weight=w)
    if not np.isfinite(model.predict(x)).all():raise ValueError('Nonfinite fitted scores')
    return model
