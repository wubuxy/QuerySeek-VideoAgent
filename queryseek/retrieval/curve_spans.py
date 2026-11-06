"""Training-free multi-threshold proposals from a full-video CLIP curve."""
import numpy as np
from queryseek.schemas import Evidence, TimeWindow

def curve_spans(times,scores,duration):
    t=np.asarray(times,dtype=float);s=np.asarray(scores,dtype=float)
    if len(t)!=len(s) or not len(t): return []
    if not np.isfinite(t).all() or not np.isfinite(s).all() or np.any(np.diff(t)<0) or not np.isfinite(duration) or duration<=0: raise ValueError('Invalid curve')
    if np.any(t<0) or np.any(t>duration):raise ValueError('Invalid timestamps')
    if len(t)==1:return [TimeWindow(start_sec=max(0,t[0]-2),end_sec=min(duration,t[0]+2),score=float(s[0]))]
    mids=(t[:-1]+t[1:])/2;edges=np.concatenate(([0.],mids,[duration]))
    smooth=np.convolve(np.pad(s,(1,1),mode='edge'),np.array([.25,.5,.25]),mode='valid')
    center=np.median(smooth);scale=max(float(np.std(smooth)),1e-6)
    z=(smooth-center)/scale;out={}
    for quantile in (.5,.7,.85):
        mask=smooth>=np.quantile(smooth,quantile)
        runs=[]
        for i,on in enumerate(mask):
            if not on:continue
            if runs and i==runs[-1][-1]+1:runs[-1].append(i)
            else:runs.append([i])
        for run in runs:
            a,b=edges[run[0]],edges[run[-1]+1]
            if b<=a or b-a>120:continue
            # Score supported temporal mass, not number of irregular scene frames.
            widths=np.diff(edges)[run];strength=float(np.average(np.maximum(z[run],0),weights=widths))
            score=strength*np.sqrt(b-a)
            out[(float(a),float(b))]=max(out.get((float(a),float(b)),0),score)
    return [TimeWindow(start_sec=a,end_sec=b,score=sc) for (a,b),sc in sorted(out.items(),key=lambda x:-x[1])]


def select_curve_windows(windows, top_k=5, nms_iou=0.9):
    """Remove near-identical proposals while retaining the highest-score span."""
    from queryseek.retrieval.fusion import nms_dedupe
    if top_k < 0 or not 0 <= nms_iou <= 1:
        raise ValueError("Invalid selection limits")
    return nms_dedupe(windows, nms_iou)[:top_k]

def short_peak_spans(times, scores, duration):
    """Additional 2/4/8-second proposals at unsmoothed local peaks; no GT."""
    t, s = np.asarray(times, dtype=float), np.asarray(scores, dtype=float)
    if not len(t):
        return []
    curve_spans(t, s, duration)  # reuse input validation
    center, scale = np.median(s), max(float(np.std(s)), 1e-6)
    threshold = np.quantile(s, .85)
    out = []
    for i in range(len(t)):
        if s[i] < threshold or (i > 0 and s[i] <= s[i-1]) or (i+1 < len(t) and s[i] < s[i+1]):
            continue
        strength = max(float((s[i]-center)/scale), 0)
        for length in (2., 4., 8.):
            a, b = max(0., t[i]-length/2), min(duration, t[i]+length/2)
            if a < b:
                out.append(TimeWindow(start_sec=a, end_sec=b, score=strength*np.sqrt(b-a)))
    return out


def curve_search(video_id, query, index, embedder, config, frame_embeddings=None, include_short=False, trace=None):
    """E4: cached CLIP features only; zero VLM observations/calls."""
    import time
    from queryseek.schemas import SearchResult
    start=time.perf_counter()
    frames=index.frames
    if not frames:
        return SearchResult(video_id=video_id,query=query,method='clip_curve')
    mat=np.asarray(frame_embeddings if frame_embeddings is not None else embedder.encode_images([f.image_path for f in frames]))
    if mat.ndim!=2 or mat.shape[0]!=len(frames):raise ValueError('Frame embedding mismatch')
    scores=mat@embedder.encode_texts([query])[0]
    windows=curve_spans([f.timestamp_sec for f in frames],scores,index.asset.duration_sec)
    if include_short:
        windows += short_peak_spans([f.timestamp_sec for f in frames], scores, index.asset.duration_sec)
    selected = select_curve_windows(windows, config.retrieval.top_k_windows)
    evidence = {}
    for window in selected:
        members = [i for i, frame in enumerate(frames)
                   if window.start_sec <= frame.timestamp_sec < window.end_sec]
        if members:
            peak = max(members, key=lambda i: scores[i])
            frame = frames[peak]
            eid = f"clip_{frame.frame_id}"
            evidence[eid] = Evidence(
                evidence_id=eid, type="frame", timestamp_sec=frame.timestamp_sec,
                content=f"CLIP 相似度 {float(scores[peak]):.6f}；仅为检索线索，未经 VLM 确认",
                source_file=frame.image_path, model_id=getattr(embedder, "model_id", None),
                score=float(scores[peak]),
            )
            window.evidence_ids = [eid]
    result = SearchResult(
        video_id=video_id, query=query, method="clip_curve", windows=selected,
        indexed_frames=len(frames), online_encoded_frames=0 if frame_embeddings is not None else len(frames),
        embedding_cache_hit=frame_embeddings is not None,
        evidence=list(evidence.values()), vlm_calls=0,
        elapsed_ms=(time.perf_counter()-start)*1000,
        warnings=["CLIP 相对峰值仅用于生成候选，不能证明查询事件存在；负查询拒答尚未验证"],
    )
    if trace is not None:
        trace.event("curve_search", query=query, indexed_frames=result.indexed_frames,
                    online_encoded_frames=result.online_encoded_frames,
                    embedding_cache_hit=result.embedding_cache_hit, vlm_calls=0,
                    windows=[w.model_dump() for w in selected],
                    evidence=[e.model_dump() for e in result.evidence], elapsed_ms=result.elapsed_ms)
    return result

