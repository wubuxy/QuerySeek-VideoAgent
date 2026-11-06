"""Temporal spans from observed transitions; no ground truth used during generation."""
import math
from queryseek.schemas import TimeWindow

def evidence_spans(observations, duration, max_gap=8.0, max_width=120.0):
    if not math.isfinite(duration) or duration<=0: raise ValueError('Invalid duration')
    if not math.isfinite(max_gap) or max_gap<=0 or not math.isfinite(max_width) or max_width<=0:
        raise ValueError('Invalid gap or width')
    rows=sorted(observations,key=lambda x:x['seconds'])
    if any(not math.isfinite(r['seconds']) or not 0<=r['seconds']<=duration for r in rows):
        raise ValueError('Invalid timestamp')
    runs=[]
    for i,r in enumerate(rows):
        if not r['relevant']: continue
        if runs and i==runs[-1][-1]+1 and r['seconds']-rows[runs[-1][-1]]['seconds']<=max_gap:
            runs[-1].append(i)
        else: runs.append([i])
    result=[]
    for run in runs:
        first,last=run[0],run[-1]; a,b=rows[first]['seconds'],rows[last]['seconds']
        # A negative neighbor brackets a transition; an unobserved edge is capped.
        left=min(max_gap/2,(a-rows[first-1]['seconds'])/2) if first else min(max_gap/2,a)
        right=min(max_gap/2,(rows[last+1]['seconds']-b)/2) if last+1<len(rows) else min(max_gap/2,duration-b)
        start,end=max(0,a-left),min(duration,b+right)
        if end<=start: continue
        # Long runs are split rather than silently violating the public contract.
        while start<end:
            stop=min(end,start+max_width)
            count=sum(start<=rows[i]['seconds']<stop for i in run)
            if count:
                result.append(TimeWindow(start_sec=start,end_sec=stop,score=float(count)))
            start=stop
    return sorted(result,key=lambda w:(-w.score,w.start_sec))

