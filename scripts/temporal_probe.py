"""Fixed-evidence 2x2 development probe; not the historical E1 baseline."""
import argparse, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from queryseek.config import Config
from queryseek.env import ensure_dotenv
from queryseek.eval.runner import load_dataset, _real_embedder
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.ingest.pipeline import ingest
from queryseek.models.factory import build_vlm
from queryseek.models.compatible_api import _image_data_url, _parse_json_strict
from queryseek.retrieval.semantic import semantic_search
from queryseek.schemas import TimeWindow

def proposals(w, duration):
    boxes = [(w.start_sec, w.end_sec), (max(0,w.start_sec-8),w.end_sec),
             (w.start_sec,min(duration,w.end_sec+8)),
             (max(0,w.start_sec-8),min(duration,w.end_sec+8))]
    return list(dict.fromkeys(boxes))

def select_frames(frames, windows):
    # Two candidate envelopes, eight temporally ordered observations each.
    selected=[]
    for w in windows[:2]:
        pool=[f for f in frames if max(0,w.start_sec-8)<=f.timestamp_sec<=w.end_sec+8]
        if len(pool)>8:
            pool=[pool[round(i*(len(pool)-1)/7)] for i in range(8)]
        for f in pool:
            if f.frame_id not in {x.frame_id for x in selected}: selected.append(f)
    return sorted(selected[:16],key=lambda f:f.timestamp_sec)

def grow(w, frames, ids, duration):
    inside=[f for f in frames if w.start_sec<=f.timestamp_sec<w.end_sec]
    if not inside or not any(f.frame_id in ids for f in inside): return None
    start,end=w.start_sec,w.end_sec
    left=sorted([f for f in frames if f.timestamp_sec<start],key=lambda f:-f.timestamp_sec)[:2]
    right=[f for f in frames if f.timestamp_sec>=end][:2]
    for f in left:
        if f.frame_id not in ids: break
        start=max(0,f.timestamp_sec-1.5)
    for f in right:
        if f.frame_id not in ids: break
        end=min(duration,f.timestamp_sec+1.5)
    return (start,end,sum(f.frame_id in ids for f in inside)/len(inside))

def validate_scores(obj,n):
    scores=obj.get('scores')
    if not isinstance(scores,list) or len(scores)!=n: raise ValueError('Incomplete span scores')
    out=[]
    for x in scores:
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1:
            raise ValueError('Invalid span score')
        out.append(float(x))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=12)
    args=ap.parse_args(); ensure_dotenv()
    cfg=Config.load(ROOT/'configs/qvhighlights_v2.yaml').resolved()
    cfg.vlm.max_output_tokens=1024
    cfg.vlm.read_timeout_sec=60
    cfg.vlm.max_retries=0
    vlm=build_vlm(cfg,use_mock=False)
    def disable_thinking(request):
        body=json.loads(request.content)
        body['enable_thinking']=False
        request._content=json.dumps(body).encode('utf-8')
        request.headers['content-length']=str(len(request._content))
        import httpx
        request.stream=httpx.ByteStream(request._content)
    if hasattr(vlm,'_client'): vlm._client.event_hooks['request']=[disable_thinking]
    if not hasattr(vlm,'_chat'): raise RuntimeError('Real API key required; no silent Mock')
    out=ROOT/'outputs/temporal_probe'/time.strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True)
    cfg.vlm.req_log_dir=str(out/'requests')
    vlm.req_log_dir=out/'requests'; vlm.req_log_dir.mkdir()
    samples=load_dataset(ROOT/'data/qvhighlights/dataset.json')[:args.limit]
    records=[]; errors=[]; preds={g:{} for g in 'ABCD'}
    gt={s.qid:{'relevant_windows':[[w.start_sec,w.end_sec] for w in s.gt_windows]} for s in samples}
    (out/'protocol.json').write_text(json.dumps({'setting':'fixed sparse-index evidence, NOT dense clip captioning; A is matched control, not historical E1','groups':{'A':'frame labels + grow','B':'temporal description + grow','C':'frame labels/reasons + text span scoring','D':'temporal description + text span scoring'},'enable_thinking':False,'max_image_presentations_per_group':16,'max_calls_per_group':3,'config':cfg.model_dump(mode='json')},ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        for s in samples:
            try:
                idx=ingest(s.video_path,video_id=s.video_id,srt_path=s.srt_path,config=cfg)
                emb,mat=_real_embedder(cfg,idx)
                ws=semantic_search(idx.video_id,s.query,idx.frames,idx.asset.duration_sec,emb,cfg,frame_embeddings=mat,scenes=idx.scenes).windows[:2]
                frames=select_frames(idx.frames,ws); boxes=list(dict.fromkeys(b for w in ws for b in proposals(w,idx.asset.duration_sec)))
                if not frames or not boxes: raise ValueError('No semantic evidence')
                for temporal in (False,True):
                    ids=set(); summaries=[]; reasons={}; usages=[]; calls=0; t0=time.monotonic()
                    for k in range(0,len(frames),8):
                        chunk=frames[k:k+8]
                        if temporal:
                            text=('Query: '+s.query+'\nFrames are sparse chronological observations; do not invent motion across gaps. Describe visible actions and transitions, distinguish query action from mere objects. Return JSON with relevant_frame_ids, reasons (frame-id to explanation), summary (event description and uncertainty).\n'+'\n'.join(f'{f.frame_id} @ {f.timestamp_sec:.2f}s' for f in chunk))
                            parts=[{'type':'text','text':text}]+[{'type':'image_url','image_url':{'url':_image_data_url(f.image_path)}} for f in chunk]
                            obj=_parse_json_strict(vlm._chat(parts)); valid={f.frame_id for f in chunk}
                            returned=obj.get('relevant_frame_ids',[])
                            if not isinstance(returned,list): raise ValueError('Invalid frame IDs')
                            ids.update(x for x in returned if isinstance(x,str) and x in valid)
                            summaries.append(str(obj.get('summary','')))
                            rr=obj.get('reasons',{}); reasons.update(rr if isinstance(rr,dict) else {})
                        else:
                            res=vlm.inspect_frames(s.query,chunk); ids.update(res.relevant_frame_ids)
                            summaries.append(res.summary); reasons.update({j.frame_id:j.reason for j in res.judgments})
                        calls+=1; usages.append(dict(vlm.last_usage))
                    group='B' if temporal else 'A'
                    grown=[x for w in ws if (x:=grow(w,frames,ids,idx.asset.duration_sec))]
                    grown.sort(key=lambda x:-x[2]); preds[group][s.qid]=grown
                    records.append({'qid':s.qid,'group':group,'images':len(frames),'calls':calls,'usage':usages.copy(),'elapsed_sec':time.monotonic()-t0,'frames':[{'id':f.frame_id,'seconds':f.timestamp_sec,'relevant':f.frame_id in ids} for f in frames],'summary':summaries,'windows':grown})
                    prompt=('Query: '+s.query+'\nScore each candidate interval for complete query event coverage and exclusion of unrelated content, using only supplied sparse evidence. Do not invent unseen transitions. Return JSON {"scores": [number 0..1 for each candidate in order]}.\n'+json.dumps({'candidates':boxes,'observations':[{'id':f.frame_id,'seconds':f.timestamp_sec,'relevant':f.frame_id in ids,'reason':reasons.get(f.frame_id,'')} for f in frames],'descriptions':summaries},ensure_ascii=False))
                    obj=_parse_json_strict(vlm._chat([{'type':'text','text':prompt}]))
                    scores=validate_scores(obj,len(boxes)); usages.append(dict(vlm.last_usage))
                    ranked=sorted([(a,b,sc) for (a,b),sc in zip(boxes,scores)],key=lambda x:-x[2])
                    group='D' if temporal else 'C'; preds[group][s.qid]=ranked
                    records.append({'qid':s.qid,'group':group,'images':len(frames),'calls':calls+1,'usage':usages.copy(),'elapsed_sec':time.monotonic()-t0,'windows':ranked})
                    with (out/'records.jsonl').open('a',encoding='utf-8') as h:
                        for r in records[-2:]: h.write(json.dumps(r,ensure_ascii=False)+'\n')
                print(f'completed qid={s.qid} ({len(preds["D"])}/{len(samples)})',flush=True)
            except Exception as exc:
                errors.append({'qid':s.qid,'error_type':type(exc).__name__})
                print(f'failed qid={s.qid}: {type(exc).__name__}',flush=True)
            (out/'errors.json').write_text(json.dumps(errors),encoding='utf-8')
        metrics={g:{'coverage':len(preds[g])/len(samples),'official':eval_mr_brief({q:preds[g].get(q,[]) for q in gt},gt)} for g in 'ABCD'}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print(json.dumps(metrics,indent=2),flush=True); print('OUTPUT='+str(out),flush=True)
    finally: vlm.close()
if __name__=='__main__': main()



