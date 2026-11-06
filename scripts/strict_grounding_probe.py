"""Strict conjunction grounding on the same 12 development observations."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from queryseek.config import Config
from queryseek.env import ensure_dotenv
from queryseek.models.factory import build_vlm
from queryseek.models.compatible_api import _image_data_url,_parse_json_strict
from queryseek.eval.runner import load_dataset
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.ingest.pipeline import ingest
from queryseek.retrieval.evidence_spans import evidence_spans
from queryseek.retrieval.semantic import semantic_search
from queryseek.eval.runner import _real_embedder
from scripts.temporal_probe import grow
import httpx

def disable_thinking(request):
    body=json.loads(request.content);body['enable_thinking']=False
    request._content=json.dumps(body).encode();request.stream=httpx.ByteStream(request._content)
    request.headers['content-length']=str(len(request._content))

def parse_labels(obj,chunk):
    rows=obj.get('observations')
    if rows is None and isinstance(obj,dict) and all(isinstance(v,dict) for v in obj.values()):
        rows=[dict(v,frame_id=k) for k,v in obj.items()]
    if not isinstance(rows,list):raise ValueError('Missing observations')
    valid={f.frame_id for f in chunk}; out={}
    for r in rows:
        if not isinstance(r,dict):raise ValueError('Invalid observation')
        fid=r.get('frame_id');value=r.get('match')
        if fid not in valid or fid in out or value not in ('yes','no','uncertain'):raise ValueError('Invalid or duplicate judgment')
        out[fid]=r
    if set(out)!=valid:raise ValueError('Incomplete judgments')
    return out

def main():
    ensure_dotenv();cfg=Config.load(Path('configs/qvhighlights_v2.yaml')).resolved()
    cfg.vlm.max_output_tokens=1800;cfg.vlm.max_retries=0;cfg.vlm.read_timeout_sec=60
    out=Path('outputs/temporal_probe')/('strict_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir()
    cfg.vlm.req_log_dir=str(out/'requests');v=build_vlm(cfg)
    if not hasattr(v,'_client'):raise RuntimeError('Real API required')
    v._client.event_hooks['request']=[disable_thinking]
    samples=load_dataset('data/qvhighlights/dataset.json')
    old=[json.loads(l) for l in Path('outputs/temporal_probe/20261011_181037/records.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    gt={s.qid:{'relevant_windows':[[g.start_sec,g.end_sec] for g in s.gt_windows]} for s in samples}
    preds={g:{} for g in ('G','H')};errors=[]
    try:
        for s in samples:
            try:
                idx=ingest(s.video_path,video_id=s.video_id,srt_path=s.srt_path,config=cfg)
                baseline=next(r for r in old if r['qid']==s.qid and r['group']=='A')
                wanted={r['id'] for r in baseline['frames']};fs=sorted([f for f in idx.frames if f.frame_id in wanted],key=lambda f:f.timestamp_sec)
                if len(fs)!=len(wanted):raise ValueError('Evidence replay mismatch')
                labels={};usage=[]
                for k in range(0,len(fs),8):
                    chunk=fs[k:k+8]
                    prompt=('Query: '+s.query+'\nTreat query as conjunction of its defining subject, action, object and spatial relations. A frame matches ONLY when those defining constraints are visibly supported, using neighboring frames for context. Mere presence, a similar action or nearest semantic match is insufficient. Do not infer missing objects or unseen motion. If ambiguous use uncertain. Frames are sparse, so missing motion alone need not negate clearly visible static states. Return JSON {"observations":[{"frame_id":"...","match":"yes|no|uncertain","visible":"short factual description","missing":"unmet query constraints or empty"}]} with exactly one entry per frame.\n'+'\n'.join(f'{f.frame_id} @ {f.timestamp_sec}s' for f in chunk))
                    parts=[{'type':'text','text':prompt}]+[{'type':'image_url','image_url':{'url':_image_data_url(f.image_path)}} for f in chunk]
                    obj=_parse_json_strict(v._chat(parts))
                    (out/f'judgments_{s.qid}_{k}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
                    labels.update(parse_labels(obj,chunk));usage.append(dict(v.last_usage))
                obs=[{'id':f.frame_id,'seconds':f.timestamp_sec,'relevant':labels[f.frame_id]['match']=='yes'} for f in fs]
                spans=evidence_spans(obs,idx.asset.duration_sec)
                preds['H'][s.qid]=[[w.start_sec,w.end_sec,w.score] for w in spans]
                emb,mat=_real_embedder(cfg,idx)
                windows=semantic_search(idx.video_id,s.query,idx.frames,idx.asset.duration_sec,emb,cfg,frame_embeddings=mat,scenes=idx.scenes).windows[:2]
                ids={r['id'] for r in obs if r['relevant']}
                grown=[x for w in windows if (x:=grow(w,fs,ids,idx.asset.duration_sec))];grown.sort(key=lambda x:-x[2]);preds['G'][s.qid]=grown
                with (out/'records.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps({'qid':s.qid,'labels':labels,'frames':obs,'usage':usage,'images':len(fs),'G':grown,'H':preds['H'][s.qid]},ensure_ascii=False)+'\n')
                print(f'completed {s.qid} ({len(preds["G"])}/12)',flush=True)
            except Exception as e:
                errors.append({'qid':s.qid,'type':type(e).__name__});print(f'failed {s.qid}: {type(e).__name__}: {str(e) if isinstance(e,ValueError) else chr(45)}',flush=True)
        metrics={g:{'coverage':len(p)/len(samples),'official':dict(eval_mr_brief({q:p.get(q,[]) for q in gt},gt))} for g,p in preds.items()}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');(out/'errors.json').write_text(json.dumps(errors),encoding='utf-8')
        (out/'protocol.json').write_text(json.dumps({'setting':'strict query conjunction, same sparse observations as previous 12-query probe; historical comparison exploratory only','enable_thinking':False,'G':'strict labels + grow','H':'strict labels + midpoint spans','config':cfg.model_dump(mode='json')},indent=2),encoding='utf-8')
        print(json.dumps(metrics,indent=2),flush=True);print('OUTPUT='+str(out),flush=True)
    finally:v.close()
if __name__=='__main__':main()


