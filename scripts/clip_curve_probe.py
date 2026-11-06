import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from queryseek.config import Config
from queryseek.eval.runner import load_dataset,_real_embedder
from queryseek.ingest.pipeline import ingest
from queryseek.retrieval.curve_spans import curve_spans
from queryseek.eval.qvhighlights import eval_mr_brief
cfg=Config.load(Path('configs/qvhighlights_v2.yaml')).resolved()
ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='data/qvhighlights/dataset.json');ap.add_argument('--tag',default='clip_curve');args=ap.parse_args()
if not args.tag.replace('_','').isalnum():raise ValueError('Invalid output tag')
samples=load_dataset(args.dataset);pred={};baseline={};gt={};details=[]
for s in samples:
 idx=ingest(s.video_path,video_id=s.video_id,srt_path=s.srt_path,config=cfg)
 emb,mat=_real_embedder(cfg,idx)
 if mat is None:raise ValueError('Cached embeddings required')
 from queryseek.retrieval.semantic import semantic_search
 bw=semantic_search(idx.video_id,s.query,idx.frames,idx.asset.duration_sec,emb,cfg,frame_embeddings=mat,scenes=idx.scenes).windows
 baseline[s.qid]=[[w.start_sec,w.end_sec,w.score] for w in bw]
 scores=mat@emb.encode_texts([s.query])[0]
 ws=curve_spans([f.timestamp_sec for f in idx.frames],scores,idx.asset.duration_sec)
 pred[s.qid]=[[w.start_sec,w.end_sec,w.score] for w in ws]
 gt[s.qid]={'relevant_windows':[[w.start_sec,w.end_sec] for w in s.gt_windows]}
 details.append({'qid':s.qid,'query':s.query,'windows':pred[s.qid]})
 print(f'completed {s.qid}',flush=True)
out=Path('outputs/temporal_probe')/args.tag;out.mkdir(exist_ok=True)
(out/'predictions.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
m={'n_samples':len(samples),'curve':dict(eval_mr_brief(pred,gt)),'semantic_only':dict(eval_mr_brief(baseline,gt))};(out/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps(m,indent=2))

