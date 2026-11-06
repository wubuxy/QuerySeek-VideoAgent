import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from queryseek.eval.runner import load_dataset
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.retrieval.evidence_spans import evidence_spans
from queryseek.tools.validate import iou
from queryseek.schemas import TimeWindow
ROOT=Path(__file__).resolve().parents[1]
records=[json.loads(l) for l in (ROOT/'outputs/temporal_probe/20261011_181037/records.jsonl').read_text(encoding='utf-8-sig').splitlines()]
samples=load_dataset(ROOT/'data/qvhighlights/dataset.json'); byq={s.qid:s for s in samples}
gt={s.qid:{'relevant_windows':[[w.start_sec,w.end_sec] for w in s.gt_windows]} for s in samples}
out=ROOT/'outputs/temporal_probe/evidence_replay';out.mkdir(exist_ok=True)
pred={g:{} for g in ['A','B','C','D','E','F']}; details=[]
for r in records:
 q=r['qid'];pred[r['group']][q]=r['windows']
 if r['group'] not in 'AB':continue
 s=byq[q];duration=s.duration_sec
 if not duration:
  from queryseek.ingest.ffprobe import probe
  duration=float(probe(s.video_path)['format']['duration'])
 ws=evidence_spans(r['frames'],duration)
 target='E' if r['group']=='A' else 'F'
 pred[target][q]=[[w.start_sec,w.end_sec,w.score] for w in ws]
 def best(box):return max(iou(TimeWindow(start_sec=box[0],end_sec=box[1]),g) for g in s.gt_windows)
 details.append({'qid':q,'group':target,'query':s.query,'windows':pred[target][q],'top1_iou':best(pred[target][q][0]) if ws else 0,'oracle_iou':max((best(w) for w in pred[target][q]),default=0)})
metrics={g:dict(eval_mr_brief(p,gt)) for g,p in pred.items()}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
(out/'per_sample.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'predictions.json').write_text(json.dumps(pred),encoding='utf-8')
print(json.dumps(metrics,indent=2))

