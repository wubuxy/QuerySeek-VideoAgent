import json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from queryseek.eval.runner import load_dataset
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.retrieval.fusion import nms_dedupe
from queryseek.schemas import TimeWindow
from queryseek.tools.validate import iou
for tag,ds in [('80','data/qvhighlights_test/dataset.json'),('200','data/qvhighlights_test200/dataset.json')]:
 root=Path('outputs/temporal_probe')/f'clip_curve_{tag}'
 if not (root/'predictions.json').exists():continue
 rows=json.loads((root/'predictions.json').read_text(encoding='utf-8'));samples=load_dataset(ds)
 gt={s.qid:{'relevant_windows':[[g.start_sec,g.end_sec] for g in s.gt_windows]} for s in samples}
 strength={};suppressed={};top5={};oracle=0
 for r in rows:
  q=r['qid'];ws=r['windows']
  strength[q]=sorted([[a,b,score/math.sqrt(b-a)] for a,b,score in ws],key=lambda w:-w[2])
  top5[q]=ws[:5]
  cleaned=nms_dedupe([TimeWindow(start_sec=a,end_sec=b,score=sc) for a,b,sc in ws],.5)[:5]
  suppressed[q]=[[w.start_sec,w.end_sec,w.score] for w in cleaned]
  oracle+=any(max(iou(TimeWindow(start_sec=w[0],end_sec=w[1]),TimeWindow(start_sec=g[0],end_sec=g[1])) for g in gt[q]['relevant_windows'])>=.5 for w in ws)
 result={'without_duration_prior':dict(eval_mr_brief(strength,gt)),'top5':dict(eval_mr_brief(top5,gt)),'nms_top5':dict(eval_mr_brief(suppressed,gt)),'proposal_oracle_R1@0.5':oracle/len(samples)}
 (root/'ablations.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(tag,json.dumps(result))
