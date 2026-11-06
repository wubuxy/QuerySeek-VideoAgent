import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scipy.stats import binomtest
from queryseek.eval.runner import load_dataset
from queryseek.schemas import TimeWindow
from queryseek.tools.validate import iou
cases=[('80','data/qvhighlights_test/dataset.json','job_20261010_155001_0cad5b'),('200','data/qvhighlights_test200/dataset.json','job_20261011_060742_79c580')]
for n,ds,job in cases:
 path=Path(f'outputs/temporal_probe/clip_curve_{n}/predictions.json')
 if not path.exists():continue
 samples=load_dataset(ds);gt={s.qid:s.gt_windows for s in samples}
 curve={r['qid']:r['windows'] for r in json.loads(path.read_text(encoding='utf-8'))}
 old={r['qid']:r['pred_relevant_windows'] for r in [json.loads(l) for l in Path(f'outputs/eval/{job}/predictions/E1-K16.jsonl').read_text(encoding='utf-8').splitlines()]}
 if set(curve)!=set(gt) or set(old)!=set(gt):raise ValueError('Incomplete or mismatched qids')
 def hit(ws,q,th):return bool(ws) and max(iou(TimeWindow(start_sec=ws[0][0],end_sec=ws[0][1]),g) for g in gt[q])>=th
 res={}
 for th in (.5,.7):
  gains=sum(hit(curve[q],q,th) and not hit(old[q],q,th) for q in gt)
  losses=sum(hit(old[q],q,th) and not hit(curve[q],q,th) for q in gt)
  res[str(th)]={'gains':gains,'losses':losses,'exact_mcnemar_p':binomtest(gains,gains+losses,.5).pvalue if gains+losses else 1}
 out=Path(f'outputs/temporal_probe/clip_curve_{n}/paired_vs_historical_e1.json')
 out.write_text(json.dumps(res,indent=2),encoding='utf-8');print(n,json.dumps(res))
