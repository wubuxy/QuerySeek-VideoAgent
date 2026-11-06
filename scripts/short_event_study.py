import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from queryseek.eval.runner import load_dataset
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.schemas import TimeWindow
from queryseek.tools.validate import iou

def rerank(ws,exponent=.5,cap=None,diversity=None):
    ranked=sorted([[a,b,score/math.sqrt(b-a)*(min(b-a,cap) if cap else b-a)**exponent] for a,b,score in ws],key=lambda x:-x[2])
    if exponent==.5 and cap is None:ranked=[list(w) for w in ws]
    if diversity is None:return ranked[:5]
    from queryseek.retrieval.curve_spans import select_curve_windows
    return [[w.start_sec,w.end_sec,w.score] for w in select_curve_windows([TimeWindow(start_sec=a,end_sec=b,score=sc) for a,b,sc in ranked],5,diversity)]

def load(tag,ds):
    rows=json.loads(Path(f'outputs/temporal_probe/clip_curve_{tag}/predictions.json').read_text(encoding='utf-8'))
    samples=load_dataset(ds);gt={s.qid:{'relevant_windows':[[w.start_sec,w.end_sec] for w in s.gt_windows]} for s in samples}
    if {r['qid'] for r in rows}!=set(gt):raise ValueError('Coverage mismatch')
    return rows,gt

def main():
    out=Path('outputs/temporal_probe/short_event_study');out.mkdir(exist_ok=True)
    specs={'baseline':{},'power_0':{'exponent':0},'power_025':{'exponent':.25},'power_04':{'exponent':.4},'cap_8':{'cap':8},'cap_16':{'cap':16},'cap_32':{'cap':32},'diversity_09':{'diversity':.9},'diversity_07':{'diversity':.7}}
    rows,gt=load('80','data/qvhighlights_test/dataset.json')
    dev={name:dict(eval_mr_brief({r['qid']:rerank(r['windows'],**spec) for r in rows},gt)) for name,spec in specs.items()}
    # Fixed screening criterion: retain R1 and mAP, maximize short-event mAP.
    base=dev['baseline'];eligible=[n for n,m in dev.items() if m['MR-full-R1@0.5']>=base['MR-full-R1@0.5'] and m['MR-full-mAP']>=base['MR-full-mAP']]
    winner=max(eligible,key=lambda n:(dev[n]['MR-short-mAP'],dev[n]['MR-full-mAP']))
    (out/'frozen_selection.json').write_text(json.dumps({'criterion':'dev R1/mAP must not drop; maximize short mAP','winner':winner,'parameters':specs[winner],'dev':dev},indent=2),encoding='utf-8')
    print('DEV='+json.dumps(dev));print('FROZEN='+winner,flush=True)
    rows,gt=load('200','data/qvhighlights_test200/dataset.json')
    validation={n:dict(eval_mr_brief({r['qid']:rerank(r['windows'],**specs[n]) for r in rows},gt)) for n in dict.fromkeys(['baseline',winner])}
    (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    details=[];counts={}
    for r in rows:
        q=r['qid'];ws=r['windows'];g=gt[q]['relevant_windows']
        def overlap(w):return max(iou(TimeWindow(start_sec=w[0],end_sec=w[1]),TimeWindow(start_sec=x[0],end_sec=x[1])) for x in g)
        top=ws[0] if ws else None;best=overlap(top) if top else 0;oracle=max(map(overlap,ws[:5]),default=0)
        if best>=.5:kind='hit'
        elif oracle>=.5:kind='ranking'
        elif top is None:kind='no_proposal'
        elif best==0:kind='wrong_location'
        else:
            matched=max(g,key=lambda x:iou(TimeWindow(start_sec=top[0],end_sec=top[1]),TimeWindow(start_sec=x[0],end_sec=x[1])))
            if top[0]<=matched[0] and top[1]>=matched[1]:kind='too_wide'
            elif matched[0]<=top[0] and matched[1]>=top[1]:kind='too_narrow'
            else:kind='boundary_shift'
        counts[kind]=counts.get(kind,0)+1
        details.append({'qid':q,'kind':kind,'top1_iou':best,'top5_oracle_iou':oracle,'top':top,'gt':g})
    (out/'failure_analysis.json').write_text(json.dumps({'counts':counts,'details':details},indent=2),encoding='utf-8')
    print('VALIDATION='+json.dumps(validation));print('FAILURES='+json.dumps(counts))
if __name__=='__main__':main()

