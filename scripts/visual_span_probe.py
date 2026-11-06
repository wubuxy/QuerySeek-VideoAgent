"""Small matched E4 / short proposals / short+visual boundary probe."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from queryseek.config import Config
from queryseek.env import ensure_dotenv
from queryseek.eval.runner import load_dataset, _real_embedder
from queryseek.eval.qvhighlights import eval_mr_brief
from queryseek.ingest.pipeline import ingest
from queryseek.models.factory import build_vlm
from queryseek.retrieval.curve_spans import curve_search
from queryseek.retrieval.visual_spans import verify_curve_result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=6)
    ap.add_argument("--mock",action="store_true")
    ap.add_argument("--visual-source",choices=("original","short"),default="short",
                    help="original: E4+VLM (E4V); short: E4S+VLM (E5)")
    args=ap.parse_args()
    if not 1<=args.limit<=12:raise ValueError("Pilot limit must be 1..12")
    ensure_dotenv()
    cfg=Config.load(ROOT/"configs/qvhighlights_v2.yaml").resolved()
    cfg.vlm.max_retries=0;cfg.vlm.read_timeout_sec=60;cfg.vlm.max_output_tokens=768
    cfg.vlm.enable_thinking=False
    out=ROOT/"outputs/temporal_probe"/("visual_span_"+time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    cfg.vlm.req_log_dir=str(out/"requests")
    vlm=build_vlm(cfg,use_mock=args.mock)
    if not args.mock and not hasattr(vlm,"_chat"):raise RuntimeError("Real key required; refusing silent Mock")
    if hasattr(vlm,"enable_thinking"):vlm.enable_thinking=False
    samples=load_dataset(ROOT/"data/qvhighlights/dataset.json")[:args.limit]
    visual_label = "E4V" if args.visual_source == "original" else "E5"
    (out/"protocol.json").write_text(json.dumps({"sample_qids":[s.qid for s in samples],
        "mock":args.mock,"max_image_presentations":16,"max_calls":2,
        "margin_sec":8,"min_confidence":.7,"enable_thinking":False,
        "groups":["E4","E4S",visual_label],"visual_source":args.visual_source,
        "ranking":"CLIP scores unchanged during VLM boundary adjustment",
        "config":cfg.model_dump(mode="json")},indent=2),encoding="utf-8")
    predictions={k:{} for k in ("E4","E4S",visual_label)};errors=[];completed=0
    try:
        for s in samples:
            try:
                idx=ingest(s.video_path,video_id=s.video_id,srt_path=s.srt_path,config=cfg)
                emb,mat=_real_embedder(cfg,idx)
                baseline=curve_search(idx.video_id,s.query,idx,emb,cfg,mat)
                short=curve_search(idx.video_id,s.query,idx,emb,cfg,mat,include_short=True)
                source = baseline if args.visual_source == "original" else short
                visual=verify_curve_result(source,idx,vlm,out/"frames")
                for label,result in zip(predictions,(baseline,short,visual)):
                    predictions[label][s.qid]=[[w.start_sec,w.end_sec,w.score or 0] for w in result.windows]
                    record={"qid":s.qid,"group":label,"result":result.model_dump(mode="json")}
                    with (out/"records.jsonl").open("a",encoding="utf-8") as f:
                        f.write(json.dumps(record,ensure_ascii=False)+"\n")
                completed+=1
                print(f"completed {completed}/{len(samples)} qid={s.qid}; calls={visual.vlm_calls}",flush=True)
            except Exception as exc:
                errors.append({"qid":s.qid,"error_type":type(exc).__name__})
                print(f"failed qid={s.qid}: {type(exc).__name__}",flush=True)
        gt={s.qid:{"relevant_windows":[[w.start_sec,w.end_sec] for w in s.gt_windows]} for s in samples}
        metrics={k:dict(eval_mr_brief({q:predictions[k].get(q,[]) for q in gt},gt)) for k in predictions}
        summary={"n_samples":len(samples),"completed":completed,"mock":args.mock,
                 "metrics":metrics,"usage":getattr(vlm,"total_usage",{}),"errors":errors}
        (out/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2),flush=True)
        print("OUTPUT="+str(out),flush=True)
    finally:
        if hasattr(vlm,"close"):vlm.close()


if __name__=="__main__":main()
