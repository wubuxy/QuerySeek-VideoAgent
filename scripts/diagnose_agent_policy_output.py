"""At most two explicitly diagnostic repeats; never replace frozen pilot results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from queryseek.agent.search_policy import (DecisionValidationError, SearchState, VisualPolicy,
                                         decision_shape, validate_decision)
from queryseek.config import Config
from queryseek.env import ensure_dotenv
from queryseek.ingest.ffprobe import sha256_file
from queryseek.ingest.frames import load_frames_index
from queryseek.models.factory import build_vlm
from queryseek.tools.export import new_job_id
from scripts.agent_policy_pilot import write_json
from scripts.sampling_probe import checked_output


def diagnose(source, out):
    source, out = checked_output(source), checked_output(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in (source/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    inputs = {r["qid"]: r for r in json.loads((source/"inputs.json").read_text(encoding="utf-8"))}
    selected = [(r, t) for r in rows for t in r["trace"] if t.get("error_type") == "ValueError"][:2]
    if len(selected) != 2:
        raise ValueError("Diagnostic requires first two validation failures in recorded execution order")
    protocol = {"source": str(source), "source_records_sha256": sha256_file(source/"records.jsonl"),
                "selection": [{"qid": r["qid"], "method": r["method"], "turn": t["turn"]} for r, t in selected],
                "max_calls": 2, "max_retries": 0, "changes_to_prompt_or_state": False,
                "claim": "new responses to two failed states; cannot retrospectively classify unsaved original outputs",
                "source_sha256": {n: sha256_file(ROOT/n) for n in
                    ("scripts/diagnose_agent_policy_output.py", "queryseek/agent/search_policy.py")}}
    write_json(out/"protocol.json", protocol)
    ensure_dotenv()
    cfg = Config.load()
    cfg.vlm.max_retries = 0
    cfg.vlm.max_output_tokens = 700
    cfg.vlm.enable_thinking = False
    cfg.vlm.req_log_dir = str(out/"requests")
    vlm = build_vlm(cfg, use_mock=False)
    if not hasattr(vlm, "_chat"):
        raise RuntimeError("Real API required")
    policy = VisualPolicy(vlm, 2)
    records = []
    try:
        for row, trace in selected:
            payload = trace["state"]
            by_id = {f.frame_id: f for f in load_frames_index(Path(inputs[row["qid"]]["index_path"]))}
            frames = [by_id[item["id"]] for item in payload["images"]]
            state = SearchState(payload["query"], payload["duration_sec"], row["method"],
                                tuple(payload["tree_path"]), frames, payload["history"])
            result = {"qid": row["qid"], "method": row["method"], "turn": trace["turn"]}
            try:
                response = policy(payload, frames)
                result["shape"] = decision_shape(response)
                ids = response.get("evidence_ids") if isinstance(response, dict) else None
                if isinstance(ids, list) and all(isinstance(i, str) for i in ids):
                    result["unobserved_id_count"] = sum(i not in {f.frame_id for f in frames} for i in ids)
                    window = result["shape"].get("window")
                    if window:
                        result["observed_citations_outside_window"] = sum(
                            not window[0] <= by_id[i].timestamp_sec < window[1] for i in ids if i in by_id)
                        result["citations_exactly_at_end"] = sum(by_id[i].timestamp_sec == window[1] for i in ids if i in by_id)
                validate_decision(response, state, payload["allowed_actions"])
                result["validation"] = "passed"
            except DecisionValidationError as exc:
                result["validation"] = str(exc)
            except Exception as exc:
                result["validation"] = type(exc).__name__
            records.append(result)
            write_json(out/"diagnostic.json", records)
        write_json(out/"research_status.json", {"phase": "complete", "actual_api_calls": policy.calls,
            "image_presentations": policy.presentations, "usage": dict(vlm.total_usage)})
    finally:
        vlm.close()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    out = ROOT/"outputs/agent_policy_diagnostic"/new_job_id()
    print(out, flush=True)
    print(json.dumps(diagnose(args.source, out), ensure_ascii=False), flush=True)
