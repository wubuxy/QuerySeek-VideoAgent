"""Opt-in E5: local image evidence for bidirectional E4 boundary adjustment."""
from __future__ import annotations

import math
import time
import uuid
from pathlib import Path

from ..ingest.frames import extract_local_frames
from ..schemas import TimeWindow


def validated_adjustment(obj, original, frames, envelope, min_confidence=0.7):
    """Reject invented evidence, uncertain boundaries and out-of-envelope times."""
    if not isinstance(obj, dict):
        raise ValueError("Expected boundary object")
    verdict = obj.get("verdict")
    confidence = obj.get("confidence")
    if verdict not in ("confirmed", "refuted", "insufficient"):
        raise ValueError("Invalid verdict")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    if verdict != "confirmed" or confidence < min_confidence:
        return original
    ids = obj.get("evidence_frame_ids")
    known = {f.frame_id: f for f in frames}
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or x not in known for x in ids):
        raise ValueError("Unsupported visual evidence")
    a, b = obj.get("start_sec"), obj.get("end_sec")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (a, b)):
        raise ValueError("Invalid boundary")
    if not envelope.start_sec <= a < b <= envelope.end_sec:
        raise ValueError("Boundary outside observed envelope")
    if not any(a <= known[x].timestamp_sec < b for x in ids):
        raise ValueError("No cited observation inside predicted span")
    # Keep CLIP ranking fixed: isolate boundary effects from score calibration.
    return TimeWindow(start_sec=a, end_sec=b, score=original.score, evidence_ids=list(dict.fromkeys(ids)))


def verify_curve_result(result, index, vlm, output_dir, budget_frames=16,
                        max_candidates=2, margin_sec=8.0, min_confidence=0.7):
    """At most two image calls, global presentation budget; safe E4 fallback."""
    output_dir = Path(output_dir).resolve()
    allowed = Path("outputs").resolve()
    if not output_dir.is_relative_to(allowed):
        raise ValueError("Visual observations must stay inside outputs")
    if budget_frames < 0 or max_candidates < 0 or not math.isfinite(margin_sec) or margin_sec < 0:
        raise ValueError("Invalid visual budget")
    started = time.perf_counter()
    windows = list(result.windows)
    used, records, warnings = [], [], list(result.warnings)
    calls = 0
    run_dir = output_dir / uuid.uuid4().hex
    for i, original in enumerate(windows[:max_candidates]):
        count = min(vlm.max_frames_per_call, budget_frames - len(used))
        if count < 2:
            break
        envelope = TimeWindow(start_sec=max(0, original.start_sec-margin_sec),
                              end_sec=min(index.asset.duration_sec, original.end_sec+margin_sec))
        record = {"candidate": i, "before": [original.start_sec, original.end_sec],
                  "envelope": [envelope.start_sec, envelope.end_sec], "calls": 0}
        try:
            # Fresh run directory prevents accidental stale local-frame reuse.
            frames = extract_local_frames(index.asset.path, run_dir / str(i),
                envelope.start_sec, envelope.end_sec,
                fps=count/envelope.duration_sec, tag=f"candidate_{i}")[:count]
            frames = [f for f in frames if envelope.start_sec <= f.timestamp_sec < envelope.end_sec]
            if len(frames) < 2:
                raise ValueError("Insufficient extracted observations")
            used.extend(f.frame_id for f in frames)
            calls += 1
            record["calls"] = 1
            obj = vlm.verify_span(result.query, original, frames, envelope)
            record["usage"] = dict(getattr(vlm, "last_usage", {}))
            adjusted = validated_adjustment(obj, original, frames, envelope, min_confidence)
            windows[i] = adjusted
            record.update(verdict=obj["verdict"], confidence=obj["confidence"],
                          changed=(adjusted.start_sec, adjusted.end_sec) != (original.start_sec, original.end_sec),
                          after=[adjusted.start_sec, adjusted.end_sec])
        except Exception as exc:
            # Store exception class only: provider errors may include response text.
            record.update(error_type=type(exc).__name__, changed=False,
                          after=[original.start_sec, original.end_sec])
            warnings.append(f"Visual boundary candidate {i}: {type(exc).__name__}; retained E4 span")
        records.append(record)
    return result.model_copy(update={"method": "clip_curve_visual_boundary", "windows": windows,
        "used_frame_ids": used, "vlm_calls": calls, "warnings": warnings,
        "elapsed_ms": result.elapsed_ms + (time.perf_counter()-started)*1000,
        "refinement_rounds": records})
