"""Finite evidence-anchor actions. No model-generated seconds or hidden GT.

Cells are the midpoints between unique SOURCE timestamps in the frozen pool.
Selecting two observed anchors returns the union envelope of their cells.
This is an explicit discretized localization interface, not boundary regression.
"""
from __future__ import annotations

import json
import math

from queryseek.agent.search_policy import DecisionValidationError

ANCHOR_VERSION = "source-midpoint-anchors-v1"
ANCHOR_PROMPT = """Locate the exact visible event described by the query.
Only the supplied images are evidence. You are choosing one search action.
zoom_in_left/right narrows the current window. zoom_out returns to its parent.
shift moves to the sibling window. observe obtains more unseen frames here.
If the current branch lacks the event, consider returning or switching to search elsewhere.
Earlier images remain available. Stop when sufficient evidence supports an answer.
Each image has an anchor_index and a support_cell_sec. These are source-time coordinates.
To answer choose "answer:i:j" for observed anchor indices i <= j: i is the first
supporting image, j the last. The program returns their cell envelope and cites
those endpoint images. Select a single contiguous occurrence, not unrelated scenes.
Both endpoints must visually support the event. "answer:i:i" selects one cell.
Other choices are exactly one of the available navigation actions or "abstain".
Do not invent an event in unseen images. Abstain if evidence is insufficient.
Return ONLY {"choice":"..."} with one valid choice. No explanations or other fields.
"""


def support_cells(frames, duration):
    if not math.isfinite(duration) or duration <= 0 or not frames:
        raise ValueError("Invalid anchor pool")
    if len({f.frame_id for f in frames}) != len(frames):
        raise ValueError("Duplicate frame id")
    times = sorted({float(f.timestamp_sec) for f in frames})
    if any(not math.isfinite(t) or not 0 <= t < duration for t in times):
        raise ValueError("Source timestamps outside video")
    boundaries = [0.] + [(a+b)/2 for a, b in zip(times, times[1:])] + [float(duration)]
    by_time = {t: [boundaries[i], boundaries[i+1]] for i, t in enumerate(times)}
    return {f.frame_id: by_time[float(f.timestamp_sec)] for f in frames}


def anchor_catalog(payload, frames, cells):
    # Frames supplied here are observed frames ONLY, never the full image pool.
    ordered = sorted(frames, key=lambda f: (f.timestamp_sec, f.frame_id))
    if len({f.frame_id for f in ordered}) != len(ordered):
        raise ValueError("Duplicate observed id")
    catalog = {a: {"action": a, "window": None, "evidence_ids": [], "reason": "anchor-policy"}
               for a in payload["allowed_actions"] if a != "answer"}
    if "answer" in payload["allowed_actions"]:
        for i, first in enumerate(ordered):
            for j in range(i, len(ordered)):
                last = ordered[j]
                ids = list(dict.fromkeys([first.frame_id, last.frame_id]))
                catalog[f"answer:{i}:{j}"] = {
                    "action": "answer",
                    "window": [float(cells[first.frame_id][0]), float(cells[last.frame_id][1])],
                    "evidence_ids": ids, "reason": "selected-observed-anchors"}
    shown = [{**item, "anchor_index": i, "support_cell_sec": cells[f.frame_id]}
             for i, f in enumerate(ordered)
             for item in [{"id": f.frame_id, "source_time_sec": f.timestamp_sec}]]
    return ordered, catalog, shown


class AnchoredPolicy:
    def __init__(self, backend, frames, duration):
        self.backend = backend
        self.cells = support_cells(frames, duration)

    def __call__(self, payload, frames):
        ordered, catalog, shown = anchor_catalog(payload, frames, self.cells)
        safe_payload = {**payload, "images": shown}
        schema = {"type": "object", "properties": {
            "choice": {"type": "string", "enum": list(catalog)}},
            "required": ["choice"], "additionalProperties": False}
        obj = self.backend(ANCHOR_PROMPT + json.dumps(safe_payload, ensure_ascii=False),
                           ordered, schema)
        if (not isinstance(obj, dict) or set(obj) != {"choice"}
                or not isinstance(obj["choice"], str) or obj["choice"] not in catalog):
            raise DecisionValidationError("invalid_action_schema")
        return catalog[obj["choice"]]
