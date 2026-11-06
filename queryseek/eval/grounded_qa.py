"""Grounded-QA input/evaluation glue with a label-free inference interface.

CG-Bench interval-set IoU and acc@IoU definitions are documented in
docs/EVAL_GROUNDED_QA_R1.md. Missing/invalid outputs remain in the denominator.
This module never loads a model or reads ground truth into an inference prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Expected a finite numeric value")
    return result


def checked_intervals(value: Any, duration: float, *, allow_empty: bool = False):
    duration = finite_number(duration)
    if duration <= 0 or not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise ValueError("Invalid duration or intervals")
    intervals = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Expected [start_sec, end_sec)")
        start, end = map(finite_number, pair)
        if not 0 <= start < end <= duration:
            raise ValueError("Interval outside video duration")
        intervals.append((start, end))
    return tuple(intervals)


def merged_intervals(intervals):
    """Union of already validated intervals, without mutating caller input."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_set_iou(predicted, ground_truth, duration):
    pred = merged_intervals(checked_intervals(predicted, duration, allow_empty=True))
    truth = merged_intervals(checked_intervals(ground_truth, duration))
    overlap = sum(max(0., min(b, d) - max(a, c)) for a, b in pred for c, d in truth)
    union = sum(b - a for a, b in pred + truth) - overlap
    return overlap / union if union else 0.


@dataclass(frozen=True)
class GroundedQASample:
    qid: str
    video_id: str
    question: str
    choices: tuple[str, ...]
    duration_sec: float
    right_answer: str
    gt_intervals: tuple[tuple[float, float], ...]

    def model_input(self):
        """Explicit whitelist: labels and answer text never cross this boundary."""
        return {"qid": self.qid, "video_id": self.video_id, "question": self.question,
                "choices": list(self.choices), "duration_sec": self.duration_sec}


def load_samples(rows):
    samples = []
    seen = set()
    for row in rows:
        if isinstance(row.get("qid"), bool) or not isinstance(row.get("qid"), (str, int)):
            raise ValueError("Invalid qid")
        qid = str(row["qid"])
        video = row.get("video_id", row.get("video_uid"))
        choices = row.get("choices")
        question = row.get("question")
        answer = row.get("right_answer")
        if not qid or qid in seen:
            raise ValueError("Empty or duplicate qid")
        if not isinstance(video, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", video):
            raise ValueError("Unsafe video identifier")
        if (not isinstance(choices, list) or not 2 <= len(choices) <= 26
                or any(not isinstance(x, str) or not x.strip() for x in choices)
                or not isinstance(question, str) or not question.strip()):
            raise ValueError("Invalid question or choices")
        if not isinstance(answer, str) or answer not in [chr(65+i) for i in range(len(choices))]:
            raise ValueError("Answer must be an exact choice letter")
        duration = finite_number(row["duration"])
        # All clues are authoritative; never silently select only the first clue.
        intervals = row.get("gt_all_timestamps", row.get("clue_intervals"))
        if intervals is None:
            intervals = [row["gt_timestamps"]]
        truth = checked_intervals(intervals, duration)
        samples.append(GroundedQASample(qid, video, question, tuple(choices), duration, answer, truth))
        seen.add(qid)
    if not samples:
        raise ValueError("Empty dataset")
    return samples


def uniform_times(duration, count):
    duration = finite_number(duration)
    if duration <= 0 or isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Invalid sampling budget")
    return [(i + .5) * duration / count for i in range(count)]


def score_prediction(sample, prediction):
    result = {"qid": sample.qid, "video_id": sample.video_id, "status": "missing",
              "qa_correct": 0, "interval_set_iou": 0., "acc_at_iou": 0., "joint_at_05": 0}
    if prediction is None:
        return result
    if not isinstance(prediction, dict) or prediction.get("status") != "ok":
        result["status"] = "failed"
        return result
    try:
        answer = prediction.get("answer")
        intervals = prediction.get("intervals")
        if answer not in [chr(65+i) for i in range(len(sample.choices))]:
            raise ValueError("Invalid answer")
        if not isinstance(intervals, list) or not 1 <= len(intervals) <= 5:
            raise ValueError("One to five intervals required")
        iou = interval_set_iou(intervals, sample.gt_intervals, sample.duration_sec)
    except (ValueError, TypeError):
        result["status"] = "invalid"
        return result
    correct = int(answer == sample.right_answer)
    result.update(status="ok", qa_correct=correct, interval_set_iou=iou,
                  acc_at_iou=correct * sum(iou >= t for t in THRESHOLDS) / len(THRESHOLDS),
                  joint_at_05=correct * int(iou >= .5))
    return result


def evaluate_predictions(samples, predictions):
    """Use the frozen sample list as denominator, including absent predictions."""
    expected = {s.qid for s in samples}
    if not expected or len(expected) != len(samples):
        raise ValueError("Expected unique nonempty samples")
    indexed = {}
    for row in predictions:
        qid = str(row.get("qid"))
        if qid in indexed or qid not in expected:
            raise ValueError("Duplicate or unexpected prediction qid")
        indexed[qid] = row
    records = [score_prediction(s, indexed.get(s.qid)) for s in samples]
    metrics = {key: sum(r[key] for r in records) / len(records)
               for key in ["qa_correct", "interval_set_iou", "acc_at_iou", "joint_at_05"]}
    metrics.update(n=len(records), valid_outputs=sum(r["status"] == "ok" for r in records),
                   missing_outputs=sum(r["status"] == "missing" for r in records))
    return {"metrics": metrics, "records": records}
