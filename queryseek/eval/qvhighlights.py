"""QVHighlights / Moment-DETR 官方评测协议适配（开发说明书 §8.2、PAPER_NOTES §3）。

- 数据与提交格式核实自官方仓库 moment_detr@b7e553a（standalone_eval/）：
  GT: {qid, query, vid, duration, relevant_windows: [[s,e],...]}
  pred: {qid, query, vid, pred_relevant_windows: [[s,e,score],...]}
- 指标定义**对齐官方** eval.py/utils.py 语义（IoU[0.5:0.95:0.05] 10 档、逐查询
  AP 的插值 PR、mAP = 查询×阈值均值、R1 与最佳匹配 GT、short/middle/long/full
  长度分层）——实现为独立重写（官方代码 MIT，允许；未复制源码）。
- 官方 eval.py 对"零预测"查询会 IndexError；本项目明确处理：零预测 → AP/R1 记 0
  （该偏差记录于 docs/DECISIONS.md）。
- mIoU / 边界误差是**本项目自定义指标**（非官方），所有输出显式标注 custom。
- 标注文件 CC BY-NC-SA 4.0：评测可运行，禁止商用重分发（THIRD_PARTY.md）。
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import TimeWindow
from ..tools.validate import iou as _window_iou

IOU_THDS = [round(float(x), 2) for x in np.linspace(0.5, 0.95, 10)]
LENGTH_RANGES: dict[str, tuple[float, float]] = {
    "short": (0.0, 10.0),
    "middle": (10.0, 30.0),
    "long": (30.0, 150.0),
    "full": (0.0, 150.0),
}

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_qv_gt(path: str | Path) -> dict[int, dict[str, Any]]:
    """QVHighlights GT JSONL -> {qid: {query, vid, duration, relevant_windows}}。"""
    out: dict[int, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[int(d["qid"])] = {
            "qid": int(d["qid"]),
            "query": d["query"],
            "vid": d.get("vid", ""),
            "duration": float(d.get("duration", 0.0)),
            "relevant_windows": [[float(w[0]), float(w[1])] for w in d.get("relevant_windows", [])],
        }
    return out


def load_submission(path: str | Path) -> dict[int, dict[str, Any]]:
    """官方提交格式 JSONL -> {qid: {...}}（pred_relevant_windows 保留分数）。"""
    out: dict[int, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[int(d["qid"])] = d
    return out


def save_submission(submission: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for d in submission:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def submission_from_windows(
    qid: int, query: str, vid: str, windows: list[TimeWindow]
) -> dict[str, Any]:
    """统一 TimeWindow 契约 -> 官方提交条目（[s, e, score]）。"""
    preds = [
        [round(w.start_sec, 3), round(w.end_sec, 3), float(w.score if w.score is not None else 0.0)]
        for w in windows
    ]
    return {"qid": qid, "query": query, "vid": vid, "pred_relevant_windows": preds}


# ---------------------------------------------------------------------------
# 基础度量（与官方语义一致）
# ---------------------------------------------------------------------------


def temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return float(inter / union) if union > 0 else 0.0


def _interpolated_precision_recall(precision: np.ndarray, recall: np.ndarray) -> float:
    mprec = np.hstack([[0.0], precision, [0.0]])
    mrec = np.hstack([[0.0], recall, [1.0]])
    for i in range(len(mprec) - 1, 0, -1):
        mprec[i - 1] = max(mprec[i - 1], mprec[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def compute_ap(gt_windows: list[tuple[float, float]], pred_windows: list[tuple[float, float, float]]) -> np.ndarray:
    """逐查询 AP（每 IoU 阈值一个值；与官方 compute_average_precision_detection 同语义）。"""
    n_thds = len(IOU_THDS)
    if not pred_windows:
        return np.zeros(n_thds)
    num_gts = len(gt_windows)
    preds = sorted(pred_windows, key=lambda p: -p[2])  # 分数降序
    tp = np.zeros((n_thds, len(preds)))
    fp = np.zeros((n_thds, len(preds)))
    lock_gt = np.full((n_thds, num_gts), -1)
    for idx, (ps, pe, _score) in enumerate(preds):
        ious = np.array([temporal_iou(ps, pe, gs, ge) for gs, ge in gt_windows])
        order = np.argsort(-ious)
        for t_idx, thd in enumerate(IOU_THDS):
            matched = False
            for j_idx in order:
                if ious[j_idx] < thd:
                    break
                if lock_gt[t_idx, j_idx] >= 0:
                    continue
                tp[t_idx, idx] = 1
                lock_gt[t_idx, j_idx] = idx
                matched = True
                break
            if not matched:
                fp[t_idx, idx] = 1
    tp_cs = np.cumsum(tp, axis=1).astype(float)
    fp_cs = np.cumsum(fp, axis=1).astype(float)
    recall = tp_cs / num_gts if num_gts > 0 else np.zeros_like(tp_cs)
    precision = tp_cs / (tp_cs + fp_cs + 1e-12)
    ap = np.zeros(n_thds)
    for t_idx in range(n_thds):
        ap[t_idx] = _interpolated_precision_recall(precision[t_idx], recall[t_idx])
    return ap


def _round2(v: float) -> float:
    return float(f"{v:.2f}")


def compute_mr_ap(pred_by_qid: dict[int, list[tuple[float, float, float]]],
                  gt_by_qid: dict[int, dict[str, Any]],
                  max_pred_windows: int = 10) -> dict[str, float]:
    """mAP @ IoU 10 档 + average（与官方 compute_mr_ap 同语义，单位 %）。"""
    per_qid = {}
    for qid, preds in pred_by_qid.items():
        gt = gt_by_qid.get(qid)
        if not gt:
            continue
        per_qid[qid] = compute_ap(
            [tuple(w) for w in gt["relevant_windows"]], preds[:max_pred_windows]
        )
    if not per_qid:
        return {str(t): 0.0 for t in IOU_THDS} | {"average": 0.0}
    ap_array = np.array(list(per_qid.values()))
    out = {str(t): _round2(float(ap_array[:, i].mean()) * 100) for i, t in enumerate(IOU_THDS)}
    out["average"] = _round2(float(ap_array.mean()) * 100)
    return out


def compute_mr_r1(pred_by_qid: dict[int, list[tuple[float, float, float]]],
                  gt_by_qid: dict[int, dict[str, Any]]) -> dict[str, float]:
    """R1：top-1 预测与 IoU 最高的 GT 窗口比较（官方语义：取第一个预测窗口）。"""
    per_qid: dict[int, float] = {}
    for qid, preds in pred_by_qid.items():
        gt = gt_by_qid.get(qid)
        if not gt:
            continue
        gt_ws = [tuple(w) for w in gt["relevant_windows"]]
        if not preds or not gt_ws:
            per_qid[qid] = 0.0
            continue
        ps, pe, _ = preds[0]  # 官方取第一个（而非分数最高）
        per_qid[qid] = max(temporal_iou(ps, pe, gs, ge) for gs, ge in gt_ws)
    vals = list(per_qid.values())
    if not vals:
        return {str(t): 0.0 for t in IOU_THDS}
    return {str(t): _round2(float(np.mean(np.array(vals) >= t)) * 100) for t in IOU_THDS}


def _filter_length(pred_by_qid, gt_by_qid, lo: float, hi: float):
    """GT 窗口长度在 (lo, hi] 内的查询子集（官方 get_data_by_range 语义）。"""
    gt_out: dict[int, dict[str, Any]] = {}
    for qid, gt in gt_by_qid.items():
        ws = [w for w in gt["relevant_windows"] if lo < (w[1] - w[0]) <= hi]
        if ws:
            gt_out[qid] = {**gt, "relevant_windows": ws}
    pred_out = {qid: pred_by_qid[qid] for qid in gt_out if qid in pred_by_qid}
    return pred_out, gt_out


def eval_moment_retrieval(pred_by_qid, gt_by_qid) -> dict[str, dict[str, Any]]:
    """full/short/middle/long 分层的 MR-mAP 与 MR-R1（官方 eval_moment_retrieval 语义）。"""
    ret: dict[str, dict[str, Any]] = {}
    for name, (lo, hi) in LENGTH_RANGES.items():
        p, g = _filter_length(pred_by_qid, gt_by_qid, lo, hi)
        ret[name] = {
            "MR-mAP": compute_mr_ap(p, g),
            "MR-R1": compute_mr_r1(p, g),
        }
    return ret


def eval_mr_brief(pred_by_qid, gt_by_qid) -> OrderedDict:
    """官方 brief 键集（MR 部分）：MR-full-mAP / mAP@0.5 / mAP@0.75 / R1@0.5 / R1@0.7 等。"""
    m = eval_moment_retrieval(pred_by_qid, gt_by_qid)
    brief = OrderedDict(
        sorted(
            {
                "MR-full-mAP": m["full"]["MR-mAP"]["average"],
                "MR-full-mAP@0.5": m["full"]["MR-mAP"]["0.5"],
                "MR-full-mAP@0.75": m["full"]["MR-mAP"]["0.75"],
                "MR-short-mAP": m["short"]["MR-mAP"]["average"],
                "MR-middle-mAP": m["middle"]["MR-mAP"]["average"],
                "MR-long-mAP": m["long"]["MR-mAP"]["average"],
                "MR-full-R1@0.5": m["full"]["MR-R1"]["0.5"],
                "MR-full-R1@0.7": m["full"]["MR-R1"]["0.7"],
            }.items()
        )
    )
    return brief


# ---------------------------------------------------------------------------
# 自定义指标（显式标记 custom，非官方）
# ---------------------------------------------------------------------------


def compute_miou(pred_by_qid, gt_by_qid) -> dict[str, float]:
    """custom-mIoU：top-1 预测与最佳匹配 GT 的 IoU，按查询取均值。"""
    vals = []
    for qid, preds in pred_by_qid.items():
        gt = gt_by_qid.get(qid)
        if not gt:
            continue
        gt_ws = [tuple(w) for w in gt["relevant_windows"]]
        if not preds or not gt_ws:
            vals.append(0.0)
            continue
        ps, pe, _ = preds[0]
        vals.append(max(temporal_iou(ps, pe, gs, ge) for gs, ge in gt_ws))
    return {"custom_miou": float(np.mean(vals)) if vals else 0.0,
            "custom_hit1_iou05": float(np.mean(np.array(vals) >= 0.5)) if vals else 0.0}


def compute_boundary_error(pred_by_qid, gt_by_qid) -> dict[str, float]:
    """custom-boundary-error：|中心偏差| / GT 时长（匹配对取最佳 IoU 的 GT）。"""
    errs = []
    for qid, preds in pred_by_qid.items():
        gt = gt_by_qid.get(qid)
        if not gt:
            continue
        gt_ws = [tuple(w) for w in gt["relevant_windows"]]
        if not preds or not gt_ws:
            continue
        ps, pe, _ = preds[0]
        gs, ge = max(gt_ws, key=lambda w: temporal_iou(ps, pe, w[0], w[1]))
        errs.append(abs((ps + pe) / 2 - (gs + ge) / 2) / max(ge - gs, 1e-9))
    return {"custom_boundary_error": float(np.mean(errs)) if errs else float("nan")}


# ---------------------------------------------------------------------------
# 官方提交级评估入口
# ---------------------------------------------------------------------------


def evaluate_submission(submission: list[dict[str, Any]], gt_by_qid: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """对官方格式提交做 MR 评估（不含 HL 显著性——本项目不预测 saliency）。"""
    pred_by_qid: dict[int, list[tuple[float, float, float]]] = {}
    for d in submission:
        pred_by_qid[int(d["qid"])] = [
            (float(w[0]), float(w[1]), float(w[2])) for w in d.get("pred_relevant_windows", [])
        ]
    shared = set(pred_by_qid) & set(gt_by_qid)
    pred_by_qid = {q: pred_by_qid[q] for q in shared}
    gt_shared = {q: gt_by_qid[q] for q in shared}
    return {
        "brief": eval_mr_brief(pred_by_qid, gt_shared),
        **eval_moment_retrieval(pred_by_qid, gt_shared),
        **compute_miou(pred_by_qid, gt_shared),
        **compute_boundary_error(pred_by_qid, gt_shared),
    }
