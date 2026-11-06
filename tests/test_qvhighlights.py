"""QVHighlights 评测适配测试。

关键测试：本实现与官方 standalone_eval 的样本指标文件逐项对照（官方
sample_val_preds_metrics_raw.json 由官方代码生成），差异 <= 0.02 视为一致。
官方仓库文件不复制进本仓库（标注 CC BY-NC-SA），测试从环境变量
QS_MOMENT_DETR_ROOT（或常见临时目录）读取，缺失则跳过。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from queryseek.eval.qvhighlights import (
    IOU_THDS,
    compute_ap,
    compute_mr_ap,
    compute_mr_r1,
    eval_mr_brief,
    evaluate_submission,
    load_qv_gt,
    load_submission,
    temporal_iou,
)
from queryseek.schemas import TimeWindow

_SAMPLE_ROOT = Path(
    os.environ.get(
        "QS_MOMENT_DETR_ROOT",
        str(Path(os.environ.get("TEMP", "/tmp")) / "qs_paper_review" / "moment_detr"),
    )
)


class TestPrimitives:
    def test_temporal_iou(self):
        assert temporal_iou(0, 10, 5, 15) == pytest.approx(1 / 3)
        assert temporal_iou(0, 10, 10, 20) == 0.0
        assert temporal_iou(0, 10, 0, 10) == 1.0

    def test_ap_perfect(self):
        ap = compute_ap([(0.0, 10.0)], [(0.0, 10.0, 0.9)])
        assert ap[0] == pytest.approx(1.0)  # IoU=1 在所有阈值上都是 TP

    def test_ap_empty_preds(self):
        ap = compute_ap([(0.0, 10.0)], [])
        assert (ap == 0).all()


@pytest.fixture(scope="module")
def official_sample():
    """官方样本：sample_val_preds.jsonl + GT + 官方生成的指标文件。"""
    pred_path = _SAMPLE_ROOT / "standalone_eval" / "sample_val_preds.jsonl"
    gt_path = _SAMPLE_ROOT / "data" / "highlight_val_release.jsonl"
    metrics_path = _SAMPLE_ROOT / "standalone_eval" / "sample_val_preds_metrics_raw.json"
    if not (pred_path.is_file() and gt_path.is_file() and metrics_path.is_file()):
        pytest.skip(f"官方样本文件不可用（{_SAMPLE_ROOT}），跳过对照测试")
    submission = list(load_submission(pred_path).values())
    gt = load_qv_gt(gt_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return submission, gt, metrics


class TestAgainstOfficial:
    def test_brief_metrics_match(self, official_sample):
        """本实现 brief 指标与官方生成值逐项一致（容差 0.02）。"""
        submission, gt, official = official_sample
        ours = eval_mr_brief(
            {d["qid"]: [(w[0], w[1], w[2]) for w in d["pred_relevant_windows"]] for d in submission},
            gt,
        )
        for key, official_val in official["brief"].items():
            if not key.startswith("MR-") or "HL" in key:
                continue  # HL 需要显著性预测，本项目不做
            assert key in ours, f"缺少指标 {key}"
            assert abs(ours[key] - official_val) <= 0.02, f"{key}: ours={ours[key]} official={official_val}"

    def test_full_mrap_curve_matches(self, official_sample):
        submission, gt, official = official_sample
        pred_by_qid = {
            d["qid"]: [(w[0], w[1], w[2]) for w in d["pred_relevant_windows"]] for d in submission
        }
        ours = compute_mr_ap(pred_by_qid, gt)
        official_full = official["full"]["MR-mAP"]
        for key, official_val in official_full.items():
            assert abs(ours[key] - official_val) <= 0.02, f"mAP@{key}: {ours[key]} vs {official_val}"

    def test_full_r1_curve_matches(self, official_sample):
        submission, gt, official = official_sample
        pred_by_qid = {
            d["qid"]: [(w[0], w[1], w[2]) for w in d["pred_relevant_windows"]] for d in submission
        }
        ours = compute_mr_r1(pred_by_qid, gt)
        official_full = official["full"]["MR-R1"]
        for key, official_val in official_full.items():
            assert abs(ours[key] - official_val) <= 0.02, f"R1@{key}: {ours[key]} vs {official_val}"

    def test_evaluate_submission_end_to_end(self, official_sample):
        submission, gt, official = official_sample
        out = evaluate_submission(submission, gt)
        assert out["brief"]["MR-full-mAP"] == pytest.approx(official["brief"]["MR-full-mAP"], abs=0.02)
        assert "custom_miou" in out and "custom_boundary_error" in out  # 自定义指标显式在场


class TestSchemaAdapters:
    def test_submission_roundtrip(self, tmp_path):
        from queryseek.eval.qvhighlights import save_submission, submission_from_windows

        subs = [
            submission_from_windows(1, "查询一", "vid1", [TimeWindow(start_sec=1.0, end_sec=5.0, score=0.8)]),
            submission_from_windows(2, "查询二", "vid2", []),  # 零预测也必须合法
        ]
        p = tmp_path / "sub.jsonl"
        save_submission(subs, p)
        loaded = load_submission(p)
        assert loaded[1]["pred_relevant_windows"] == [[1.0, 5.0, 0.8]]
        assert loaded[2]["pred_relevant_windows"] == []

    def test_iou_thds_official_values(self):
        assert len(IOU_THDS) == 10
        assert IOU_THDS[0] == 0.5 and IOU_THDS[-1] == 0.95
