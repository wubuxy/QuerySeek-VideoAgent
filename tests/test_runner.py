"""评测 runner 测试：合成集端到端 + 产出结构（说明书 §9.3）。"""

from __future__ import annotations

import json

import pytest

from queryseek.config import Config
from queryseek.eval.runner import (
    dataset_from_labels,
    load_dataset,
    run_benchmark,
)
from queryseek.schemas import TimeWindow
from queryseek.tools.validate import iou


def _cfg() -> Config:
    cfg = Config.load().resolved()
    cfg.embedding.backend = "mock"
    cfg.retrieval.semantic.min_sim = 0.3
    return cfg


def _samples(demo):
    return dataset_from_labels(demo["labels"], demo["video"], demo["srt"])


def test_dataset_loading(demo):
    samples = _samples(demo)
    assert len(samples) == 5
    assert all(isinstance(w, TimeWindow) for w in samples[0].gt_windows)
    neg = [s for s in samples if not s.gt_windows]
    assert len(neg) == 1  # 负样本


def test_dataset_json_roundtrip(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"samples": [
        {"video": "v.mp4", "video_id": "v", "srt": "v.srt",
         "query": "q", "windows": [[1.0, 5.0]], "type": "mixed"},
    ]}, ensure_ascii=False), encoding="utf-8")
    samples = load_dataset(p)
    assert samples[0].query == "q"
    assert samples[0].gt_windows[0].start_sec == 1.0


def test_benchmark_outputs(qs_env, demo):
    """E0/E1 两种方法在合成集上跑通并落盘全部产物。"""
    cfg = _cfg()
    samples = _samples(demo)
    metrics = run_benchmark(cfg, samples, methods=["E0", "E1"], budgets=[16],
                            out_dir=cfg.output_root, use_mock=True)
    run_dir = cfg.output_root / "eval" / metrics["run_id"]
    for name in ("config_snapshot.yaml", "metrics.json", "per_sample.csv",
                 "errors.csv", "summary.md"):
        assert (run_dir / name).is_file(), f"缺少 {name}"
    preds = list((run_dir / "predictions").glob("*.jsonl"))
    assert {p.stem for p in preds} == {"E0-K16", "E1-K16"}
    # predictions 为官方格式：qid/query/vid/pred_relevant_windows
    first = json.loads(preds[0].read_text(encoding="utf-8").splitlines()[0])
    for key in ("qid", "query", "vid", "pred_relevant_windows"):
        assert key in first
    # 指标结构
    for key in ("E0-K16", "E1-K16"):
        m = metrics["per_method"][key]
        assert 0.0 <= m["R1@IoU0.5 (top-1)"] <= 1.0
        assert "official_brief" in m and "custom_miou" in m
    # E1 在合成集上帧数应明显少于 E0（语义召回替代均匀观察）
    assert metrics["per_method"]["E1-K16"]["avg_observed_frames"] < \
        metrics["per_method"]["E0-K16"]["avg_observed_frames"]


def test_benchmark_e3(qs_env, demo):
    """E3（Agent Verifier 路径）在合成集上跑通。"""
    cfg = _cfg()
    samples = _samples(demo)[:2]
    metrics = run_benchmark(cfg, samples, methods=["E3"], budgets=[32],
                            out_dir=cfg.output_root, use_mock=True)
    m = metrics["per_method"]["E3-K32"]
    assert m["R1@IoU0.5 (top-1)"] > 0.0
    assert m["avg_vlm_calls"] > 0  # verify 真实发生


def test_negative_sample_handled(qs_env, demo):
    """负样本（内容不存在）：正确行为是 no_result，不计入 top-1 IoU 命中。"""
    cfg = _cfg()
    samples = [s for s in _samples(demo) if not s.gt_windows]
    metrics = run_benchmark(cfg, samples, methods=["E0"], budgets=[16],
                            out_dir=cfg.output_root, use_mock=True)
    m = metrics["per_method"]["E0-K16"]
    assert m["no_result_rate"] == 1.0  # 全部 no_result
    assert m["R1@IoU0.5 (top-1)"] == 0.0
