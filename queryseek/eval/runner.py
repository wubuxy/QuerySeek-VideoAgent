"""公平对照评测 runner（开发说明书 §9）：E0-E3 + A0，同视频/同查询/同预算。

方法定义：
- A0 subtitle（BM25，零视觉预算）
- E0 uniform + VLM（预算 K 帧均匀观察）
- E1 semantic 召回 + VLM 粗观察（adaptive rounds=0）
- E2 E1 + Query-Adaptive 细化（adaptive rounds=2）
- E3 E2 + Agent Verifier（LangGraph 图）

指标：官方协议（MR mAP/R1，经 tests/test_qvhighlights.py 对照官方样本验证）
+ 自定义（custom mIoU / 边界误差，显式标记）+ 效率（帧数/调用/延迟）+ 可靠性
（no_result 率、越界数）。

输出目录 outputs/eval/<run_id>/：config_snapshot.yaml / predictions.jsonl /
metrics.json / per_sample.csv / errors.csv / summary.md（说明书 §9.3）。
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..agent.graph import run_agent
from ..agent.state import default_budget, initial_state
from ..config import Config
from ..embeddings.base import MockEmbedding
from ..ingest.pipeline import VideoIndex, ingest, load_index
from ..models.mock import MockVLM
from ..retrieval.adaptive import adaptive_search
from ..retrieval.subtitle import subtitle_search
from ..retrieval.uniform import uniform_search
from ..schemas import SearchResult, TimeWindow
from ..tools.export import new_job_id
from ..tools.validate import iou
from .qvhighlights import compute_boundary_error, compute_miou, eval_mr_brief, save_submission, submission_from_windows

METHODS = ("A0", "E0", "E1", "E2", "E3")


@dataclass
class EvalSample:
    qid: int
    video_path: str
    video_id: str
    srt_path: str | None
    query: str
    gt_windows: list[TimeWindow]
    query_type: str = "mixed"
    duration_sec: float | None = None


def load_dataset(path: str | Path) -> list[EvalSample]:
    """评测集 JSON：{samples: [{video, video_id, srt?, query, windows: [[s,e],...], type?}]}。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for i, s in enumerate(raw.get("samples", [])):
        out.append(
            EvalSample(
                qid=int(s.get("qid", i)),
                video_path=s["video"],
                video_id=s.get("video_id") or Path(s["video"]).stem,
                srt_path=s.get("srt"),
                query=s["query"],
                gt_windows=[TimeWindow(start_sec=w[0], end_sec=w[1]) for w in s.get("windows", [])],
                query_type=s.get("type", "mixed"),
                duration_sec=float(s["duration"]) if s.get("duration") else None,
            )
        )
    return out


def dataset_from_labels(labels_path: str | Path, video_path: str | Path, srt_path: str | Path | None) -> list[EvalSample]:
    """从合成样本的 labels.json 构造评测集（queries 字段）。"""
    raw = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    out = []
    for i, q in enumerate(raw.get("queries", [])):
        out.append(
            EvalSample(
                qid=i,
                video_path=str(video_path),
                video_id=Path(video_path).stem,
                srt_path=str(srt_path) if srt_path else None,
                query=q["query"],
                gt_windows=[TimeWindow(start_sec=w[0], end_sec=w[1]) for w in q.get("windows", [])],
                query_type=q.get("type", "mixed"),
            )
        )
    return out


@dataclass
class MethodResult:
    qid: int
    query: str
    method: str
    budget: int
    windows: list[TimeWindow] = field(default_factory=list)
    used_frames: int = 0
    vlm_calls: int = 0
    elapsed_ms: float = 0.0
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    indexed_frames: int = 0
    online_encoded_frames: int = 0
    embedding_cache_hit: bool | None = None


def _real_embedder(cfg: Config, index: VideoIndex):
    """真实 OpenCLIP 后端：加载 ingest 缓存的帧嵌入（若可用）。"""
    from ..retrieval.dispatch import embedder_for

    return embedder_for(index, cfg, False)


def _run_method(
    cfg: Config,
    index: VideoIndex,
    sample: EvalSample,
    method: str,
    budget: int,
    use_mock: bool,
) -> MethodResult:
    """单样本单方法执行；Mock 后端为确定性实现。"""
    t0 = time.perf_counter()
    q = sample.query
    duration = index.asset.duration_sec
    vlm = MockVLM(index.content_hints) if use_mock else None
    if vlm is None and method not in ("E4", "E4S"):  # pragma: no cover
        from ..models.factory import build_vlm

        vlm = build_vlm(cfg, use_mock=False, hints=index.content_hints)

    res: SearchResult
    if method == "A0":
        res = subtitle_search(index.video_id, q, index.subtitles, duration, cfg)
    elif method in ("E4", "E4S", "E4V", "E5"):
        from ..retrieval.curve_spans import curve_search
        embedder, mat = _real_embedder(cfg, index) if not use_mock else (
            MockEmbedding(index.content_hints, index.frames), None
        )
        res = curve_search(index.video_id, q, index, embedder, cfg, frame_embeddings=mat,
                           include_short=method in ("E4S", "E5"))
        if method in ("E4V", "E5"):
            from ..retrieval.visual_spans import verify_curve_result
            res = verify_curve_result(res, index, vlm, Path("outputs/visual_spans"),
                                      budget_frames=budget)
    elif method == "E0":
        res = uniform_search(index.video_id, q, index.frames, duration, vlm, cfg,
                             budget_frames=budget, scenes=index.scenes)
    elif method in ("E1", "E2"):
        embedder, mat = _real_embedder(cfg, index) if not use_mock else (
            MockEmbedding(index.content_hints, index.frames), None
        )
        rounds = 0 if method == "E1" else cfg.retrieval.max_refinement_rounds
        res = adaptive_search(index.video_id, q, index, vlm, embedder, cfg,
                              budget_frames=budget, max_rounds=rounds,
                              frame_embeddings=mat)
    elif method == "E3":
        budget_map = default_budget({
            "max_iterations": cfg.agent.max_iterations,
            "max_vlm_calls": cfg.agent.max_vlm_calls,
            "max_frames": budget,
            "wall_time_limit_sec": cfg.agent.wall_time_limit_sec,
        })
        embedder, mat = _real_embedder(cfg, index) if not use_mock else (
            MockEmbedding(index.content_hints, index.frames), None
        )
        state = initial_state(index.video_id, q, strategy="adaptive", budget=budget_map)
        final = run_agent(
            state,
            {"thread_id": f"eval_{sample.qid}_{method}",
             "cfg": cfg.model_dump(), "mock": use_mock,
             "vlm": vlm, "embedder": embedder, "frame_embeddings": mat},
        )
        return MethodResult(
            qid=sample.qid, query=q, method=method, budget=budget,
            windows=[TimeWindow(start_sec=w["start_sec"], end_sec=w["end_sec"], score=w.get("score"))
                     for w in final.get("selected_windows", [])],
            used_frames=len(final.get("used_frame_ids", [])),
            vlm_calls=int(final.get("vlm_calls", 0)),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            status=final.get("status", "failed"),
            warnings=list(final.get("warnings", [])),
        )
    else:
        raise ValueError(f"未知方法 {method}")

    return MethodResult(
        qid=sample.qid, query=q, method=method, budget=budget,
        windows=list(res.windows), used_frames=len(res.used_frame_ids),
        vlm_calls=res.vlm_calls, elapsed_ms=res.elapsed_ms,
        status="no_result" if not res.windows else "ok",
        warnings=list(res.warnings),
        indexed_frames=res.indexed_frames,
        online_encoded_frames=res.online_encoded_frames,
        embedding_cache_hit=res.embedding_cache_hit,
    )


def _method_metrics(
    results: list[MethodResult],
    gt_by_qid: dict[int, list[TimeWindow]],
    expected_n: int,
) -> dict[str, Any]:
    """单方法聚合指标：官方 MR brief + custom + 效率 + 可靠性。

    公平性守卫：结果数 != 期望样本数时显式标记 coverage（说明书 §9.1——
    不允许某方法因部分样本失败而按更少样本计算指标、静默占便宜）。
    """
    pred_by_qid: dict[int, list[tuple[float, float, float]]] = {}
    r1_iou05, mean_iou, n_noresult, n_invalid = 0, 0.0, 0, 0
    frames = calls = latency = indexed_frames = online_encoded_frames = 0
    n_positive = n_negative = negative_rejected = positive_no_result = 0
    n = len(results)
    for r in results:
        gt = gt_by_qid.get(r.qid, [])
        pred_by_qid[r.qid] = [
            (w.start_sec, w.end_sec, w.score if w.score is not None else 0.0) for w in r.windows
        ]
        # 拒答查询也消耗计算，必须在检查空结果之前累计成本。
        frames += r.used_frames
        calls += r.vlm_calls
        latency += r.elapsed_ms
        indexed_frames += r.indexed_frames
        online_encoded_frames += r.online_encoded_frames
        if gt:
            n_positive += 1
            positive_no_result += int(not r.windows)
        else:
            n_negative += 1
            negative_rejected += int(not r.windows)
        if not r.windows:
            n_noresult += 1
            continue
        top = r.windows[0]
        if gt:
            best = max(iou(top, g) for g in gt)
            r1_iou05 += 1 if best >= 0.5 else 0
            mean_iou += best
    if n == 0:
        return {"coverage": 0.0, "incomplete": True}
    gt_dict = {qid: {"relevant_windows": [[w.start_sec, w.end_sec] for w in ws]}
               for qid, ws in gt_by_qid.items()}
    m = {
        "coverage": n / expected_n if expected_n else 0.0,
        "incomplete": n != expected_n,
        "R1@IoU0.5 (top-1)": r1_iou05 / n if n else 0.0,
        "custom_mIoU": mean_iou / max(n - n_noresult, 1),
        "no_result_rate": n_noresult / n,
        "avg_observed_frames": frames / n,
        "avg_vlm_calls": calls / n,
        "avg_latency_ms": latency / n,
        "avg_indexed_frames": indexed_frames / n,
        "avg_online_encoded_frames": online_encoded_frames / n,
        "positive_query_count": n_positive,
        "negative_query_count": n_negative,
        "negative_rejection_rate": negative_rejected / n_negative if n_negative else None,
        "negative_false_positive_rate": (n_negative-negative_rejected) / n_negative if n_negative else None,
        "positive_no_result_rate": positive_no_result / n_positive if n_positive else None,
        "official_brief": dict(eval_mr_brief(pred_by_qid, gt_dict)),
        **compute_miou(pred_by_qid, gt_dict),
        **compute_boundary_error(pred_by_qid, gt_dict),
    }
    return m


def run_benchmark(
    cfg: Config,
    samples: list[EvalSample],
    methods: list[str],
    budgets: list[int],
    out_dir: Path,
    use_mock: bool = True,
) -> dict[str, Any]:
    """执行评测并落盘全部产物；返回 metrics 汇总。"""
    run_id = new_job_id()
    out_dir = out_dir / "eval" / run_id
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)

    # 1. ingest 每个样本（同视频去重）
    indexes: dict[str, VideoIndex] = {}
    ingest_log = []
    for s in samples:
        if s.video_id not in indexes:
            idx = ingest(s.video_path, video_id=s.video_id, srt_path=s.srt_path, config=cfg)
            indexes[s.video_id] = idx
            ingest_log.append({"video_id": s.video_id, "cache_hit": idx.cache_hit,
                               "duration_sec": idx.asset.duration_sec})

    gt_by_qid = {s.qid: s.gt_windows for s in samples}
    all_results: list[MethodResult] = []
    errors: list[dict[str, Any]] = []

    for s in samples:
        index = indexes[s.video_id]
        for method in methods:
            for budget in budgets:
                if method == "A0" and budget != budgets[0]:
                    continue  # A0 无视觉预算，只跑一次
                try:
                    all_results.append(_run_method(cfg, index, s, method, budget, use_mock))
                except Exception as exc:  # noqa: BLE001 — 单样本失败不中断评测
                    errors.append({"qid": s.qid, "method": method, "budget": budget,
                                   "error": str(exc)})

    # 2. 指标
    metrics: dict[str, Any] = {"run_id": run_id, "n_samples": len(samples),
                               "methods": methods, "budgets": budgets, "per_method": {}}
    for method in methods:
        for budget in budgets:
            sub = [r for r in all_results if r.method == method and r.budget == budget]
            if not sub:
                continue
            metrics["per_method"][f"{method}-K{budget}"] = _method_metrics(
                sub, gt_by_qid, expected_n=len(samples)
            )

    # 3. 落盘
    (out_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    per_method_preds: dict[str, list[dict]] = {}
    for r in all_results:
        key = f"{r.method}-K{r.budget}"
        per_method_preds.setdefault(key, []).append(
            submission_from_windows(r.qid, r.query, str(r.qid), r.windows)
        )
    for key, subs in per_method_preds.items():
        save_submission(subs, out_dir / "predictions" / f"{key}.jsonl")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query", "method", "budget", "status", "n_windows",
                    "top1_start", "top1_end", "top1_iou", "used_frames", "vlm_calls", "elapsed_ms",
                    "indexed_frames", "online_encoded_frames", "embedding_cache_hit"])
        for r in all_results:
            top_iou = ""
            if r.windows and gt_by_qid.get(r.qid):
                top_iou = f"{max(iou(r.windows[0], g) for g in gt_by_qid[r.qid]):.3f}"
            w.writerow([r.qid, r.query[:40], r.method, r.budget, r.status, len(r.windows),
                        f"{r.windows[0].start_sec:.2f}" if r.windows else "",
                        f"{r.windows[0].end_sec:.2f}" if r.windows else "",
                        top_iou, r.used_frames, r.vlm_calls, f"{r.elapsed_ms:.1f}",
                        r.indexed_frames, r.online_encoded_frames, r.embedding_cache_hit])
    with (out_dir / "errors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "method", "budget", "error"])
        for e in errors:
            w.writerow([e["qid"], e["method"], e["budget"], e["error"]])

    _write_summary(out_dir, metrics, all_results, gt_by_qid, ingest_log)
    return metrics


def _write_summary(out_dir: Path, metrics: dict, results: list[MethodResult],
                   gt_by_qid: dict[int, list[TimeWindow]], ingest_log: list[dict]) -> None:
    lines = [
        "# 评测摘要",
        "",
        f"- run_id: {metrics['run_id']}",
        f"- 样本数: {metrics['n_samples']}（ingest 缓存命中 {sum(1 for i in ingest_log if i['cache_hit'])}/{len(ingest_log)}）",
        "",
        "## 各方法指标",
        "",
        "| 方法 | R@1 IoU>=0.5 | custom mIoU | 平均帧数 | 平均调用 | 平均延迟(ms) | no_result 率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, m in metrics["per_method"].items():
        lines.append(
            f"| {key} | {m['R1@IoU0.5 (top-1)']:.3f} | {m['custom_mIoU']:.3f} | "
            f"{m['avg_observed_frames']:.1f} | {m['avg_vlm_calls']:.1f} | "
            f"{m['avg_latency_ms']:.0f} | {m['no_result_rate']:.2f} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- 以上指标由 scripts/benchmark.py 生成；**Mock 组件结果只验证流程，不代表真实 VLM 准确率**（AGENTS.md 规则 5）。",
        "- custom mIoU / 边界误差为自定义指标；官方 MR mAP/R1 见 metrics.json 的 official_brief（实现经官方样本文件对照验证）。",
        "- E4/E4S 的 used_frames 仅统计 VLM 观察；全视频 CLIP 成本见 indexed_frames、online_encoded_frames。",
        "- 负查询拒答率/误报率与正查询空结果率单独见 metrics.json；没有该类样本时为 null，不视为已验证。",
        "- 每次运行的数据、方法、预算与后端以 config_snapshot.yaml 和预测产物为准。",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

