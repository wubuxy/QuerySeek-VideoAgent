"""合成样本上的最小对照实验（P2 临时评测工具，P4 由 eval/runner.py 取代）。

在 scripts/make_synthetic_fixture.py 的 5 条标准查询上运行：
- A0  subtitle（BM25 字幕检索）
- E0  uniform + Mock VLM（预算 K ∈ {16, 32}）
- E1  semantic 召回 + VLM 粗观察（adaptive 的 rounds=0，预算 32）
- E2  E1 + Query-Adaptive 细化（rounds=2，预算 32）

指标：R@1（top-1 IoU >= 0.5 与 GT）、mean top-1 IoU、平均观察帧数/调用数/细化轮数。
**Mock 只验证流程与算法逻辑，不代表真实 VLM 准确率**（AGENTS.md 规则 5）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from queryseek.config import Config  # noqa: E402
from queryseek.embeddings.base import MockEmbedding  # noqa: E402
from queryseek.ingest.pipeline import ingest, load_index  # noqa: E402
from queryseek.models.mock import MockVLM  # noqa: E402
from queryseek.retrieval.adaptive import adaptive_search  # noqa: E402
from queryseek.retrieval.subtitle import subtitle_search  # noqa: E402
from queryseek.retrieval.uniform import uniform_search  # noqa: E402
from queryseek.schemas import TimeWindow  # noqa: E402
from queryseek.tools.validate import iou  # noqa: E402


def main() -> None:
    demo_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "demo"
    video = demo_dir / "sample.mp4"
    srt = demo_dir / "sample.srt"
    labels = json.loads((demo_dir / "sample.labels.json").read_text(encoding="utf-8"))

    cfg = Config.load().resolved()
    cfg.embedding.backend = "mock"  # 离线确定性 Mock 后端
    cfg.retrieval.semantic.min_sim = 0.3  # 过滤特征哈希单桶碰撞（~0.18）
    ingest(video, video_id="sample", srt_path=srt, config=cfg)
    idx = load_index("sample", cfg)
    vlm = MockVLM(idx.content_hints)
    embedder = MockEmbedding(idx.content_hints, idx.frames)

    queries = labels["queries"]

    def run_method(name, budget, fn):
        hits = ious = frames = calls = rounds = 0
        for item in queries:
            q, gt_windows = item["query"], item["windows"]
            res = fn(q, budget)
            frames += len(res.used_frame_ids)
            calls += res.vlm_calls
            rounds += len(res.refinement_rounds)
            if gt_windows:
                gt = TimeWindow(start_sec=gt_windows[0][0], end_sec=gt_windows[0][1])
                top_iou = max((iou(w, gt) for w in res.windows), default=0.0)
                ious += top_iou
                hits += 1 if top_iou >= 0.5 else 0
            else:  # 负样本：正确行为 = 返回空
                hits += 1 if not res.windows else 0
                ious += 1.0 if not res.windows else 0.0
        n = len(queries)
        print(
            f"{name:<14} {budget:>4}  {hits}/{n} ({hits / n * 100:>5.1f}%)"
            f"   {ious / n:>6.3f}   {frames / n:>6.1f}   {calls / n:>4.1f}   {rounds / n:>4.1f}"
        )

    print(f"{'method':<14} {'K':>4}  {'R@1 IoU>=0.5':<14} {'mean IoU':>8} {'frames':>7} {'calls':>5} {'rounds':>6}")
    run_method("A0 subtitle", 0, lambda q, k: subtitle_search(
        idx.video_id, q, idx.subtitles, idx.asset.duration_sec, cfg))
    for k in (16, 32):
        run_method(f"E0 uniform", k, lambda q, k: uniform_search(
            idx.video_id, q, idx.frames, idx.asset.duration_sec, vlm, cfg, budget_frames=k))
    run_method("E1 semantic+VLM", 32, lambda q, k: adaptive_search(
        idx.video_id, q, idx, vlm, embedder, cfg, budget_frames=k, max_rounds=0))
    run_method("E2 adaptive", 32, lambda q, k: adaptive_search(
        idx.video_id, q, idx, vlm, embedder, cfg, budget_frames=k, max_rounds=2))


if __name__ == "__main__":
    main()
