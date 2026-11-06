"""公平对照实验 CLI（说明书 §9）：E0-E3 + A0，同视频/同查询/同预算。

用法：
    python scripts/benchmark.py --methods A0,E0,E1,E2,E3 --budgets 16,32 --mock
    python scripts/benchmark.py --dataset data/mydataset.json --methods E0,E2 --budgets 32 --mock

默认数据集：assets/demo/sample.labels.json（合成样本，Mock 组件，确定性）。
产出 outputs/eval/<run_id>/：config_snapshot.yaml / predictions/ / metrics.json /
per_sample.csv / errors.csv / summary.md（说明书 §9.3）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from queryseek.config import Config  # noqa: E402
from queryseek.env import ensure_dotenv  # noqa: E402
from queryseek.eval.runner import (  # noqa: E402
    dataset_from_labels,
    load_dataset,
    run_benchmark,
)

DEMO_DIR = ROOT / "assets" / "demo"


def main() -> None:
    ap = argparse.ArgumentParser(description="QuerySeek-VideoAgent 公平对照实验")
    ap.add_argument("--dataset", type=str, default=None,
                    help="评测集 JSON；缺省用 assets/demo/sample.labels.json（合成样本）")
    ap.add_argument("--video", type=str, default=str(DEMO_DIR / "sample.mp4"))
    ap.add_argument("--srt", type=str, default=str(DEMO_DIR / "sample.srt"))
    ap.add_argument("--labels", type=str, default=str(DEMO_DIR / "sample.labels.json"))
    ap.add_argument("--methods", type=str, default="A0,E0,E1,E2,E3")
    ap.add_argument("--budgets", type=str, default="16,32")
    ap.add_argument("--mock", action="store_true", default=False,
                    help="使用确定性 Mock 组件（无密钥/离线）")
    ap.add_argument("--config", type=str, default=None,
                    help="自定义配置文件（如 configs/qvhighlights.yaml）")
    ap.add_argument("--out-root", type=str, default="outputs")
    args = ap.parse_args()

    ensure_dotenv()
    cfg = Config.load(Path(args.config) if args.config else None).resolved()
    if cfg.embedding.backend != "mock" and args.mock:
        cfg.embedding.backend = "mock"
    if args.mock:
        cfg.retrieval.semantic.min_sim = 0.3  # Mock 特征哈希的干净分离阈值

    if args.dataset:
        samples = load_dataset(args.dataset)
    else:
        samples = dataset_from_labels(args.labels, args.video, args.srt)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    print(f"评测集: {len(samples)} 样本 × 方法 {methods} × 预算 {budgets} "
          f"（mock={args.mock}）")

    metrics = run_benchmark(
        cfg, samples, methods, budgets,
        Path(args.out_root), use_mock=args.mock,
    )
    print(f"run_id: {metrics['run_id']}")
    print(f"{'方法':<12} {'R@1@0.5':>8} {'mIoU':>6} {'帧':>5} {'调用':>5} {'延迟ms':>7} {'no_res':>7}")
    for key, m in metrics["per_method"].items():
        print(f"{key:<12} {m['R1@IoU0.5 (top-1)']:>8.3f} {m['custom_mIoU']:>6.3f} "
              f"{m['avg_observed_frames']:>5.1f} {m['avg_vlm_calls']:>5.1f} "
              f"{m['avg_latency_ms']:>7.1f} {m['no_result_rate']:>7.2f}")
    print(f"输出: {cfg.output_root}/eval/{metrics['run_id']}/")


if __name__ == "__main__":
    main()

