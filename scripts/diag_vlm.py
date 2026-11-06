"""VLM 网络诊断（P1/P2，DECISIONS #41）：最小化复现调用 + 请求级观测。

用法：
    python scripts/diag_vlm.py --mode text     # 纯文本调用
    python scripts/diag_vlm.py --mode single   # 单图调用
    python scripts/diag_vlm.py --mode multi    # 原始 E2 多帧调用（样本 0 真实帧+查询）
    python scripts/diag_vlm.py --mode multi --frames 4

诊断配置：max_retries=0（禁用自动重试）、分项超时 connect=8/read=45/write=20/
pool=5、trust_env=False（直连）、http2=False、max_keepalive_connections=0、
max_connections=1。完整记录（含 httpcore 网络事件）写
outputs/vlm_reqlog/requests.jsonl，控制台打印最近一条。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from queryseek.env import ensure_dotenv
from queryseek.models.compatible_api import CompatibleAPIVLM
from queryseek.schemas import VideoFrame

REQ_LOG = ROOT / "outputs" / "vlm_reqlog"

# 样本 0 的 E2 refine 帧（评测挂死时的真实输入，data/cache/<sha256>/refine/）
_SAMPLE0_CACHE = (
    ROOT / "data" / "cache"
    / "7bd0b4d839ac3c43edb3845702d671fc40ff456aa3e3b9224d57ff15c60af0c0"
)
_SAMPLE0_QUERY = "The worker lifts the crate further into the truck."


def _build_vlm() -> CompatibleAPIVLM:
    ensure_dotenv()
    return CompatibleAPIVLM(
        api_key=os.environ["VLM_API_KEY"],
        base_url=os.environ["VLM_BASE_URL"],
        model=os.environ["VLM_MODEL"],
        max_retries=0,                      # P1：禁用自动重试
        max_output_tokens=64,
        trust_env=False,                    # 直连，不走系统代理
        connect_timeout_sec=8,
        read_timeout_sec=45,
        write_timeout_sec=20,
        pool_timeout_sec=5,
        http2=False,
        max_keepalive_connections=0,
        max_connections=1,
        req_log_dir=str(REQ_LOG),
    )


def _refine_frames(n: int) -> list[VideoFrame]:
    refine = _SAMPLE0_CACHE / "refine"
    paths = sorted(refine.glob("r0_w0_*.jpg"))[:n]
    if len(paths) < n:
        sys.exit(f"refine 帧不足：{len(paths)} < {n}（{refine}）")
    return [VideoFrame(frame_id=f"frame_{i + 1:06d}", timestamp_sec=100.0 + i,
                       image_path=str(p)) for i, p in enumerate(paths)]


def main() -> None:
    ap = argparse.ArgumentParser(description="VLM 网络诊断（P1/P2）")
    ap.add_argument("--mode", choices=("text", "single", "multi"), default="text")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    vlm = _build_vlm()
    t0 = time.time()
    try:
        if args.mode == "text":
            vlm._chat([{"type": "text", "text": "只回复一个词：OK"}])
            print("text: OK")
        elif args.mode == "single":
            res = vlm.inspect_frames(_SAMPLE0_QUERY, _refine_frames(1))
            print("single: OK", res.relevant_frame_ids)
        else:
            res = vlm.inspect_frames(_SAMPLE0_QUERY, _refine_frames(args.frames))
            print("multi: OK", res.relevant_frame_ids)
    except Exception as exc:  # noqa: BLE001 — 诊断脚本：任何异常都记录后报告
        print(f"{args.mode}: FAIL {type(exc).__name__} in {time.time() - t0:.1f}s: {str(exc)[:200]}")
    time.sleep(0.2)
    log_file = REQ_LOG / "requests.jsonl"
    if log_file.is_file():
        lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            rec = lines[-1]
            print("\n--- 最近一条请求记录 ---")
            print(json.dumps({
                "request_id": rec["request_id"], "model": rec["model"],
                "n_images": rec["n_images"], "body_bytes": rec["body_bytes"],
                "timeout": rec["timeout"], "duration_ms": rec["duration_ms"],
                "status": rec["status"], "exception": rec["exception"],
            }, ensure_ascii=False, indent=2))
            print("trace events:", " | ".join(e["event"] for e in rec["trace"]))


if __name__ == "__main__":
    main()
