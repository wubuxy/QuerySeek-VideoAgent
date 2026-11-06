"""生成 Qwen-VL 微调数据集（设计见 docs/FINE_TUNING.md）。

数据流：train GT → 抽样（seed，holdout 分离）→ 下载视频 → ffprobe 时长
→ 帧计划（正/困难负/简单负/inspect）→ ffmpeg 抽帧 → 拼接图 → ChatML 记录
→ 分片写出（data.jsonl + 裸名图片）→ 准入校验 + 可选 zip。

用法：
    # Mock 模式（无网络/无 ffmpeg）：验证全管线（AGENTS.md 规则 5）
    python scripts/make_ft_dataset.py --mock --n-queries 4 --seed 42
    # 真实模式：下载训练视频、抽帧、生成记录（需网络与 FFmpeg）
    python scripts/make_ft_dataset.py --n-queries 800 --holdout 200 --seed 42 --zip

产出 data/ft_qwenvl/：holdout.jsonl / manifest.json /
shard_0000/{data.jsonl, *.jpg}[.zip] / work/frames/（中间帧，可删）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from queryseek.ft.montage import build_montage
from queryseek.ft.records import (
    ShardWriter,
    build_inspect_record,
    build_verify_record,
    make_shard_zips,
)
from queryseek.ft.sampling import load_train_entries, plan_frames, sample_queries
from queryseek.ft.validator import validate_shard

DEFAULT_GT = (
    Path(os.environ.get(
        "QS_MOMENT_DETR_ROOT",
        str(Path(os.environ.get("TEMP", "/tmp")) / "qs_paper_review" / "moment_detr"),
    )) / "data" / "highlight_train_release.jsonl"
)

_VERIFY_REASON = {
    "positive": "画面内容与查询直接相关",
    "hard_negative": "画面内容与查询无关",
    "easy_negative": "画面内容与查询无关",
}


def _mock_frame_bytes(vid: str, t: float, side: int = 512) -> bytes:
    """确定性合成帧（纯色 + 时间戳文字），Mock 模式专用（AGENTS.md 规则 5）。"""
    seed = int(hashlib.sha256(f"{vid}@{t:.2f}".encode()).hexdigest()[:8], 16)
    color = np.random.default_rng(seed).integers(0, 255, size=3).tolist()
    img = np.full((side, side, 3), color, dtype=np.uint8)
    cv2.putText(img, f"t={t:.2f}", (side // 4, side // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _duration_sec(video: Path) -> float:
    from queryseek.ingest.ffprobe import probe

    info = probe(video)
    if info.get("format", {}).get("duration"):
        return float(info["format"]["duration"])
    for s in info.get("streams", []):
        if s.get("duration"):
            return float(s["duration"])
    raise RuntimeError(f"ffprobe 无时长: {video}")


def _download(vid: str, videos_dir: Path) -> Path | None:
    """优先 HF 镜像，失败回退 YouTube（与 make_qvhighlights_subset 同策略）。"""
    import make_qvhighlights_subset as mqs

    hf = mqs._download_hf(vid, videos_dir)
    if hf is not None:
        return hf
    m = mqs._VID_RE.match(vid)
    if m and mqs._available(vid, m.group(1)) and mqs._download_yt(
            vid, m.group(1), float(m.group(2)), float(m.group(3)), videos_dir):
        cands = list(videos_dir.glob(f"{vid}.*"))
        return next((c for c in cands if c.suffix == ".mp4"), cands[0] if cands else None)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 Qwen-VL 微调数据集")
    ap.add_argument("--gt", type=str, default=str(DEFAULT_GT),
                    help="train GT jsonl（moment_detr highlight_train_release.jsonl）")
    ap.add_argument("--n-queries", type=int, default=800, help="训练查询数")
    ap.add_argument("--holdout", type=int, default=200, help="预留评测查询数（不生成记录）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="data/ft_qwenvl")
    ap.add_argument("--frames-per-record", type=int, default=4)
    ap.add_argument("--records-per-shard", type=int, default=3000)
    ap.add_argument("--tasks", type=str, default="verify,inspect")
    ap.add_argument("--mock", action="store_true",
                    help="确定性合成帧，无网络/无 ffmpeg（只验证流程）")
    ap.add_argument("--zip", action="store_true", help="校验后打包各 shard 为 zip")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    frames_root = out_dir / "work" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    rng = random.Random(args.seed)

    entries = load_train_entries(args.gt)
    train, holdout = sample_queries(entries, args.seed,
                                    args.n_queries, args.holdout)
    (out_dir / "holdout.jsonl").write_text(
        "".join(json.dumps({"vid": s.vid, "query": s.query,
                            "windows": [[round(a, 2), round(b, 2)]
                                        for a, b in s.windows]},
                           ensure_ascii=False) + "\n" for s in holdout),
        encoding="utf-8",
    )
    print(f"抽样：train={len(train)}（请求 {args.n_queries}） holdout={len(holdout)}",
          flush=True)

    writer = ShardWriter(out_dir, args.records_per_shard)
    counts: dict[str, int] = {}
    skipped: list[dict] = []
    for i, sample in enumerate(train, 1):
        frames_dir = frames_root / sample.vid.replace(".", "_")
        video: Path | None = None
        if not args.mock:
            video = videos_dir / f"{sample.vid}.mp4"
            if not video.is_file():
                video = _download(sample.vid, videos_dir)
            if video is None:
                skipped.append({"vid": sample.vid, "reason": "视频下载失败"})
                continue
            try:
                sample.duration_sec = _duration_sec(video)
            except RuntimeError as exc:
                skipped.append({"vid": sample.vid, "reason": f"ffprobe 失败: {exc}"})
                continue
        else:
            sample.duration_sec = sample.sorted_windows[-1][1] + 60.0

        for plan in plan_frames(sample, rng, args.frames_per_record, tasks):
            if args.mock:
                frames_dir.mkdir(parents=True, exist_ok=True)
                for t in plan.timestamps:
                    p = frames_dir / f"t{int(round(t, 2) * 1000):08d}.jpg"
                    if not p.is_file():
                        p.write_bytes(_mock_frame_bytes(sample.vid, t))
                paths = [frames_dir / f"t{int(round(t, 2) * 1000):08d}.jpg"
                         for t in plan.timestamps]
            else:
                from queryseek.ft.frames import extract_frames_at

                paths, _failed = extract_frames_at(video, plan.timestamps, frames_dir)
                if len(paths) < len(plan.timestamps):
                    skipped.append({"vid": sample.vid, "reason": "抽帧不完整"})
                    continue
            cols = 2 if len(paths) <= 4 else 4
            image_bytes = build_montage(paths, cols=cols)
            name = writer.reserve_name(sample.vid)
            if plan.task == "verify":
                verdict = "confirmed" if plan.label == "positive" else "refuted"
                confidence = 0.9 if plan.label != "easy_negative" else 0.95
                rec = build_verify_record(sample.query, plan.window,
                                          plan.timestamps, verdict, confidence,
                                          name, _VERIFY_REASON[plan.label])
            else:
                rec = build_inspect_record(sample.query, plan.timestamps,
                                           plan.relevant_indexes, name)
            writer.add(rec, image_bytes)
            key = f"{plan.task}/{plan.label}"
            counts[key] = counts.get(key, 0) + 1
        print(f"[{i}/{len(train)}] {sample.vid} 记录 {sum(counts.values())} 条",
              flush=True)

    writer.flush()
    manifest = {
        "name": "qvhighlights-train-ft-subset",
        "source": "moment_detr highlight_train_release.jsonl（标注 CC BY-NC-SA 4.0，仅训练用途）",
        "seed": args.seed,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "frames_per_record": args.frames_per_record,
        "tasks": list(tasks),
        "mock": args.mock,
        "counts": counts,
        "skipped": skipped,
        "shards": [str(d) for d in writer.shard_dirs],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n记录统计: {json.dumps(counts, ensure_ascii=False)}", flush=True)
    for d in writer.shard_dirs:
        report = validate_shard(d)
        status = "通过" if report["ok"] else f"未通过：{report['errors'][:5]}"
        print(f"校验 {d.name}: 记录 {report['records']} 图片 {report['images']} "
              f"共 {report['total_mb']}MB → {status}", flush=True)
    if args.zip:
        for z in make_shard_zips(out_dir):
            print(f"打包: {z}", flush=True)
    print(f"跳过 {len(skipped)} 条（原因见 manifest.json）", flush=True)


if __name__ == "__main__":
    main()
