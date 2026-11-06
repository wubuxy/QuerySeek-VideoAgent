"""构建 QVHighlights 评测子集（开发说明书 §8.2 的透明筛选流程）。

- 官方视频 tar（nlp.cs.unc.edu）实测 404；按官方 data/README 指引改用
  YouTube 分段下载（vid = {youtube_id}_{start}_{end}）。
- 抽样固定种子、优先去重 vid；逐条检查可用性（yt-dlp -s），不可用记录原因。
- 产出 data/qvhighlights/dataset.json（本项目 runner 格式）+ availability.json
  （可用率与筛选规则透明化）。
- 标注文件 CC BY-NC-SA 4.0：仅用于评测，不入库、不重分发；视频存入 data/
  （gitignore）。

用法：
    python scripts/make_qvhighlights_subset.py --n 20 --seed 42
    （需 QS_MOMENT_DETR_ROOT 指向 moment_detr 克隆，否则用默认临时目录）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_VID_RE = re.compile(r"(.+)_([\d.]+)_([\d.]+)$")

DEFAULT_GT = Path(os.environ.get(
    "QS_MOMENT_DETR_ROOT",
    str(Path(os.environ.get("TEMP", "/tmp")) / "qs_paper_review" / "moment_detr"),
)) / "data" / "highlight_val_release.jsonl"


def source_video_id(vid: str) -> str:
    """不同时间裁片可能来自同一个 YouTube 原视频，不能只按裁片去重。"""
    match = _VID_RE.fullmatch(vid)
    return match.group(1) if match else vid


def select_entries(entries, n, seed, excluded_vids=(), source_disjoint=False):
    if n <= 0:
        raise ValueError("n 必须为正整数")
    key = source_video_id if source_disjoint else lambda vid: vid
    excluded = {key(vid) for vid in excluded_vids}
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    seen = set()
    selected = []
    for entry in shuffled:
        identity = key(entry["vid"])
        if identity in seen or identity in excluded:
            continue
        seen.add(identity)
        selected.append(entry)
        if len(selected) == n:
            break
    return selected


def _available(vid: str, yid: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-s", "--print", "%(title)s",
         f"https://www.youtube.com/watch?v={yid}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
    )
    return r.returncode == 0


_HF_REPO = "zhaoshiwen/storage-large-public"
_HF_VIDEO_PREFIX = "FlashVTGvis/qvhighlight/videos/"
_HF_INDEX: dict[str, str] | None = None  # vid -> 仓库内相对路径（惰性构建）


def _hf_index() -> dict[str, str]:
    """镜像仓库按 vid 索引（多个 shard，首次调用时列出全部文件）。"""
    global _HF_INDEX
    if _HF_INDEX is not None:
        return _HF_INDEX
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(_HF_REPO, repo_type="dataset")
    _HF_INDEX = {
        f.rsplit("/", 1)[-1].removesuffix(".mp4"): f
        for f in files
        if f.startswith(_HF_VIDEO_PREFIX) and f.endswith(".mp4")
    }
    return _HF_INDEX


def _download_hf(vid: str, out_dir: Path) -> Path | None:
    """HuggingFace 逐文件镜像（googlevideo CDN 不可达时的首选来源）。"""
    import shutil

    from huggingface_hub import hf_hub_download

    remote = _hf_index().get(vid)
    if remote is None:
        return None
    try:
        p = hf_hub_download(
            repo_id=_HF_REPO, repo_type="dataset",
            filename=remote, local_dir=str(out_dir),
        )
        dst = out_dir / f"{vid}.mp4"
        shutil.copyfile(p, dst)  # 统一落位，便于断点续跑与后续 ingest
        return dst
    except Exception:  # noqa: BLE001 — 网络失败回退 YouTube
        return None


def _download_yt(vid: str, yid: str, start: float, end: float, out_dir: Path) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "yt_dlp",
         "-f", "mp4/bv*[height<=480]+ba/b[height<=480]",
         "--download-sections", f"*{start}-{end}",
         "--merge-output-format", "mp4",
         "--no-playlist", "--no-progress",
         "-o", str(out_dir / f"{vid}.%(ext)s"),
         f"https://www.youtube.com/watch?v={yid}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="抽样查询数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gt", type=str, default=str(DEFAULT_GT))
    ap.add_argument("--out-dir", type=str, default="data/qvhighlights")
    ap.add_argument("--exclude-dataset", type=str, action="append", default=[],
                    help="排除旧 dataset.json；可重复传入以排除所有开发/回归集")
    ap.add_argument("--source-disjoint", action="store_true", help="按 YouTube 原视频隔离，不仅按裁片 ID")
    ap.add_argument("--plan-only", action="store_true", help="仅冻结抽样清单，不联网或下载")
    args = ap.parse_args()

    exclude_vids: set[str] = set()
    for excluded_path in args.exclude_dataset:
        old = json.loads(Path(excluded_path).read_text(encoding="utf-8"))
        exclude_vids.update(s["video_id"] for s in old.get("samples", []))

    gt_path = Path(args.gt)
    if not gt_path.is_file():
        sys.exit(f"GT 文件不存在: {gt_path}（可用 QS_MOMENT_DETR_ROOT 指定 moment_detr 克隆）")

    entries = [json.loads(l) for l in gt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = select_entries(entries, args.n, args.seed, exclude_vids, args.source_disjoint)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "seed": args.seed, "requested_n": args.n, "picked_n": len(picked),
        "source_disjoint": args.source_disjoint, "excluded_datasets": args.exclude_dataset,
        "entries": picked,
    }
    selection_path = out_dir / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise ValueError("该目录已有不同的冻结抽样清单，请使用新的 --out-dir")
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.plan_only:
        print(f"冻结抽样 {len(picked)}/{args.n} 条：{selection_path}（未下载、未评测）")
        return
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    samples, downloaded, skipped = [], [], []
    for e in picked:
        vid = e["vid"]
        m = _VID_RE.match(vid)
        if not m:
            skipped.append({"vid": vid, "reason": "vid 格式无法解析"})
            continue
        yid, start, end = m.group(1), float(m.group(2)), float(m.group(3))
        video_file = videos_dir / f"{vid}.mp4"
        if not video_file.is_file():
            source = None
            # 1) HF 镜像（googlevideo CDN 不可达网络下的首选）
            hf_file = _download_hf(vid, videos_dir)
            if hf_file is not None:
                source = "hf-mirror"
                video_file = hf_file
            # 2) YouTube 分段下载
            elif _available(vid, yid) and _download_yt(vid, yid, start, end, videos_dir):
                source = "youtube"
                cands = list(videos_dir.glob(f"{vid}.*"))
                video_file = next((c for c in cands if c.suffix == ".mp4"), cands[0] if cands else None)
            else:
                skipped.append({"vid": vid, "reason": "HF 镜像无此文件且 YouTube 下载失败"})
                continue
            if video_file is None or video_file.stat().st_size < 100_000:
                skipped.append({"vid": vid, "reason": "下载产物无效"})
                continue
        downloaded.append(vid)
        samples.append({
            "qid": int(e["qid"]),
            "video": str(video_file),
            "video_id": vid,
            "query": e["query"],
            "windows": [[float(w[0]), float(w[1])] for w in e["relevant_windows"]],
            "type": "visual",  # QVHighlights val 无字幕，视觉主导
        })
        print(f"[{len(samples)}/{args.n}] {vid}  q={e['query'][:50]!r}")

    dataset = {
        "name": "qvhighlights-val-subset",
        "source": "Moment-DETR highlight_val_release.jsonl（标注 CC BY-NC-SA 4.0，仅评测使用）",
        "seed": args.seed,
        "source_disjoint": args.source_disjoint,
        "excluded_datasets": args.exclude_dataset,
        "n_picked": len(picked),
        "n_samples": len(samples),
        "n_skipped": len(skipped),
        "samples": samples,
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "availability.json").write_text(
        json.dumps({"checked": len(picked), "available": len(downloaded),
                    "skipped": skipped}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完成：{len(samples)}/{len(picked)} 可用（跳过 {len(skipped)}，原因见 availability.json）")
    print(f"评测集：{out_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()
