"""QVHighlights train 集抽样与正/负帧时间戳计划（微调数据管线第一步）。

训练数据设计（docs/FINE_TUNING.md），每查询约 6 条记录：
- 正样本（×2）：GT 窗口内均匀取帧，两轮错开半个步长 → 应判 confirmed
- 困难负样本（×2）：紧邻 GT 窗口前后的短时间带 → 训练"边界在哪"
- 简单负样本（×1）：远离任何 GT 窗口的随机段 → 训练"内容无关"
- inspect 混合（×1）：跨 GT 结束边界取帧，前半相关后半不相关 → 训练"哪些帧相关"
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

# vid 白名单（DECISIONS #9 同类约束）：QV 格式 {youtube_id}_{start}_{end}，
# 显式拒绝 `..` 等路径穿越（vid 会进入工作目录名）。
VID_WHITELIST = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

DEFAULT_TASKS = ("verify", "inspect")
MARGIN_SEC = 2.0        # 负样本与 GT 窗口的最小间隔
HARD_BAND_SEC = 12.0    # 困难负样本时间带长度
EASY_MIN_DIST_SEC = 30.0  # 简单负样本距任何 GT 窗口的最小距离


@dataclass
class QuerySample:
    """一条训练查询：视频 + 查询文本 + GT 窗口（下载后经 ffprobe 填充时长）。"""
    vid: str
    query: str
    windows: list[tuple[float, float]]
    duration_sec: float | None = None

    @property
    def sorted_windows(self) -> list[tuple[float, float]]:
        return sorted(self.windows)


@dataclass
class FramePlan:
    """一条训练记录的帧时间戳计划。"""
    task: str          # "verify" | "inspect"
    label: str         # "positive" | "hard_negative" | "easy_negative" | "mixed"
    window: tuple[float, float] | None  # verify 任务的候选窗口（inspect 为 None）
    timestamps: list[float] = field(default_factory=list)
    relevant_indexes: list[int] = field(default_factory=list)  # inspect：1 起始相关序号


def load_train_entries(gt_path: str | Path) -> list[dict]:
    """读 train GT（moment_detr highlight_train_release.jsonl）。"""
    return [json.loads(line) for line in
            Path(gt_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_queries(
    entries: list[dict],
    seed: int,
    n_train: int,
    holdout_n: int,
    exclude_vids: tuple[str, ...] = (),
) -> tuple[list[QuerySample], list[QuerySample]]:
    """按 vid 去重 + 白名单校验后固定种子洗牌；先取 holdout_n 条再取 n_train 条。

    holdout 不生成训练记录，留作微调后效果评测（dev/test 分离，说明书 §8.3）。
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    unique: list[dict] = []
    for e in entries:
        vid = e.get("vid", "")
        if not vid or vid in seen or vid in exclude_vids:
            continue
        if not VID_WHITELIST.match(vid) or ".." in vid:
            continue
        if not e.get("relevant_windows"):
            continue
        seen.add(vid)
        unique.append(e)
    rng.shuffle(unique)

    def to_sample(e: dict) -> QuerySample:
        return QuerySample(
            vid=e["vid"],
            query=e["query"],
            windows=[(float(w[0]), float(w[1])) for w in e["relevant_windows"]],
        )

    holdout = [to_sample(e) for e in unique[:holdout_n]]
    train = [to_sample(e) for e in unique[holdout_n:holdout_n + n_train]]
    return train, holdout


def plan_frames(
    sample: QuerySample,
    rng: random.Random,
    frames_per_record: int = 4,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
) -> list[FramePlan]:
    """为一个查询生成全部训练记录的帧计划。"""
    if sample.duration_sec is None:
        raise ValueError("需要 duration_sec（先下载视频并 ffprobe）")
    windows = sample.sorted_windows
    plans: list[FramePlan] = []
    if "verify" in tasks:
        plans += _verify_plans(sample, windows, rng, frames_per_record)
    if "inspect" in tasks:
        p = _inspect_plan(sample, windows[0], frames_per_record)
        if p is not None:
            plans.append(p)
    return plans


def _verify_plans(
    sample: QuerySample,
    windows: list[tuple[float, float]],
    rng: random.Random,
    k: int,
) -> list[FramePlan]:
    s0, e0 = windows[0]
    dur = sample.duration_sec
    eps = max(0.5, min(1.5, (e0 - s0) / 10))
    plans: list[FramePlan] = []
    # 正样本：窗口内均匀取帧 ×2 轮（错开半个步长，同答案不同画面）
    for shift in (0.0, 0.5):
        step = (e0 - s0 - 2 * eps) / max(1, k - 1)
        ts = sorted({round(s0 + eps + (i + shift) * step, 2) for i in range(k)})
        ts = [min(t, e0 - eps) for t in ts]
        plans.append(FramePlan("verify", "positive", (s0, e0), ts))
    # 困难负样本：窗口前后紧邻时间带
    for band in _hard_bands(s0, e0, dur):
        step = (band[1] - band[0]) / max(1, k - 1)
        ts = [round(band[0] + i * step, 2) for i in range(k)]
        plans.append(FramePlan("verify", "hard_negative", band, ts))
    # 简单负样本：距所有 GT 窗口 ≥ EASY_MIN_DIST_SEC 的随机段
    ts = _far_timestamps(windows, dur, k, rng)
    if ts:
        wlen = min(max(e0 - s0, 4.0), 20.0)
        plans.append(FramePlan("verify", "easy_negative",
                               (ts[0], min(ts[0] + wlen, dur)), ts))
    return plans


def _hard_bands(s0: float, e0: float, dur: float) -> list[tuple[float, float]]:
    bands = []
    if s0 - MARGIN_SEC - HARD_BAND_SEC >= 0.0:
        bands.append((s0 - MARGIN_SEC - HARD_BAND_SEC, s0 - MARGIN_SEC))
    if e0 + MARGIN_SEC + HARD_BAND_SEC <= dur:
        bands.append((e0 + MARGIN_SEC, e0 + MARGIN_SEC + HARD_BAND_SEC))
    return bands


def _far_timestamps(
    windows: list[tuple[float, float]],
    dur: float,
    k: int,
    rng: random.Random,
    attempts: int = 30,
) -> list[float] | None:
    span = max(6.0, k * 2.0)
    if dur - span <= 0:
        return None
    for _ in range(attempts):
        t0 = rng.uniform(0.0, dur - span)
        ts = [round(t0 + i * (span / max(1, k - 1)), 2) for i in range(k)]
        if all(t < s - EASY_MIN_DIST_SEC or t > e + EASY_MIN_DIST_SEC
               for t in ts for s, e in windows):
            return ts
    return None


def _inspect_plan(
    sample: QuerySample,
    w0: tuple[float, float],
    k: int,
) -> FramePlan | None:
    """跨 GT 结束边界取帧：前半在窗口内（相关），后半紧邻窗口外（不相关）。"""
    s0, e0 = w0
    dur = sample.duration_sec
    half = max(1, k // 2)
    eps = min(1.5, max(0.5, (e0 - s0) / (half + 2)))
    inside = [round(e0 - (half - i) * eps, 2) for i in range(half)]
    outside = [round(e0 + MARGIN_SEC + i * 1.5, 2) for i in range(k - half)]
    if inside[0] < s0 + 0.3 or outside[-1] > dur - 0.5:
        return None
    return FramePlan(
        task="inspect", label="mixed", window=None,
        timestamps=inside + outside,
        relevant_indexes=list(range(1, half + 1)),
    )
