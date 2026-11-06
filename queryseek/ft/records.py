"""构建百炼视觉 SFT 训练记录（ChatML messages + content 数组）并分片写出。

格式依据：阿里云百炼"调优数据上传规则"（来源与日期见 docs/FINE_TUNING.md）：
- data.jsonl 位于 zip 根目录；图片字段只写裸文件名、包内全局唯一、仅 ASCII
- 单图宽高 ≤1024px、≤10MB；单 zip ≤2GB（默认每片 ≤3000 条、zip ≤10 个）
- 提示词与 queryseek/models/compatible_api.py 的推理提示词同构（同一种问法），
  微调后模型可直接增强现有 inspect/verify 调用
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from queryseek.models.compatible_api import _SYSTEM_PROMPT


def _frame_lines(timestamps: list[float]) -> str:
    return "\n".join(
        f"frame_{i + 1:06d} t={t:.2f}s（拼接图 {i + 1} 号格）"
        for i, t in enumerate(timestamps)
    )


def _verify_user_text(query: str, window: tuple[float, float],
                      timestamps: list[float]) -> str:
    return (
        f"查询：{query}\n"
        f"候选时间窗口：[{window[0]:.2f}, {window[1]:.2f}) 秒\n"
        f"证据：拼接图中按时间顺序排列的帧（{_frame_lines(timestamps)}）\n"
        "判断该窗口内容是否确实与查询相关。\n"
        '只输出 JSON 对象，格式：{"verdict": "confirmed|refuted|insufficient", '
        '"confidence": 0.0 到 1.0 之间的小数, "reason": "一句话理由"}'
    )


def _inspect_user_text(query: str, timestamps: list[float]) -> str:
    return (
        f"查询：{query}\n"
        "下面是按时间顺序排列的视频帧，每帧带有 frame_id 与时间戳（秒）。\n"
        f"{_frame_lines(timestamps)}\n"
        "判断哪些帧的画面内容与查询相关（画面能作为回答/定位查询的证据）。\n"
        '只输出 JSON 对象，格式：{"relevant_frame_ids": ["frame_000001", ...], '
        '"reasons": {"frame_000001": "一句话理由"}, "summary": "总体一句话判断"}'
    )


def _messages(user_text: str, assistant_json: dict, image_name: str) -> dict:
    return {"messages": [
        {"role": "system", "content": [{"text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"image": image_name},
            {"text": user_text},
        ]},
        {"role": "assistant", "content": [
            {"text": json.dumps(assistant_json, ensure_ascii=False)},
        ]},
    ]}


def build_verify_record(
    query: str,
    window: tuple[float, float],
    timestamps: list[float],
    verdict: str,
    confidence: float,
    image_name: str,
    reason: str,
) -> dict:
    """verify 任务记录：候选窗口 + 拼接图 → verdict JSON（与 _VERIFY_PROMPT 同构）。"""
    if verdict not in {"confirmed", "refuted", "insufficient"}:
        raise ValueError(f"非法 verdict: {verdict!r}")
    return _messages(
        _verify_user_text(query, window, timestamps),
        {"verdict": verdict, "confidence": round(confidence, 2), "reason": reason},
        image_name,
    )


def build_inspect_record(
    query: str,
    timestamps: list[float],
    relevant_indexes: list[int],
    image_name: str,
) -> dict:
    """inspect 任务记录：拼接图 → 相关帧清单 JSON（与 _INSPECT_PROMPT 同构）。"""
    rel_ids = [f"frame_{i:06d}" for i in relevant_indexes]
    if not rel_ids:
        raise ValueError("relevant_indexes 不能为空")
    reasons = {i: "画面内容与查询直接相关" for i in rel_ids}
    summary = f"拼接图第 {relevant_indexes[0]}-{relevant_indexes[-1]} 格的内容与查询相关"
    return _messages(
        _inspect_user_text(query, timestamps),
        {"relevant_frame_ids": rel_ids, "reasons": reasons, "summary": summary},
        image_name,
    )


class ShardWriter:
    """把记录与拼接图按片写出：每片一个目录（data.jsonl + 裸名图片），可打包 zip。"""

    def __init__(self, out_dir: str | Path, records_per_shard: int = 3000):
        self.out_dir = Path(out_dir)
        self.limit = records_per_shard
        self._shard_index = -1
        self._rows: list[dict] = []
        self._used_names: set[str] = set()
        self.shard_dirs: list[Path] = []

    def _current_dir(self) -> Path:
        if self._shard_index < 0 or len(self._rows) >= self.limit:
            self.flush()
            self._shard_index += 1
            d = self.out_dir / f"shard_{self._shard_index:04d}"
            d.mkdir(parents=True, exist_ok=True)
            self.shard_dirs.append(d)
        return self.shard_dirs[-1]

    def reserve_name(self, prefix: str) -> str:
        """分配全局唯一的裸文件名（仅 ASCII，无目录成分）。"""
        base = re.sub(r"[^A-Za-z0-9_-]", "_", prefix)[:40] or "img"
        i = 0
        while True:
            name = f"{base}_{i:06d}.jpg"
            if name not in self._used_names:
                self._used_names.add(name)
                return name
            i += 1

    def add(self, record: dict, image_bytes: bytes) -> None:
        d = self._current_dir()
        name = None
        for m in record.get("messages", []):
            for item in m.get("content", []):
                if "image" in item:
                    name = item["image"]
        if name is None or not isinstance(name, str):
            raise ValueError("记录不含 image 字段")
        (d / name).write_bytes(image_bytes)
        self._rows.append(record)

    def flush(self) -> None:
        if not self._rows:
            return
        d = self.shard_dirs[-1]
        (d / "data.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self._rows),
            encoding="utf-8",
        )
        self._rows = []


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def make_shard_zips(out_dir: str | Path) -> list[Path]:
    """把每个 shard 目录打包成 zip：data.jsonl 位于 zip 根目录，图片裸名同层。"""
    out_dir = Path(out_dir)
    zips = []
    for shard_dir in sorted(out_dir.glob("shard_*")):
        if not shard_dir.is_dir():
            continue
        zip_path = shard_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(shard_dir / "data.jsonl", "data.jsonl")
            for img in sorted(shard_dir.glob("*")):
                if img.suffix.lower() in _IMG_EXTS:
                    zf.write(img, img.name)
        zips.append(zip_path)
    return zips
