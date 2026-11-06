"""微调数据管线测试（全部离线：合成帧 + 合成 GT，无网络/ffmpeg/API）。

覆盖：拼接图尺寸与确定性、抽样去重与 holdout 分离、正/负帧计划分离、
记录结构、分片轮转与 zip、准入校验、Mock CLI 端到端、路径穿越拒绝。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

from queryseek.ft.montage import MAX_SIDE, build_montage
from queryseek.ft.records import (
    ShardWriter,
    build_inspect_record,
    build_verify_record,
    make_shard_zips,
)
from queryseek.ft.sampling import QuerySample, plan_frames, sample_queries
from queryseek.ft.validator import validate_shard


def _tile_bytes(text: str, side: int = 512) -> bytes:
    img = np.full((side, side, 3), (60, 60, 60), dtype=np.uint8)
    cv2.putText(img, text, (40, side // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf


def _write_frames(tmp_path: Path, ts: list[float]) -> list[Path]:
    out = tmp_path / "frames"
    out.mkdir(exist_ok=True)
    paths = []
    for i, t in enumerate(ts):
        p = out / f"t{int(t * 1000):08d}.jpg"
        p.write_bytes(_tile_bytes(f"t={t:.2f}"))
        paths.append(p)
    return paths


def _fake_entries(n: int) -> list[dict]:
    return [{
        "vid": f"vid{i:03d}_60.0_210.0",
        "query": f"查询 {i}",
        "relevant_windows": [[60.0 + i, 80.0 + i]],
    } for i in range(n)]


def _write_gt(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "train_gt.jsonl"
    p.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
                 encoding="utf-8")
    return p


# --- montage ---------------------------------------------------------------

def test_montage_dims_and_determinism(tmp_path):
    paths = _write_frames(tmp_path, [10.0, 20.0, 30.0, 40.0])
    b1 = build_montage(paths)
    b2 = build_montage(paths)
    assert b1 == b2  # JPEG 编码确定性
    arr = cv2.imdecode(np.frombuffer(b1, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(arr.shape[:2]) <= MAX_SIDE
    assert len(b1) < 10_000_000  # 百炼单图 10MB 上限


def test_montage_eight_frames_fits(tmp_path):
    paths = _write_frames(tmp_path, [float(i) for i in range(1, 9)])
    b = build_montage(paths, cols=4)
    arr = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(arr.shape[:2]) <= MAX_SIDE


# --- sampling --------------------------------------------------------------

def test_sample_queries_deterministic_holdout_and_exclude():
    entries = _fake_entries(10)
    t1, h1 = sample_queries(entries, seed=42, n_train=4, holdout_n=2)
    t2, h2 = sample_queries(entries, seed=42, n_train=4, holdout_n=2)
    assert [s.vid for s in t1] == [s.vid for s in t2]
    assert {s.vid for s in t1}.isdisjoint({s.vid for s in h1})
    assert len(t1) == 4 and len(h1) == 2
    # exclude_vids 生效
    t3, _ = sample_queries(entries, seed=42, n_train=4, holdout_n=0,
                           exclude_vids=tuple(s.vid for s in t1))
    assert {s.vid for s in t3}.isdisjoint({s.vid for s in t1})


def test_sample_queries_rejects_path_traversal():
    entries = _fake_entries(2)
    entries.append({"vid": "../evil_60.0_210.0", "query": "恶意",
                    "relevant_windows": [[1.0, 2.0]]})
    train, _ = sample_queries(entries, seed=0, n_train=10, holdout_n=0)
    assert all(".." not in s.vid for s in train)
    assert len(train) == 2


def test_plan_frames_pos_neg_separation():
    sample = QuerySample(vid="v_60.0_210.0", query="某事件",
                         windows=[(60.0, 80.0)], duration_sec=210.0)
    import random

    plans = plan_frames(sample, random.Random(7), frames_per_record=4)
    by_label: dict[str, list] = {}
    for p in plans:
        by_label.setdefault(p.label, []).append(p)
    # 正样本帧全部落在 GT 窗口内
    for p in by_label["positive"]:
        assert all(60.0 <= t <= 80.0 for t in p.timestamps)
        assert p.window == (60.0, 80.0)
    # 困难负样本在窗口外（≥2s 间隔），且落在紧邻时间带内（带端点 58.0/82.0 合法）
    for p in by_label["hard_negative"]:
        assert all(t <= 58.0 or t >= 82.0 for t in p.timestamps)
        assert p.window is not None and 60.0 not in range(int(p.window[0]), int(p.window[1]))
    # 简单负样本距窗口 ≥30s
    for p in by_label["easy_negative"]:
        assert all(t < 30.0 or t > 110.0 for t in p.timestamps)
    # inspect 混合：前半相关、后半不相关
    (mixed,) = by_label["mixed"]
    assert mixed.relevant_indexes == [1, 2]
    assert all(t < 80.0 for t in mixed.timestamps[:2])
    assert all(t > 80.0 for t in mixed.timestamps[2:])
    # 每查询 6 条：2 正 + 2 困难负 + 1 简单负 + 1 inspect
    assert len(plans) == 6


# --- records ---------------------------------------------------------------

def test_verify_record_structure():
    rec = build_verify_record("查询 X", (60.0, 80.0), [62.0, 66.0, 70.0, 74.0],
                              "confirmed", 0.9, "img_000000.jpg", "相关")
    msgs = rec["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant"]
    user_items = msgs[1]["content"]
    assert user_items[0]["image"] == "img_000000.jpg"  # 裸文件名
    assert "查询 X" in user_items[1]["text"]
    assert "[60.00, 80.00)" in user_items[1]["text"]
    ans = json.loads(msgs[2]["content"][0]["text"])
    assert ans == {"verdict": "confirmed", "confidence": 0.9, "reason": "相关"}


def test_inspect_record_relevant_ids():
    rec = build_inspect_record("查询 Y", [78.0, 79.5, 82.0, 83.5], [1, 2],
                               "img_000001.jpg")
    ans = json.loads(rec["messages"][2]["content"][0]["text"])
    assert ans["relevant_frame_ids"] == ["frame_000001", "frame_000002"]
    assert set(ans["reasons"]) == {"frame_000001", "frame_000002"}


def test_shard_writer_rotation_and_zip(tmp_path):
    w = ShardWriter(tmp_path, records_per_shard=2)
    for i in range(5):
        name = w.reserve_name(f"vid{i}")
        rec = build_verify_record(f"查询 {i}", (1.0, 2.0), [1.1, 1.2, 1.3, 1.4],
                                  "confirmed", 0.9, name, "相关")
        w.add(rec, _tile_bytes(f"t={i}"))
    w.flush()
    assert len(w.shard_dirs) == 3  # 2+2+1
    for d in w.shard_dirs:
        rows = (d / "data.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) in (1, 2)
        for line in rows:
            rec = json.loads(line)
            name = rec["messages"][1]["content"][0]["image"]
            assert (d / name).is_file()
    zips = make_shard_zips(tmp_path)
    assert len(zips) == 3
    with zipfile.ZipFile(zips[0]) as zf:
        names = zf.namelist()
        assert "data.jsonl" in names
        assert all("/" not in n for n in names)  # 全部裸名


# --- validator -------------------------------------------------------------

def _valid_shard(tmp_path) -> Path:
    d = tmp_path / "shard_0000"
    d.mkdir()
    name = "img_000000.jpg"
    (d / name).write_bytes(_tile_bytes("ok"))
    rec = build_verify_record("查询", (1.0, 2.0), [1.1, 1.2, 1.3, 1.4],
                              "refuted", 0.9, name, "无关")
    (d / "data.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    return d


def test_validator_passes_clean_shard(tmp_path):
    report = validate_shard(_valid_shard(tmp_path))
    assert report["ok"], report["errors"]
    assert report["records"] == 1 and report["images"] == 1


def test_validator_catches_violations(tmp_path):
    d = _valid_shard(tmp_path)
    # 1) 图片缺失
    (d / "img_000000.jpg").unlink()
    r = validate_shard(d)
    assert any("缺失" in e for e in r["errors"]) and not r["ok"]
    # 2) 图片带路径成分
    (d / "img_000000.jpg").write_bytes(_tile_bytes("ok"))
    line = (d / "data.jsonl").read_text(encoding="utf-8")
    (d / "data.jsonl").write_text(
        line.replace('"img_000000.jpg"', '"sub/img_000000.jpg"'), encoding="utf-8")
    r = validate_shard(d)
    assert any("裸文件名" in e for e in r["errors"])
    # 3) 图片超 1024px
    (d / "data.jsonl").write_text(line, encoding="utf-8")
    big = np.full((1200, 800, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", big)
    (d / "img_000000.jpg").write_bytes(buf.tobytes())
    r = validate_shard(d)
    assert any("边长超" in e for e in r["errors"])
    # 4) jsonl 损坏
    (d / "data.jsonl").write_text("not json\n", encoding="utf-8")
    r = validate_shard(d)
    assert any("JSON" in e for e in r["errors"])


# --- CLI 端到端（Mock，无网络/ffmpeg/API） -----------------------------------

def test_cli_mock_end_to_end(tmp_path):
    gt = _write_gt(tmp_path, _fake_entries(5))
    out_dir = tmp_path / "ft_out"
    env = dict(os.environ)
    env["QUERYSEEK_LOAD_DOTENV"] = "0"
    for k in ("VLM_API_KEY", "VLM_MODEL", "VLM_BASE_URL"):
        env.pop(k, None)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_ft_dataset.py"),
         "--mock", "--n-queries", "3", "--holdout", "1", "--seed", "42",
         "--gt", str(gt), "--out-dir", str(out_dir), "--zip"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    # holdout 分离：1 条 holdout + 3 条训练，训练 3×6=18 条记录
    holdout = [json.loads(l) for l in
               (out_dir / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(holdout) == 1
    assert manifest["n_train"] == 3 and manifest["skipped"] == []
    total_records = sum(manifest["counts"].values())
    assert total_records == 18
    for d in out_dir.glob("shard_*"):
        if d.is_dir():
            assert validate_shard(d)["ok"]
    zips = list(out_dir.glob("shard_*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as zf:
        assert "data.jsonl" in zf.namelist()
