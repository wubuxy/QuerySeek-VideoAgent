"""测试公共 fixture：合成样本（session 级）+ 隔离的数据目录（env 覆盖）。"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def demo(has_ffmpeg: bool, tmp_path_factory):
    """用 scripts/make_synthetic_fixture.py 生成合成样本（无网络依赖）。"""
    if not has_ffmpeg:
        pytest.skip("FFmpeg 未安装，跳过依赖 FFmpeg 的测试")
    out_dir = tmp_path_factory.mktemp("demo_assets")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_fixture.py"),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return {
        "video": out_dir / "sample.mp4",
        "srt": out_dir / "sample.srt",
        "labels": out_dir / "sample.labels.json",
        "dir": out_dir,
    }


@pytest.fixture()
def qs_env(tmp_path, monkeypatch):
    """隔离 cache/outputs/data 目录；清除 VLM 密钥并禁用 .env 加载
    （保证测试走 Mock 路径、绝不消耗真实 API）。"""
    monkeypatch.setenv("QUERYSEEK_LOAD_DOTENV", "0")
    monkeypatch.setenv("QUERYSEEK_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("QUERYSEEK_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("QUERYSEEK_DATA_ROOT", str(tmp_path / "data"))
    for k in ("VLM_API_KEY", "VLM_MODEL", "VLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path
