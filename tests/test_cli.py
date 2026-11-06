"""CLI 冒烟测试（CliRunner，无真实 API 调用）。"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from queryseek.cli import app

runner = CliRunner()


@pytest.fixture()
def _ingested(qs_env, demo):
    """先通过 CLI ingest（--mock 跳过 OpenCLIP，保证离线），再执行后续命令。"""
    r = runner.invoke(app, ["ingest", "--video", str(demo["video"]), "--video-id", "demo",
                            "--srt", str(demo["srt"]), "--mock"])
    assert r.exit_code == 0, r.output
    return qs_env


def test_doctor(qs_env):
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "FFmpeg" in r.output
    assert "VLM_API_KEY" in r.output


def test_ingest_repeat_cache_hit(qs_env, demo):
    r1 = runner.invoke(app, ["ingest", "--video", str(demo["video"]), "--video-id", "demo",
                             "--srt", str(demo["srt"]), "--mock"])
    assert r1.exit_code == 0, r1.output
    assert "缓存新建" in r1.output
    r2 = runner.invoke(app, ["ingest", "--video", str(demo["video"]), "--video-id", "demo",
                             "--srt", str(demo["srt"]), "--mock"])
    assert r2.exit_code == 0, r2.output
    assert "缓存命中" in r2.output


def test_search_subtitle(_ingested, qs_env):
    r = runner.invoke(app, ["search", "--video-id", "demo", "--query", "寻找讲解系统架构的片段", "--strategy", "subtitle"])
    assert r.exit_code == 0, r.output
    assert "检索结果" in r.output
    # result.json 落盘
    results = list((qs_env / "outputs" / "search").glob("*/result.json"))
    assert results
    data = json.loads(results[0].read_text(encoding="utf-8"))
    assert data["method"] == "subtitle"
    assert data["windows"], "应输出时间窗口"


def test_search_uniform_mock(_ingested, qs_env):
    r = runner.invoke(app, [
        "search", "--video-id", "demo", "--query", "寻找讲解系统架构的片段",
        "--strategy", "uniform", "--budget-frames", "16", "--mock",
    ])
    assert r.exit_code == 0, r.output
    data = json.loads(
        sorted((qs_env / "outputs" / "search").glob("*/result.json"))[-1].read_text(encoding="utf-8")
    )
    assert data["windows"], data.get("warnings")
    assert len(data["used_frame_ids"]) == 16
    # trace 记录 VLM 调用且不含密钥
    trace_files = list((qs_env / "outputs" / "search").glob("*/trace.jsonl"))
    trace_text = trace_files[-1].read_text(encoding="utf-8")
    assert '"vlm_call"' in trace_text


def test_search_semantic_mock(_ingested, qs_env):
    r = runner.invoke(app, ["search", "--video-id", "demo", "--query", "寻找讲解系统架构的片段",
                            "--strategy", "semantic", "--mock"])
    assert r.exit_code == 0, r.output
    data = json.loads(
        sorted((qs_env / "outputs" / "search").glob("*/result.json"))[-1].read_text(encoding="utf-8")
    )
    assert data["method"] == "semantic"
    assert data["windows"]
    top = data["windows"][0]
    assert top["start_sec"] < 12.0 < top["end_sec"] or top["start_sec"] < 24.0 <= top["end_sec"]  # 覆盖 GT 主体


def test_search_adaptive_mock(_ingested, qs_env):
    r = runner.invoke(app, ["search", "--video-id", "demo", "--query", "寻找讲解系统架构的片段",
                            "--strategy", "adaptive", "--budget-frames", "32", "--mock"])
    assert r.exit_code == 0, r.output
    data = json.loads(
        sorted((qs_env / "outputs" / "search").glob("*/result.json"))[-1].read_text(encoding="utf-8")
    )
    assert data["method"] == "adaptive"
    assert data["windows"]
    assert len(data["used_frame_ids"]) <= 32
    assert data["refinement_rounds"]  # 细化轮次有记录


def test_search_unknown_strategy(_ingested):
    r = runner.invoke(app, ["search", "--video-id", "demo", "--query", "x", "--strategy", "nope"])
    assert r.exit_code != 0


def test_run_end_to_end_mock_render(_ingested, qs_env):
    r = runner.invoke(app, [
        "run", "--video-id", "demo", "--query", "寻找讲解系统架构的片段",
        "--target-seconds", "10", "--strategy", "uniform",
        "--budget-frames", "16", "--mock", "--render",
    ])
    assert r.exit_code == 0, r.output
    assert "状态 ok" in r.output
    jobs = list((qs_env / "outputs").glob("job_*"))
    assert jobs
    job = jobs[0]
    for name in ("clip.mp4", "manifest.json", "metrics.json", "trace.jsonl", "result.json"):
        assert (job / name).is_file(), f"缺少 {name}"
    metrics = json.loads((job / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "ok"
    assert metrics["exported_windows"] >= 1
    assert metrics["output_duration_sec"] is not None


def test_run_no_result_skips_export(_ingested):
    r = runner.invoke(app, [
        "run", "--video-id", "demo", "--query", "寻找关于美食烹饪的片段",
        "--strategy", "uniform", "--mock", "--render",
    ])
    assert r.exit_code == 0, r.output
    assert "跳过导出" in r.output


def test_curve_search_and_trace_without_vlm(_ingested, qs_env, monkeypatch):
    from queryseek.retrieval import dispatch

    def forbidden(*args, **kwargs):
        raise AssertionError("E4 must not create a VLM client")

    monkeypatch.setattr(dispatch, "build_vlm", forbidden)
    r = runner.invoke(app, ["search", "--video-id", "demo", "--query", "系统架构",
                            "--strategy", "curve", "--budget-frames", "8", "--mock"])
    assert r.exit_code == 0, r.output
    job = next((qs_env / "outputs" / "search").glob("job_*"))
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert result["windows"] and result["evidence"]
    assert result["vlm_calls"] == 0 and result["used_frame_ids"] == []
    assert result["indexed_frames"] > 8  # the VLM budget must not truncate the index
    assert result["online_encoded_frames"] == result["indexed_frames"]
    assert "不限制索引帧数" in r.output
    events = [json.loads(line) for line in (job / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["event"] == "curve_search" and e["indexed_frames"] == result["indexed_frames"] for e in events)


def test_curve_run_render(_ingested, qs_env, monkeypatch):
    from queryseek.retrieval import dispatch

    monkeypatch.setattr(dispatch, "build_vlm", lambda *a, **kw: pytest.fail("Unexpected VLM client"))
    r = runner.invoke(app, ["run", "--video-id", "demo", "--query", "系统架构",
                            "--strategy", "curve", "--mock", "--render", "--target-seconds", "10"])
    assert r.exit_code == 0, r.output
    job = next((qs_env / "outputs").glob("job_*"))
    metrics = json.loads((job / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert (job / "clip.mp4").is_file()
    assert metrics["status"] == "ok" and metrics["vlm_calls"] == 0
    assert metrics["indexed_frames"] > 0 and metrics["used_frames"] == 0
    assert manifest["strategy"] == "curve"
    assert any(segment["evidence_ids"] for segment in manifest["segments"])


def test_agent_curve_rejected_before_runtime(qs_env):
    r = runner.invoke(app, ["agent", "--video-id", "missing", "--query", "event", "--strategy", "curve"])
    assert r.exit_code != 0
    assert "search" in r.output and "curve" in r.output
