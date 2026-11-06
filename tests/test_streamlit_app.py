"""Streamlit App 冒烟测试（AppTest，headless 运行，验证页面可加载不崩溃）。"""

from __future__ import annotations

from pathlib import Path

import pytest

apptest = pytest.importorskip("streamlit.testing.v1", reason="streamlit 未安装")

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture()
def app_test(qs_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_app_loads_without_exception(app_test):
    assert not app_test.exception
    assert "QuerySeek-VideoAgent" in app_test.title[0].value


def test_app_run_without_video_graceful(app_test):
    """未 ingest 任何视频时点击运行：给出友好提示而非崩溃。"""
    buttons = {b.label: b for b in app_test.button}
    assert "运行 Agent" in buttons
    buttons["运行 Agent"].click()
    app_test.run()
    assert not app_test.exception
    # 数据目录为空（无已注册视频）：应显示"视频未就绪"提示
    assert any("视频未就绪" in (w.value or "") for w in app_test.warning)


def test_curve_mode_has_no_vlm_budget(app_test):
    app_test.radio[0].set_value("检索与剪辑").run()
    assert not app_test.exception
    assert app_test.selectbox[0].value == "curve"
    assert not app_test.slider
    buttons = {b.label: b for b in app_test.button}
    buttons["检索片段"].click().run()
    assert not app_test.exception
    assert any("视频未就绪" in w.value for w in app_test.warning)


def test_curve_ui_runs_and_saves_result(qs_env, monkeypatch):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    from queryseek.ingest import pipeline
    from queryseek.retrieval import dispatch
    from queryseek.schemas import SearchResult, TimeWindow

    monkeypatch.setattr(pipeline, "load_index", lambda *a: SimpleNamespace())

    def fake_search(cfg, index, vid, query, strategy, budget, mock, trace):
        assert strategy == "curve" and budget is None and mock
        return SearchResult(video_id=vid, query=query, method="clip_curve",
                            windows=[TimeWindow(start_sec=1, end_sec=4, score=1)],
                            indexed_frames=100, embedding_cache_hit=True)

    monkeypatch.setattr(dispatch, "search_index", fake_search)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.radio[0].set_value("检索与剪辑").run()
    at.button(key="btn_run").click().run()
    assert not at.exception
    job = next((qs_env / "outputs").glob("job_*"))
    import json
    metrics = json.loads((job / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["indexed_frames"] == 100 and metrics["used_frames"] == 0 and metrics["vlm_calls"] == 0
    assert (job / "result.json").is_file() and (job / "trace.jsonl").is_file()
