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


def test_curve_ui_runs_and_saves_result(qs_env, monkeypatch, demo):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    from queryseek.ingest import pipeline
    from queryseek.retrieval import dispatch
    from queryseek.schemas import SearchResult, TimeWindow, VideoAsset
    from queryseek.ingest.registry import register_video
    from queryseek.config import Config

    asset = VideoAsset(video_id='sample', path=str(demo['video']), sha256='a' * 64,
                       duration_sec=60, width=320, height=240, fps=12, has_audio=False)
    register_video(Config.load().resolved().data_root, asset)
    monkeypatch.setattr(pipeline, "load_index", lambda *a: SimpleNamespace(asset=asset))

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
    # Ordinary widget reruns must not erase results or execute a new model call.
    at.checkbox[0].set_value(False).run()
    assert not at.exception
    assert any(s.value == '定位结果' for s in at.subheader)
    assert len(list((qs_env / 'outputs').glob('job_*'))) == 1
    fresh = AppTest.from_file(APP_PATH, default_timeout=60)
    fresh.query_params['run'] = job.name
    fresh.run()
    assert not fresh.exception
    assert any(s.value == '定位结果' for s in fresh.subheader)
    assert any(b.label == '打开历史结果' for b in fresh.button)


def test_workspace_ui_real_fixture_query_export_and_history(qs_env, demo):
    from streamlit.testing.v1 import AppTest
    from queryseek.workspace import Workspace
    from queryseek.config import Config

    workspace = Workspace(Config.load())
    video, _ = workspace.save_video('sample.mp4', demo['video'].read_bytes(),
                                     subtitle=demo['srt'].read_bytes())
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.radio[0].set_value('检索与剪辑').run()
    at.selectbox[0].set_value('subtitle').run()
    at.button(key='btn_run').click().run()
    assert not at.exception
    run = workspace.runs(video['id'])[0]
    assert run['status'] == 'completed'
    assert workspace.result(run['id'])['selected_windows']
    at.button(key=f"export_{run['id']}").click().run()
    assert not at.exception
    exported = workspace.run(run['id'])
    assert exported['export_status'] == 'completed'
    assert workspace.artifact(run['id'], 'clip.mp4').is_file()
    fresh = AppTest.from_file(APP_PATH, default_timeout=60).run()
    fresh.button(key='open_history').click().run()
    assert not fresh.exception
    assert any(s.value == '定位结果' for s in fresh.subheader)
    assert any('片段已保存' in s.value for s in fresh.success)
