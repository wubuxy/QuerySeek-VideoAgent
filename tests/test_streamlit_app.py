"""Three-column workspace, persisted chat, progress and export regressions."""
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('streamlit.testing.v1')
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / 'app' / 'streamlit_app.py')


@pytest.fixture()
def app_test(qs_env):
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def test_three_regions_put_controls_chat_and_progress_in_distinct_columns(app_test):
    assert not app_test.exception
    assert 'QuerySeek-VideoAgent' in app_test.title[0].value
    columns = app_test.get('column')
    assert len(columns) == 3
    assert columns[0].radio(key='execution').value == 'Agent'
    assert columns[0].select_slider(key='budget').value == 32
    assert columns[1].chat_input(key='chat_prompt')
    assert any(s.value == '任务进程' for s in columns[2].subheader)
    assert not app_test.sidebar.radio


def test_chat_without_video_is_graceful(app_test):
    app_test.chat_input(key='chat_prompt').set_value('寻找系统架构').run()
    assert not app_test.exception
    assert any('视频未就绪' in w.value for w in app_test.warning)


def test_curve_mode_does_not_offer_vlm_frame_budget(app_test):
    app_test.radio(key='execution').set_value('检索与剪辑').run()
    assert not app_test.exception
    assert app_test.selectbox(key='strategy').value == 'curve'
    assert not app_test.select_slider
    app_test.chat_input(key='chat_prompt').set_value('系统架构').run()
    assert not app_test.exception
    assert any('视频未就绪' in w.value for w in app_test.warning)


def test_chat_result_survives_widget_rerun_and_url_restore(qs_env, monkeypatch, demo):
    from queryseek.ingest import pipeline
    from queryseek.retrieval import dispatch
    from queryseek.ingest.registry import register_video
    from queryseek.config import Config
    from queryseek.schemas import SearchResult, TimeWindow, VideoAsset
    from queryseek.workspace import Workspace

    cfg = Config.load().resolved()
    asset = VideoAsset(video_id='sample', path=str(demo['video']), sha256='a' * 64,
                       duration_sec=60, width=320, height=240, fps=12, has_audio=False)
    register_video(cfg.data_root, asset)
    monkeypatch.setattr(pipeline, 'load_index', lambda *a: SimpleNamespace(asset=asset))
    calls = []

    def search(cfg, index, vid, query, strategy, budget, mock, trace):
        assert strategy == 'curve' and budget is None and mock
        calls.append(query)
        return SearchResult(video_id=vid, query=query, method='clip_curve',
                            windows=[TimeWindow(start_sec=1, end_sec=4)],
                            indexed_frames=100, embedding_cache_hit=True)

    monkeypatch.setattr(dispatch, 'search_index', search)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.radio(key='execution').set_value('检索与剪辑').run()
    at.chat_input(key='chat_prompt').set_value('寻找系统架构').run()
    assert not at.exception
    workspace = Workspace(cfg)
    run = workspace.runs('sample')[0]
    assert run['status'] == 'completed' and calls == ['寻找系统架构']
    assert len(at.chat_message) == 2
    assert any(s.value == '定位结果' for s in at.subheader)
    assert run['metrics']['indexed_frames'] == 100
    assert [p['stage'] for p in run['progress']] == ['indexing', 'searching', 'saving', 'completed']
    at.checkbox(key='all_history').set_value(True).run()
    assert not at.exception and calls == ['寻找系统架构']
    fresh = AppTest.from_file(APP_PATH, default_timeout=60)
    fresh.query_params['run'] = run['id']
    fresh.run()
    assert not fresh.exception
    assert len(fresh.chat_message) == 2
    assert fresh.button(key='open_history')


def test_chat_fixture_export_followup_new_chat_and_history(qs_env, demo):
    from queryseek.workspace import Workspace
    from queryseek.config import Config

    workspace = Workspace(Config.load())
    video, _ = workspace.save_video('sample.mp4', demo['video'].read_bytes(),
                                     subtitle=demo['srt'].read_bytes())
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.radio(key='execution').set_value('检索与剪辑').run()
    at.selectbox(key='strategy').set_value('subtitle').run()
    at.chat_input(key='chat_prompt').set_value('寻找讲解系统架构的片段').run()
    assert not at.exception
    first = workspace.runs(video['id'])[0]
    assert first['status'] == 'completed'
    at.button(key=f"export_{first['id']}").click().run()
    assert not at.exception
    assert workspace.run(first['id'])['export_status'] == 'completed'
    assert workspace.artifact(first['id'], 'clip.mp4').is_file()
    at.chat_input(key='chat_prompt').set_value('寻找导出剪辑的片段').run()
    assert not at.exception
    second = workspace.runs(video['id'])[0]
    assert second['parent_id'] == first['id']
    assert len(at.chat_message) == 4
    at.button(key='new_conversation').click().run()
    assert not at.exception
    assert len(at.chat_message) == 1  # Greeting, without erasing persistent runs.
    assert len(workspace.runs(video['id'])) == 2
    at.button(key='open_history').click().run()
    assert not at.exception and len(at.chat_message) == 4
    fresh = AppTest.from_file(APP_PATH, default_timeout=60).run()
    fresh.selectbox(key='history').set_value(first['id']).run()
    fresh.button(key='open_history').click().run()
    assert not fresh.exception
    assert any('片段已保存' in s.value for s in fresh.success)


def test_history_url_selects_its_video_when_library_has_multiple_videos(qs_env, demo):
    from queryseek.workspace import Workspace
    from queryseek.config import Config

    workspace = Workspace(Config.load())
    first, _ = workspace.save_video('first.mp4', demo['video'].read_bytes(), subtitle=demo['srt'].read_bytes())
    run = workspace.execute(first['id'], '系统架构', strategy='subtitle', execution='检索与剪辑')
    workspace.save_video('second.mp4', demo['video'].read_bytes() + b'\0')
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.query_params['run'] = run['id']
    at.run()
    assert not at.exception
    assert at.selectbox(key='selected_video').value == first['id']
    assert len(at.chat_message) == 2
