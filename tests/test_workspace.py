"""Persistent uploads, cache identity, execution and immutable export regressions."""
from pathlib import Path

import pytest

from queryseek.config import Config
from queryseek.workspace import Workspace, WorkspaceBusy


@pytest.fixture()
def workspace(qs_env):
    return Workspace(Config.load())


def save_demo(workspace, demo):
    return workspace.save_video('sample.mp4', demo['video'].read_bytes(),
                                subtitle=demo['srt'].read_bytes())[0]


def test_uploads_persist_deduplicate_and_isolate_names(workspace, demo):
    content = demo['video'].read_bytes()
    first, duplicate = workspace.save_video('../../sample.mp4', content)
    assert not duplicate
    assert Path(first['asset']['path']).is_relative_to(workspace.root)
    second, duplicate = workspace.save_video('different.mp4', content)
    assert duplicate and second['id'] == first['id']
    third, duplicate = workspace.save_video('sample.mp4', content + b'\0')
    assert not duplicate and third['id'] != first['id']
    assert Path(first['asset']['path']).read_bytes() == content
    reloaded = Workspace(workspace.cfg)
    assert {v['id'] for v in reloaded.videos()} == {first['id'], third['id']}


@pytest.mark.parametrize('name, content', [('bad.txt', b'abc'), ('bad.mp4', b''), ('bad.mp4', b'not video')])
def test_invalid_media_never_enters_library(workspace, name, content):
    with pytest.raises(ValueError):
        workspace.save_video(name, content)
    assert workspace.videos() == []


def test_size_is_checked_before_probe(workspace, monkeypatch):
    workspace.cfg.video.max_file_size_mb = 0
    monkeypatch.setattr('queryseek.workspace.asset_from_probe', lambda *a: pytest.fail('must not probe'))
    with pytest.raises(ValueError, match='大小'):
        workspace.save_video('large.mp4', b'content')


@pytest.mark.parametrize('subtitle', [b'', b'not a subtitle'])
def test_invalid_subtitle_does_not_publish_video(workspace, demo, subtitle):
    with pytest.raises(ValueError, match='SRT'):
        workspace.save_video('sample.mp4', demo['video'].read_bytes(), subtitle=subtitle)
    assert workspace.videos() == []


def test_subtitle_revision_forces_reindex_without_changing_old_result(workspace, demo):
    video = save_demo(workspace, demo)
    first = workspace.execute(video['id'], '系统架构', strategy='subtitle', execution='检索与剪辑')
    assert first['status'] == 'completed'
    old_result = workspace.artifact(first['id'], 'result.json').read_bytes()
    old_signature = workspace.video(video['id'])['index_signature']
    replacement = '1\n00:00:30,000 --> 00:00:40,000\n新的标签测试\n'
    same, duplicate = workspace.save_video('sample.mp4', demo['video'].read_bytes(),
                                          subtitle=replacement.encode())
    assert duplicate and same['status'] == 'saved'
    index = workspace.prepare_video(video['id'])
    assert not index.cache_hit
    assert [s.text for s in index.subtitles] == ['新的标签测试']
    assert workspace.video(video['id'])['index_signature'] != old_signature
    assert workspace.artifact(first['id'], 'result.json').read_bytes() == old_result
    again = workspace.prepare_video(video['id'])
    assert again.cache_hit


def test_run_is_idempotent_and_parent_link_is_durable(workspace, demo, monkeypatch):
    video = save_demo(workspace, demo)
    arguments = dict(strategy='subtitle', execution='检索与剪辑', request_id='same')
    first = workspace.execute(video['id'], '系统架构', **arguments)
    assert first['status'] == 'completed'
    assert workspace.result(first['id'])['selected_windows']
    monkeypatch.setattr(workspace, '_prepare', lambda *a: pytest.fail('must not execute twice'))
    assert workspace.execute(video['id'], '系统架构', **arguments)['id'] == first['id']
    with pytest.raises(ValueError, match='标识'):
        workspace.execute(video['id'], 'different query', **arguments)
    reloaded = Workspace(workspace.cfg)
    child = reloaded.execute(video['id'], '导出', strategy='subtitle', execution='检索与剪辑',
                             parent_id=first['id'])
    assert child['id'] != first['id'] and child['parent_id'] == first['id']
    assert len(reloaded.runs(video['id'])) == 2
    assert child['metrics']['input_tokens'] is None


def test_backend_failure_is_saved_without_secret(workspace, demo, monkeypatch):
    video = save_demo(workspace, demo)
    secret = 'secret-test-value'
    monkeypatch.setenv('VLM_API_KEY', secret)
    def fail(*a):
        raise RuntimeError(secret)
    monkeypatch.setattr(workspace, '_prepare', fail)
    run = workspace.execute(video['id'], 'query', strategy='subtitle', execution='检索与剪辑')
    assert run['status'] == 'failed'
    assert secret not in str(Workspace(workspace.cfg).run(run['id']))
    assert secret not in workspace.path(run['id'], 'trace.jsonl').read_text()


def test_real_vlm_does_not_silently_fall_back_to_mock(workspace):
    with pytest.raises(ValueError, match='密钥'):
        workspace.execute('sample', 'query', mock=False)
    assert workspace.runs() == []


def test_workspace_agent_timer_and_call_budget(workspace, demo):
    video = save_demo(workspace, demo)
    run = workspace.execute(video['id'], '系统架构', strategy='subtitle', budget=8)
    assert run['status'] == 'completed'
    result = workspace.result(run['id'])
    assert 0 <= result['elapsed_sec'] < 60
    assert result['vlm_calls'] <= workspace.cfg.agent.max_vlm_calls
    assert len(result['used_frame_ids']) <= 8


def test_progress_snapshots_are_live_durable_and_cannot_mutate_task(workspace, demo):
    video = save_demo(workspace, demo)
    stages = []
    persisted_stages = []

    def receive(snapshot):
        stages.append(snapshot['progress'][-1]['stage'])
        persisted = workspace.run(snapshot['id'])
        persisted_stages.append(persisted['progress'][-1]['stage'])
        snapshot['status'] = 'tampered'
        snapshot['progress'].clear()

    run = workspace.execute(video['id'], '系统架构', strategy='subtitle', execution='检索与剪辑',
                             on_progress=receive)
    assert stages == ['indexing', 'searching', 'saving', 'completed']
    assert persisted_stages == stages
    restored = Workspace(workspace.cfg).run(run['id'])
    assert restored['status'] == 'completed'
    assert [p['stage'] for p in restored['progress']] == stages


def test_disconnected_progress_display_does_not_fail_model_task(workspace, demo):
    video = save_demo(workspace, demo)
    def disconnected(snapshot):
        raise RuntimeError('display disconnected')
    run = workspace.execute(video['id'], '系统架构', strategy='subtitle', execution='检索与剪辑',
                             on_progress=disconnected)
    assert run['status'] == 'completed'


def test_export_failure_keeps_localization_and_retry_uses_new_artifact(workspace, demo, monkeypatch):
    video = save_demo(workspace, demo)
    run = workspace.execute(video['id'], '系统架构', strategy='subtitle', execution='检索与剪辑')
    result_bytes = workspace.artifact(run['id'], 'result.json').read_bytes()
    import queryseek.workspace as module
    original = module.export_clip
    def fail(*a, **kw):
        raise RuntimeError('encoding failed')
    monkeypatch.setattr(module, 'export_clip', fail)
    failed = workspace.export(run['id'])
    assert failed['status'] == 'completed' and failed['export_status'] == 'failed'
    monkeypatch.setattr(module, 'export_clip', original)
    exported = workspace.export(run['id'])
    assert exported['export_status'] == 'completed'
    assert workspace.artifact(run['id'], 'clip.mp4').is_file()
    assert exported['export_id'] != run['id']
    assert workspace.artifact(run['id'], 'result.json').read_bytes() == result_bytes


def test_stale_task_recovery_and_lock(workspace, demo):
    video = save_demo(workspace, demo)
    run = workspace.execute(video['id'], '架构', strategy='subtitle', execution='检索与剪辑')
    run['status'] = 'running'
    workspace._put('runs', run)
    with workspace.execution_lock():
        another = Workspace(workspace.cfg)
        assert another.run(run['id'])['status'] == 'running'
        with pytest.raises(WorkspaceBusy):
            another.prepare_video(video['id'])
    assert Workspace(workspace.cfg).run(run['id'])['status'] == 'interrupted'


def test_paths_and_artifacts_reject_traversal(workspace):
    with pytest.raises(ValueError):
        workspace.path('../escape')
    with pytest.raises(ValueError):
        workspace.run('../secret')
    with pytest.raises(ValueError):
        workspace.artifact('../secret', '.env')


def test_cited_images_are_frozen_and_export_uses_original_asset(workspace, demo, monkeypatch):
    video = save_demo(workspace, demo)
    index = workspace.prepare_video(video['id'])
    frame = index.frames[0]
    from queryseek.schemas import SearchResult, TimeWindow, Evidence
    from queryseek.retrieval import dispatch

    def search(*args):
        return SearchResult(video_id=video['id'], query='test', method='uniform',
                            windows=[TimeWindow(start_sec=0, end_sec=2)],
                            evidence=[Evidence(evidence_id='frame', type='frame',
                                               timestamp_sec=frame.timestamp_sec,
                                               source_file=frame.image_path)])

    monkeypatch.setattr(dispatch, 'search_index', search)
    run = workspace.execute(video['id'], 'test', strategy='uniform', execution='检索与剪辑')
    assert run['status'] == 'completed'
    evidence = workspace.result(run['id'])['evidence'][0]
    frozen = Path(evidence['source_file'])
    assert frozen.is_relative_to(workspace.path(run['id'], 'evidence'))
    original = frozen.read_bytes()
    Path(frame.image_path).write_bytes(b'cache has been rebuilt')
    assert frozen.read_bytes() == original
    assert run['source_asset']['sha256'] == video['asset']['sha256']
