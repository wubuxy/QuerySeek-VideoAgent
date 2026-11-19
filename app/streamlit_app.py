"""Three-column local workspace: library/settings, conversation, task inspector."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import streamlit as st

from queryseek.config import Config
from queryseek.retrieval.dispatch import AGENT_STRATEGIES
from queryseek.tools.paths import resolved_path_is_within
from queryseek.workspace import Workspace, WorkspaceBusy

st.set_page_config(page_title='QuerySeek-VideoAgent', page_icon='◉', layout='wide')
st.markdown('''<style>
 .block-container {max-width:1900px;padding:1.5rem 2rem 2rem}
 h1 {letter-spacing:-.04em;font-size:1.8rem!important}
 h2,h3 {letter-spacing:-.02em}
 [data-testid="stChatMessage"] {border:1px solid rgba(128,128,128,.15);border-radius:14px}
 [data-testid="stChatInput"] {border-radius:14px}
 [data-testid="stStatusWidget"] {border-radius:12px}
 .stCaption {line-height:1.65}
 @media(max-width:800px) {.block-container{padding:1rem}}
 </style>''', unsafe_allow_html=True)

cfg = Config.load().resolved()
workspace = Workspace(cfg)
STATUS = {'saved': '待建立索引', 'ready': '已就绪', 'registered': '已注册',
          'indexing': '正在索引', 'queued': '等待中', 'running': '运行中',
          'completed': '已完成', 'failed': '失败', 'interrupted': '已中断'}
PHASES = [('indexing', '准备视频与索引'), ('searching', '检索与 Agent 定位'),
          ('saving', '保存结果与证据')]


def stamp(value):
    return datetime.fromisoformat(value).astimezone(timezone(timedelta(hours=8))).strftime('%m-%d %H:%M')


def readable_image(path):
    if not path:
        return False
    p = Path(path).resolve()
    return p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and any(
        resolved_path_is_within(p, root.resolve()) for root in (cfg.cache_root, cfg.output_root))


def remember(run):
    st.session_state['active_run'] = run['id']
    st.session_state['pending_history'] = run['id']
    st.query_params['run'] = run['id']


def conversation(active):
    turns, seen = [], set()
    while active and active['id'] not in seen and len(turns) < 40:
        seen.add(active['id'])
        turns.append(active)
        parent = workspace.run(active['parent_id']) if active.get('parent_id') else None
        active = parent if parent and parent['video_id'] == active['video_id'] else None
    return list(reversed(turns))


def inspect_task(run):
    st.subheader('任务进程')
    if not run:
        st.info('等待新的定位任务')
        st.caption('提交消息后，这里显示实际处理阶段、消耗和证据。')
        return
    state = 'complete' if run['status'] == 'completed' else ('error' if run['status'] in {'failed', 'interrupted'} else 'running')
    progress = run.get('progress', [])
    phase = progress[-1]['stage'] if progress else None
    label = dict(PHASES).get(phase, STATUS.get(run['status'], run['status']))
    if run.get('export_status') == 'running':
        label, state = '正在生成视频片段', 'running'
    panel = st.status(label, state=state, expanded=True)
    if progress:
        entered = [i for i, (name, _) in enumerate(PHASES) if any(p['stage'] == name for p in progress)]
        latest = max(entered, default=-1)
        for i, (_, title) in enumerate(PHASES):
            if i < latest or (i == latest and run['status'] == 'completed'):
                note = '已完成'
            elif i == latest:
                note = '进行中' if run['status'] == 'running' else '未完成'
            else:
                note = '待处理'
            panel.write(f'{title} · {note}')
    else:
        panel.write('旧任务仅保留最终状态，无逐阶段记录。')
    st.caption(f"任务 {run['id']}\n\n{stamp(run['created_at'])} · {'Mock' if run['mock'] else run['model']}")
    if run.get('error'):
        st.warning(run['error'])
    metrics = run.get('metrics', {})
    if metrics:
        st.markdown('**本次消耗**')
        frames, calls = metrics.get('used_frames'), metrics.get('vlm_calls')
        st.write(f"观察帧 ID：{frames if frames is not None else '未知'} / 上限 {run['budget_frames']}")
        st.write(f"VLM 逻辑调用：{calls if calls is not None else '未知'}")
        st.write(f"处理耗时：{metrics.get('elapsed_sec', 0):.2f} 秒")
        if metrics.get('indexed_frames') is not None:
            st.write(f"检索索引帧：{metrics['indexed_frames']}")
        st.caption('Token 尚未统一汇总；帧 ID 计数不等同于源时间去重。')
    if run.get('export_status') in {'completed', 'failed', 'interrupted'}:
        st.write(f"片段导出：{STATUS.get(run['export_status'], run['export_status'])}")
    result = workspace.result(run['id']) or {}
    if result.get('budget_exhausted'):
        st.warning('任务已触及观察或调用预算上限。')
    if result.get('evidence'):
        with st.expander(f"查看证据 · {len(result['evidence'])} 条"):
            for evidence in result['evidence'][:24]:
                st.caption(f"{evidence.get('evidence_id', '')} · {evidence.get('model_id') or evidence.get('type')}"
                           f" · {evidence.get('timestamp_sec')} 秒")
                if evidence.get('content'):
                    st.write(evidence['content'])
                if evidence.get('type') == 'frame' and readable_image(evidence.get('source_file')):
                    st.image(evidence['source_file'], use_container_width=True)
    with st.expander('执行记录与参数'):
        if progress:
            st.json(progress)
        if result.get('observations'):
            st.json(result['observations'])
        if result.get('verifier_results'):
            st.json(result['verifier_results'])
        st.json({'strategy': run['strategy'], 'execution': run['execution'],
                 'budget_frames': run['budget_frames'], 'model': run['model'],
                 'parent_id': run.get('parent_id')})
        if result:
            st.json(result)


def show_answer(run, detailed=False):
    if run['status'] in {'failed', 'interrupted'}:
        st.write(run.get('error') or '任务未完成，可以重新提交定位请求。')
        return
    result = workspace.result(run['id']) or {}
    windows = result.get('selected_windows', [])
    if not windows:
        st.write('当前观察没有得到可返回的区间；这不能证明整个视频都不存在目标。')
        return
    st.write(f"定位完成，返回 {len(windows)} 个候选片段。你可以播放并核对证据。")
    st.caption(' · '.join(f"[{w['start_sec']:.3f}, {w['end_sec']:.3f}) 秒" for w in windows[:5]))
    if not detailed:
        return
    st.subheader('定位结果')
    for warning in result.get('warnings', []):
        st.warning(warning)
    selected_window = st.selectbox('播放定位片段', list(range(len(windows))),
                                   format_func=lambda i: f"[{windows[i]['start_sec']:.3f}, {windows[i]['end_sec']:.3f}) 秒",
                                   key=f"window_{run['id']}")
    video = workspace.video(run['video_id'])
    asset = run.get('source_asset') or (video['asset'] if video else None)
    if asset and Path(asset['path']).is_file():
        w = windows[selected_window]
        st.video(asset['path'], start_time=float(w['start_sec']), end_time=float(w['end_sec']))
    for name, label in [('result.json', '下载定位 JSON'), ('metrics.json', '下载成本 JSON')]:
        path = workspace.artifact(run['id'], name)
        if path.is_file():
            st.download_button(label, path.read_bytes(), file_name=f"{run['id']}_{name}",
                               mime='application/json', key=f"download_{run['id']}_{name}")
    with st.expander('导出与下载视频', expanded=run.get('export_status') in {'completed', 'failed'}):
        target = st.number_input('目标片长（秒，0 为全部候选）', min_value=0.0, max_value=1800.0,
                                 value=0.0, key=f"target_{run['id']}")
        if st.button('生成 MP4', key=f"export_{run['id']}"):
            try:
                update_inspector({**run, 'export_status': 'running'})
                workspace.export(run['id'], target or None)
                st.rerun()
            except (ValueError, WorkspaceBusy) as exc:
                update_inspector(workspace.run(run['id']))
                st.warning(str(exc))
        if run.get('export_error'):
            st.warning(run['export_error'])
        if run.get('export_status') == 'completed':
            path = workspace.artifact(run['id'], 'clip.mp4')
            if path.is_file():
                st.success(f"片段已保存 · {run['export_duration_sec']:.3f} 秒")
                st.video(str(path))
                st.download_button('下载 MP4', path.read_bytes(), file_name=f"{run['id']}.mp4",
                                   mime='video/mp4', key=f"clip_{run['id']}")


st.title('QuerySeek-VideoAgent')
st.caption('视频证据工作台 · 选择视频与预算，和 Agent 对话，查看任务进程')
left, middle, right = st.columns([1, 2.4, 1.15], gap='large')
videos = workspace.videos()
active_id = st.session_state.get('active_run') or st.query_params.get('run')
try:
    active = workspace.run(active_id) if active_id else None
except ValueError:
    active = None

with left:
    st.subheader('工作区')
    if videos:
        ids = [v['id'] for v in videos]
        preferred = st.session_state.pop('pending_video', None)
        if not preferred and 'selected_video' not in st.session_state and active:
            preferred = active['video_id']
        if preferred in ids:
            st.session_state['selected_video'] = preferred
        video_id = st.selectbox('当前视频', ids, key='selected_video',
                                format_func=lambda vid: next(v['title'] for v in videos if v['id'] == vid))
    else:
        video_id = st.text_input('已注册的视频 ID', value='sample')
        st.caption('上传视频开始，或选择已有 CLI 注册视频。')
    selected = workspace.video(video_id)
    if selected:
        st.caption(f"{selected['asset']['duration_sec']:.3f} 秒 · {STATUS.get(selected['status'], selected['status'])}")
    with st.expander('上传与保存视频'):
        with st.form('upload_form'):
            upload = st.file_uploader('视频文件', type=['mp4', 'mkv', 'mov'])
            subtitle = st.file_uploader('SRT 字幕（可选）', type=['srt'])
            save = st.form_submit_button('保存到视频库')
        if save:
            if upload is None:
                st.warning('请先选择视频文件')
            else:
                try:
                    saved, duplicate = workspace.save_video(upload.name, upload.getvalue(),
                                                            subtitle=subtitle.getvalue() if subtitle else None)
                    st.session_state['pending_video'] = saved['id']
                    st.session_state['notice'] = '已复用相同视频' if duplicate else '视频已保存'
                    st.rerun()
                except (ValueError, WorkspaceBusy) as exc:
                    st.warning(str(exc))
    if st.session_state.get('notice'):
        st.success(st.session_state.pop('notice'))
    with st.expander('预算与运行设置', expanded=True):
        execution = st.radio('执行方式', ['Agent', '检索与剪辑'], key='execution', horizontal=True)
        mock = st.checkbox('Mock（离线流程验证）', value=True, key='mock')
        strategies = ['adaptive', 'uniform', 'subtitle', 'semantic'] if execution == 'Agent' else ['curve', *AGENT_STRATEGIES]
        if st.session_state.get('last_execution') != execution:
            st.session_state['strategy'] = strategies[0]
            st.session_state['last_execution'] = execution
        strategy = st.selectbox('定位方式', strategies, key='strategy')
        budget = st.select_slider('观察帧预算', options=[8, 16, 32, 64], value=32, key='budget') if execution == 'Agent' or strategy in {'uniform', 'adaptive'} else 32
        if mock:
            st.caption('Mock 仅验证流程。上传视频可用 SRT 字幕检索体验。')
        else:
            st.caption(f"模型：{os.environ.get('VLM_MODEL', '尚未配置')}；可能调用外部服务。")
        if strategy in {'curve', 'semantic'}:
            st.caption('使用全视频索引；帧预算不限制索引成本。')
        if selected and st.button('建立 / 更新索引', key='index_video'):
            try:
                with st.spinner('正在建立索引…'):
                    workspace.prepare_video(video_id, mock)
                st.success('索引已就绪')
            except Exception as exc:
                st.warning(f'索引未完成（{type(exc).__name__}），请检查媒体与后端配置')
    st.divider()
    if st.button('新对话', key='new_conversation', use_container_width=True):
        st.session_state.pop('active_run', None)
        st.query_params.clear()
        st.rerun()
    st.markdown('**历史对话**')
    all_videos = st.checkbox('所有视频', value=False, key='all_history')
    runs = workspace.runs(None if all_videos else video_id)
    if runs:
        run_ids = [r['id'] for r in runs]
        pending_history = st.session_state.pop('pending_history', None)
        if st.session_state.get('history') not in run_ids:
            st.session_state.pop('history', None)
        if pending_history in run_ids:
            st.session_state['history'] = pending_history
        history_id = st.selectbox('选择历史任务', [r['id'] for r in runs], key='history',
                                 format_func=lambda rid: next(f"{r['query'][:28]} · {stamp(r['created_at'])}" for r in runs if r['id'] == rid))
        if st.button('打开历史结果', key='open_history', use_container_width=True):
            chosen = workspace.run(history_id)
            remember(chosen)
            st.session_state['pending_video'] = chosen['video_id']
            st.rerun()
    else:
        st.caption('还没有对话记录。')
    st.caption('视频和对话保存在本机，不会上传至 GitHub。')

if active and active['video_id'] != video_id:
    active = None

with right:
    inspector = st.empty()


def update_inspector(run):
    with inspector.container():
        inspect_task(run)


update_inspector(active)

with middle:
    st.subheader('与 Agent 对话')
    if selected:
        with st.expander('查看当前视频'):
            st.text(selected['title'])
            if Path(selected['asset']['path']).is_file():
                st.video(selected['asset']['path'])
    st.caption('每条消息发起一次独立预算的定位；追问请保留完整事件描述。')
    messages = st.container(height=560, border=True)
    with messages:
        if not active:
            with st.chat_message('assistant'):
                st.write('你好，选择一个视频，告诉我你想寻找的事件。')
                st.caption('例如：寻找讲解系统架构的片段。定位结果和证据会保存在这段对话中。')
        else:
            for turn in conversation(active):
                with st.chat_message('user'):
                    st.write(turn['query'])
                with st.chat_message('assistant'):
                    show_answer(turn, detailed=turn['id'] == active['id'])
    prompt = st.chat_input('描述你想寻找的事件，或补充定位要求…', key='chat_prompt', max_chars=2000)
    if prompt:
        parent_id = active['id'] if active else None
        identity = json.dumps([video_id, prompt, parent_id, execution, strategy, budget, mock,
                               (selected or {}).get('subtitle_sha256'), cfg.model_dump(mode='json')],
                              ensure_ascii=False, sort_keys=True)
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        if st.session_state.get('request_fingerprint') != fingerprint:
            st.session_state['request_fingerprint'] = fingerprint
            st.session_state['request_id'] = uuid.uuid4().hex
        with messages:
            with st.chat_message('user'):
                st.write(prompt)
            with st.chat_message('assistant'):
                try:
                    with st.spinner('正在处理你的定位请求…'):
                        current = workspace.execute(video_id, prompt, strategy=strategy, execution=execution,
                                                    budget=budget, mock=mock, parent_id=parent_id,
                                                    request_id=st.session_state['request_id'],
                                                    on_progress=update_inspector)
                    remember(current)
                    st.rerun()
                except (ValueError, WorkspaceBusy) as exc:
                    st.warning(str(exc))
