"""Persistent local video workspace. Run: streamlit run app/streamlit_app.py."""
from __future__ import annotations

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
 .block-container {max-width:1200px; padding-top:2rem}
 [data-testid="stMetric"] {background:rgba(19,153,143,.07);border-radius:12px;padding:14px}
 [data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.2)}
 h1 {letter-spacing:-.04em} .stCaption {line-height:1.7}
 </style>''', unsafe_allow_html=True)

cfg = Config.load().resolved()
workspace = Workspace(cfg)
STATUS = {'saved': '已保存，待索引', 'ready': '已就绪', 'registered': '已注册',
          'indexing': '正在索引', 'queued': '等待中', 'running': '运行中',
          'completed': '已完成', 'failed': '失败', 'interrupted': '已中断'}


def readable_image(path):
    if not path:
        return False
    p = Path(path).resolve()
    return p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and any(
        resolved_path_is_within(p, root.resolve()) for root in (cfg.cache_root, cfg.output_root))


def remember(run):
    st.session_state['active_run'] = run['id']
    st.query_params['run'] = run['id']


def show_result(run):
    st.subheader('定位结果')
    st.write(run['query'])
    st.caption(f"{STATUS.get(run['status'], run['status'])} · {run['id']} · "
               f"{'Mock' if run['mock'] else run['model']} · {run['strategy']}")
    if run.get('parent_id'):
        st.caption(f"关联任务：{run['parent_id']}；本次独立计量预算。当前追问按新描述重新定位。")
    if run.get('error'):
        st.warning(run['error'])
    result = workspace.result(run['id'])
    if not result:
        return
    metrics = run.get('metrics', {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('观察帧 ID 数', metrics.get('used_frames', '未知'))
    c2.metric('VLM 逻辑调用', metrics.get('vlm_calls') if metrics.get('vlm_calls') is not None else '未知')
    c3.metric('处理耗时', f"{metrics.get('elapsed_sec', 0):.2f}s")
    c4.metric('Token 用量', '未汇总')
    st.caption('观察计数沿用现有策略的帧 ID；源时间去重、HTTP 重试及累计图像展示将在成本接口阶段统一。')
    if metrics.get('indexed_frames') is not None:
        st.caption(f"检索索引帧：{metrics['indexed_frames']}；本次编码帧：{metrics.get('online_encoded_frames')}。"
                   '全视频索引成本与 VLM 观察预算分开记录。')
    for warning in result.get('warnings', []):
        st.warning(warning)
    windows = result.get('selected_windows', [])
    video = workspace.video(run['video_id'])
    asset = run.get('source_asset') or (video['asset'] if video else None)
    source = Path(asset['path']) if asset else None
    if windows:
        st.dataframe([{'开始（秒）': w['start_sec'], '结束（秒）': w['end_sec'],
                       '时长（秒）': round(w['end_sec'] - w['start_sec'], 3)} for w in windows],
                     hide_index=True, use_container_width=True)
        chosen = st.selectbox('播放定位片段', list(range(len(windows))),
                              format_func=lambda i: f"[{windows[i]['start_sec']:.3f}, {windows[i]['end_sec']:.3f}) 秒",
                              key=f"window_{run['id']}")
        if source and source.is_file():
            st.video(str(source), start_time=float(windows[chosen]['start_sec']),
                     end_time=float(windows[chosen]['end_sec']))
    else:
        st.info('没有可展示的定位区间；这不等于已证明整段视频不存在目标。')
    for name, label in [('result.json', '下载定位 JSON'), ('metrics.json', '下载成本 JSON')]:
        file = workspace.artifact(run['id'], name)
        if file.is_file():
            st.download_button(label, file.read_bytes(), file_name=f"{run['id']}_{name}",
                               mime='application/json', key=f"download_{run['id']}_{name}")
    if windows and run['status'] == 'completed':
        with st.expander('导出片段', expanded=run.get('export_status') in {'completed', 'failed'}):
            st.caption('0 表示导出全部候选区间；实际片长以导出记录为准。')
            target = st.number_input('目标片长（秒）', min_value=0.0, max_value=1800.0,
                                     value=0.0, key=f"target_{run['id']}")
            if st.button('生成 MP4', key=f"export_{run['id']}"):
                try:
                    with st.spinner('正在导出并验证视频…'):
                        run = workspace.export(run['id'], target or None)
                except (ValueError, WorkspaceBusy) as exc:
                    st.warning(str(exc))
            if run.get('export_error'):
                st.warning(run['export_error'])
            if run.get('export_status') == 'completed':
                clip = workspace.artifact(run['id'], 'clip.mp4')
                if clip.is_file():
                    st.success(f"片段已保存 · {run['export_duration_sec']:.3f} 秒")
                    st.video(str(clip))
                    st.download_button('下载 MP4', clip.read_bytes(), file_name=f"{run['id']}.mp4",
                                       mime='video/mp4', key=f"clip_{run['id']}")
    with st.expander('画面证据与执行记录'):
        st.caption('检索相似度、模型判断和字幕均保留来源；相关性分数不是事件正确概率。'
                   '新假设标签控制器与块树视图尚未接入。')
        for evidence in result.get('evidence', [])[:24]:
            st.caption(f"{evidence.get('evidence_id', '')} · 来源：{evidence.get('model_id') or evidence.get('type')}"
                       f" · 时间：{evidence.get('timestamp_sec')} 秒")
            if evidence.get('content'):
                st.write(evidence['content'])
            if evidence.get('type') == 'frame' and readable_image(evidence.get('source_file')):
                st.image(evidence['source_file'], width=360)
        if result.get('verifier_results'):
            st.dataframe(result['verifier_results'], hide_index=True)
        if result.get('observations'):
            st.json(result['observations'])
        st.json(result)


st.title('QuerySeek-VideoAgent')
st.caption('视频证据工作台 · 保存视频，定位事件，核对每一次观察')

videos = workspace.videos()
with st.sidebar:
    st.subheader('运行设置')
    execution = st.radio('执行方式', ['Agent', '检索与剪辑'])
    mock = st.checkbox('Mock（离线流程验证）', value=True)
    if mock:
        st.caption('Mock 不理解真实画面。上传视频可用字幕检索验证流程；视觉演示需合成内容提示。')
    else:
        st.caption(f"VLM：{os.environ.get('VLM_MODEL', '尚未配置')}。真实模式可能调用外部模型服务。")
    strategies = ['adaptive', 'uniform', 'subtitle', 'semantic'] if execution == 'Agent' else ['curve', *AGENT_STRATEGIES]
    strategy = st.selectbox('定位方式', strategies)
    budget = st.select_slider('独立观察帧上限', options=[8, 16, 32, 64], value=32) if execution == 'Agent' or strategy in {'uniform', 'adaptive'} else 32
    if strategy in {'curve', 'semantic'}:
        st.caption('此方式使用全视频视觉索引；VLM 帧上限不限制索引成本。')
    st.divider()
    st.subheader('当前视频')
    if videos:
        ids = [v['id'] for v in videos]
        preferred = st.session_state.pop('pending_video', None)
        if preferred in ids:
            st.session_state['selected_video'] = preferred
        video_id = st.selectbox('从视频库选择', ids,
                                format_func=lambda vid: next(v['title'] for v in videos if v['id'] == vid),
                                key='selected_video')
    else:
        video_id = st.text_input('已注册的视频 ID', value='sample')
    st.caption('当前为本机单用户工作台。上传与结果保存在本机，不会推送到 GitHub。')

library_tab, locate_tab, history_tab = st.tabs(['视频库', '定位工作台', '历史任务'])
with library_tab:
    st.subheader('保存与管理视频')
    st.caption('保存原视频后建立索引，同一视频可反复查询。字幕随视频版本保存。')
    with st.form('upload_form', clear_on_submit=False):
        upload = st.file_uploader('视频文件', type=['mp4', 'mkv', 'mov'])
        subtitle = st.file_uploader('SRT 字幕（可选）', type=['srt'])
        save = st.form_submit_button('保存到视频库', type='primary')
    if save:
        if upload is None:
            st.warning('请先选择视频文件')
        else:
            try:
                with st.spinner('校验并保存视频…'):
                    saved, duplicate = workspace.save_video(upload.name, upload.getvalue(),
                                                            subtitle=subtitle.getvalue() if subtitle else None)
                st.session_state['pending_video'] = saved['id']
                st.session_state['notice'] = '已找到相同视频，复用已保存内容' if duplicate else '视频已保存，可建立索引'
                st.rerun()
            except (ValueError, WorkspaceBusy) as exc:
                st.warning(str(exc))
    if st.session_state.get('notice'):
        st.success(st.session_state.pop('notice'))
    if not videos:
        st.info('视频库还是空的。上传第一个视频，或用 CLI 建立演示索引。')
    selected = workspace.video(video_id)
    if selected:
        st.text(selected['title'])
        asset = selected['asset']
        st.caption(f"{asset['duration_sec']:.3f} 秒 · {asset['width']} × {asset['height']} · "
                   f"{STATUS.get(selected['status'], selected['status'])} · ID：{selected['id']}")
        if readable_image(selected.get('thumbnail')):
            st.image(selected['thumbnail'], width=240)
        if st.button('建立 / 更新索引', key='index_video'):
            try:
                with st.spinner('正在建立镜头、画面与字幕索引…'):
                    workspace.prepare_video(video_id, mock)
                st.success('索引已就绪，可以定位事件')
            except Exception as exc:
                st.warning(f'索引未完成（{type(exc).__name__}），请检查媒体与后端配置')
        if Path(asset['path']).is_file():
            with st.expander('预览原视频'):
                st.video(asset['path'])
    if videos:
        st.dataframe([{'视频': v['title'], '时长（秒）': v['asset']['duration_sec'],
                       '状态': STATUS.get(v['status'], v['status'])} for v in videos],
                     hide_index=True, use_container_width=True)

with locate_tab:
    st.subheader('你想找到什么事件？')
    st.caption('描述可在画面中核对的目标。查询会持久保存；每次定位使用独立预算。')
    active_id = st.session_state.get('active_run') or st.query_params.get('run')
    try:
        active = workspace.run(active_id) if active_id else None
    except ValueError:
        active = None
        st.warning('任务链接无效')
    parent_id = None
    if active and active['video_id'] == video_id:
        if st.checkbox('关联上一次定位（新任务重新搜索）', key='follow_up'):
            parent_id = active['id']
    query = st.text_area('事件描述或定位追问', value='寻找讲解系统架构的片段', max_chars=2000)
    current_video = workspace.video(video_id) or {}
    request_key = json.dumps([video_id, query, strategy, execution, budget, mock, parent_id,
                              current_video.get('subtitle_sha256'),
                              current_video.get('asset', {}).get('sha256'),
                              cfg.model_dump(mode='json')], ensure_ascii=False, sort_keys=True)
    if st.session_state.get('request_key') != request_key:
        st.session_state['request_key'] = request_key
        st.session_state['request_id'] = uuid.uuid4().hex
    run_label = '运行 Agent' if execution == 'Agent' else '检索片段'
    if st.button(run_label, key='btn_run', type='primary'):
        try:
            with st.spinner('正在索引并定位，完成后可从历史任务恢复…'):
                active = workspace.execute(video_id, query, strategy=strategy, execution=execution,
                                           budget=budget, mock=mock, parent_id=parent_id,
                                           request_id=st.session_state['request_id'])
            remember(active)
        except (ValueError, WorkspaceBusy) as exc:
            st.warning(str(exc))
    if st.button('允许按当前参数重新运行', key='new_request'):
        st.session_state['request_id'] = uuid.uuid4().hex
        st.info('已创建新的请求标识，再次点击运行将独立计量成本')
    if active:
        if active['video_id'] != video_id:
            st.info('当前显示的是历史任务；新查询将使用侧边栏选择的视频。')
        show_result(active)

with history_tab:
    st.subheader('持久任务记录')
    st.caption('每条记录保留模型、预算和参数。中断任务保留记录，可新建任务重跑。')
    all_videos = st.checkbox('显示所有视频的任务', value=False)
    runs = workspace.runs(None if all_videos else video_id)
    if not runs:
        st.info('还没有查询记录。完成一次定位后，可以随时回到这里。')
    else:
        history_id = st.selectbox('选择历史任务', [r['id'] for r in runs],
                                 format_func=lambda rid: next(f"{r['query'][:45]} · {STATUS.get(r['status'], r['status'])} · {r['created_at']}" for r in runs if r['id'] == rid))
        if st.button('打开历史结果', key='open_history'):
            chosen_run = workspace.run(history_id)
            remember(chosen_run)
            st.session_state['pending_video'] = chosen_run['video_id']
            st.rerun()
        st.dataframe([{'查询': r['query'], '状态': STATUS.get(r['status'], r['status']),
                       '方式': r['strategy'], '帧上限': r['budget_frames'],
                       '模式': 'Mock' if r['mock'] else r['model'],
                       '导出': r.get('export_status'), '任务': r['id']} for r in runs],
                     hide_index=True, use_container_width=True)
