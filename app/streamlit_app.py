"""QuerySeek-VideoAgent 最小 Web Demo（Streamlit，说明书 §1.3/§10 P3）。

功能：上传视频（+可选 SRT）→ ingest → 自然语言查询 → LangGraph Agent
（planner/initial_retrieve/verify/ground/finalize）→ 查看窗口/证据/Verifier
轨迹 → 可选导出 MP4 并播放。

运行：streamlit run app/streamlit_app.py
无密钥/离线：勾选 Mock 即可全流程运行（确定性 Mock，结果不代表真实准确率）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="QuerySeek-VideoAgent", layout="wide")

from queryseek.agent.state import default_budget, initial_state  # noqa: E402
from queryseek.config import Config  # noqa: E402
from queryseek.ingest.pipeline import ingest, load_index  # noqa: E402
from queryseek.logging_utils import TraceWriter  # noqa: E402
from queryseek.retrieval.dispatch import AGENT_STRATEGIES, search_index  # noqa: E402
from queryseek.schemas import RunSummary  # noqa: E402
from queryseek.tools.export import export_clip, new_job_id  # noqa: E402


def _run_demo_agent(index, cfg, video_id, query, strategy, budget, target, mock, t0):
    budget_map = default_budget({
        "max_iterations": cfg.agent.max_iterations,
        "max_vlm_calls": cfg.agent.max_vlm_calls,
        "max_frames": int(budget),
        "wall_time_limit_sec": cfg.agent.wall_time_limit_sec,
    })
    state = initial_state(video_id, query, strategy=strategy,
                          requested_duration_sec=float(target) if target else None, budget=budget_map)
    ckpt = cfg.output_root / "checkpoints" / f"demo_{video_id}.sqlite"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from queryseek.agent.graph import run_agent

        with st.spinner("Agent 运行中…"):
            with SqliteSaver.from_conn_string(str(ckpt)) as saver:
                return run_agent(state, {"thread_id": f"demo_{int(t0)}", "cfg": cfg.model_dump(), "mock": mock},
                                 checkpointer=saver)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Agent 不可用：{exc}（pip install -e '.[agent]'）")
        return None


st.title("QuerySeek-VideoAgent")
st.caption("基于主动视觉感知与 Query-Adaptive Temporal Retrieval 的长视频理解与智能剪辑")

cfg = Config.load().resolved()
uploads_dir = cfg.data_root / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 侧边栏：上传与 ingest
# ---------------------------------------------------------------------------
with st.sidebar:
    execution = st.radio("执行方式", ["Agent", "检索与剪辑"])
    mock = st.checkbox("Mock（离线确定性，无密钥可用）", value=True)
    st.header("1. 视频")
    uploaded = st.file_uploader("上传 MP4/MKV/MOV", type=["mp4", "mkv", "mov"])
    srt_up = st.file_uploader("可选 SRT 字幕", type=["srt"])
    video_id = st.text_input("video_id（或已注册 ID）", value="sample")
    if st.button("ingest", key="btn_ingest") and uploaded is not None:
        vpath = uploads_dir / Path(uploaded.name).name
        vpath.write_bytes(uploaded.getbuffer())
        srt_path = None
        if srt_up is not None:
            sp = uploads_dir / Path(srt_up.name).name
            sp.write_bytes(srt_up.getbuffer())
            srt_path = sp
        with st.spinner("ingest 中（镜头/抽帧/字幕/嵌入）…"):
            ingest_cfg = cfg.model_copy(deep=True)
            if mock:
                ingest_cfg.embedding.backend = "mock"
            idx = ingest(vpath, video_id=video_id or None, srt_path=srt_path, config=ingest_cfg)
        st.success(
            f"缓存{'命中' if idx.cache_hit else '新建'}："
            f"{len(idx.scenes)} 镜头 / {len(idx.frames)} 帧 / {len(idx.subtitles)} 字幕段"
        )

    st.header("2. 查询")
    query = st.text_input("自然语言查询", value="寻找讲解系统架构的片段")
    strategies = ["adaptive", "uniform", "subtitle", "semantic"] if execution == "Agent" else ["curve", *AGENT_STRATEGIES]
    strategy = st.selectbox("初始策略", strategies)
    budget = st.slider("VLM 观察帧预算", 8, 64, 32, 8) if execution == "Agent" or strategy in ("uniform", "adaptive") else None
    if strategy == "curve":
        st.caption("E4：使用全视频 CLIP 索引生成候选，无需 VLM；事件不存在时仍可能返回候选。")
    render = st.checkbox("导出 MP4", value=False)
    target = st.number_input("目标时长（秒）", 0, 120, 10, step=1) if render else None

# ---------------------------------------------------------------------------
# 主区：运行 Agent
# ---------------------------------------------------------------------------
run_label = "运行 Agent" if execution == "Agent" else "检索片段"
if st.button(run_label, key="btn_run") and query:
    t0 = time.perf_counter()
    try:
        index = load_index(video_id, cfg)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"视频未就绪：{exc}（请先上传并 ingest，或使用已注册的 video_id）")
        index = None

    if index is not None:
        job = None
        search_result = None
        if execution == "检索与剪辑":
            job = cfg.output_root / new_job_id()
            trace = TraceWriter(job / "trace.jsonl")
            try:
                with st.spinner("检索中…"):
                    search_result = search_index(cfg, index, video_id, query, strategy, budget, mock, trace)
                (job / "result.json").write_text(search_result.model_dump_json(indent=2), encoding="utf-8")
                final = {
                    "status": "ok" if search_result.windows else "no_result",
                    "selected_windows": [w.model_dump() for w in search_result.windows],
                    "evidence": [e.model_dump() for e in search_result.evidence],
                    "vlm_calls": search_result.vlm_calls, "used_frame_ids": search_result.used_frame_ids,
                    "indexed_frames": search_result.indexed_frames,
                    "online_encoded_frames": search_result.online_encoded_frames,
                    "embedding_cache_hit": search_result.embedding_cache_hit,
                    "warnings": search_result.warnings,
                }
            except Exception as exc:  # noqa: BLE001
                trace.event("error", error=str(exc))
                st.error(f"检索失败：{exc}")
                final = None
            finally:
                trace.close()
        else:
            final = _run_demo_agent(index, cfg, video_id, query, strategy, budget, target, mock, t0)

        if final is not None:
            elapsed = time.perf_counter() - t0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("状态", final.get("status", "?"))
            c2.metric("VLM 调用", final.get("vlm_calls", 0))
            c3.metric("VLM 观察帧数", len(final.get("used_frame_ids", [])))
            c4.metric("耗时", f"{elapsed:.1f}s")
            if final.get("embedding_cache_hit") is not None:
                st.caption(f"CLIP 索引帧数：{final['indexed_frames']}；本次编码帧数：{final['online_encoded_frames']}；"
                           f"嵌入缓存：{'命中' if final['embedding_cache_hit'] else '未命中'}")
            if final.get("plan"):
                st.info(f"plan: [{final['plan'].get('query_type')}] {final['plan'].get('rationale')}")

            if final.get("selected_windows"):
                st.subheader("选中窗口")
                st.table(
                    {
                        "start_sec": [w["start_sec"] for w in final["selected_windows"]],
                        "end_sec": [w["end_sec"] for w in final["selected_windows"]],
                        "时长": [round(w["end_sec"] - w["start_sec"], 2) for w in final["selected_windows"]],
                        "score": [w.get("score") for w in final["selected_windows"]],
                    }
                )
            else:
                st.warning("未定位到相关窗口（证据不足）")

            if final.get("evidence"):
                with st.expander("检索证据"):
                    st.json(final["evidence"])
                    for evidence in final["evidence"]:
                        source = evidence.get("source_file")
                        if evidence.get("type") == "frame" and source and Path(source).is_file():
                            st.image(source, caption=evidence.get("content", ""))

            if final.get("verifier_results"):
                st.subheader("Verifier 轨迹")
                st.table(
                    {
                        "窗口": [f"[{r['start_sec']:.1f},{r['end_sec']:.1f})" for r in final["verifier_results"]],
                        "verdict": [r["verdict"] for r in final["verifier_results"]],
                        "conf": [r.get("confidence") for r in final["verifier_results"]],
                        "reason": [str(r.get("reason", ""))[:50] for r in final["verifier_results"]],
                    }
                )

            if final.get("observations"):
                with st.expander("细化/恢复轨迹"):
                    st.json(final["observations"])
            for wmsg in final.get("warnings", []):
                st.warning(wmsg)
            for err in final.get("errors", []):
                st.error(err)

            manifest = None
            if render and final.get("status") == "ok" and final.get("selected_windows"):
                from queryseek.agent.state import dicts_to_windows

                job = job or cfg.output_root / new_job_id()
                try:
                    out, manifest = export_clip(
                        index.asset, dicts_to_windows(final["selected_windows"]),
                        cfg.output_root, cfg, job_id=job.name,
                        target_seconds=float(target) if target else None,
                    )
                    manifest = manifest.model_copy(update={"query": query, "strategy": strategy})
                    (job / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
                    st.success(f"已导出 {out.name}（{manifest.output_duration_sec:.1f}s）")
                    st.video(str(out))
                except Exception as exc:  # noqa: BLE001
                    final["status"] = "failed"
                    st.error(f"导出失败：{exc}")

            if search_result is not None:
                summary = RunSummary(
                    job_id=job.name, status=final["status"], strategy=strategy,
                    vlm_calls=search_result.vlm_calls, used_frames=len(search_result.used_frame_ids),
                    indexed_frames=search_result.indexed_frames,
                    online_encoded_frames=search_result.online_encoded_frames,
                    embedding_cache_hit=search_result.embedding_cache_hit,
                    elapsed_sec=time.perf_counter()-t0,
                    target_seconds=float(target) if target else None,
                    output_duration_sec=manifest.output_duration_sec if manifest else None,
                    exported_windows=len(manifest.segments) if manifest else 0,
                    warnings=search_result.warnings, output_dir=str(job),
                )
                (job / "metrics.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
                st.caption(f"结果与执行记录：{job}")

            with st.expander("完整结果"):
                st.json(final)
