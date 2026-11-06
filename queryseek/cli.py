"""QuerySeek-VideoAgent CLI（typer）。

命令：
- doctor  环境诊断（Python / FFmpeg / 依赖 / 配置 / 密钥状态 / 目录可写）
- ingest  视频 -> 结构化时序索引（缓存）
- search  检索时间窗口（subtitle | uniform | semantic | adaptive | curve）
- run     检索 + FFmpeg 导出（manifest.json / metrics.json / trace.jsonl）
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import PIPELINE_VERSION, __version__
from .agent.graph import LangGraphUnavailable
from .agent.state import default_budget, initial_state
from .config import Config
from .ingest.pipeline import VideoIndex, ingest, load_index
from .logging_utils import TraceWriter
from .retrieval.dispatch import AGENT_STRATEGIES, STRATEGIES, search_index
from .schemas import RunSummary, SearchResult
from .tools.export import export_clip, new_job_id

app = typer.Typer(
    name="queryseek",
    help="QuerySeek-VideoAgent：主动视觉感知与时序检索的长视频理解与智能剪辑",
    no_args_is_help=True,
)
console = Console()

def _config(config_path: Optional[Path]) -> Config:
    from .env import ensure_dotenv

    ensure_dotenv()  # 载入项目根 .env（测试以 QUERYSEEK_LOAD_DOTENV=0 禁用）
    cfg = Config.load(config_path).resolved()
    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    return cfg


def _execute_search(
    cfg: Config,
    index: VideoIndex,
    video_id: str,
    query: str,
    strategy: str,
    budget_frames: Optional[int],
    mock: bool,
    trace: TraceWriter,
) -> SearchResult:
    return search_index(cfg, index, video_id, query, strategy, budget_frames, mock, trace)


def _print_search_result(result: SearchResult, job_dir: Path, elapsed_sec: float) -> None:
    (job_dir / "result.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    if result.windows:
        t = Table(title=f"检索结果（{result.method}，{len(result.windows)} 个窗口）")
        t.add_column("#"); t.add_column("start_sec"); t.add_column("end_sec")
        t.add_column("时长"); t.add_column("score"); t.add_column("证据数")
        for i, w in enumerate(result.windows):
            t.add_row(
                str(i + 1), f"{w.start_sec:.2f}", f"{w.end_sec:.2f}",
                f"{w.duration_sec:.2f}s",
                f"{w.score:.2f}" if w.score is not None else "-",
                str(len(w.evidence_ids)),
            )
        console.print(t)
    else:
        console.print("[yellow]未定位到相关窗口（证据不足）[/yellow]")
    for wmsg in result.warnings:
        console.print(f"[yellow]警告[/yellow] {wmsg}")
    console.print(
        f"vlm_calls={result.vlm_calls} 观察帧数={len(result.used_frame_ids)} "
        f"证据={len(result.evidence)} 查询耗时={elapsed_sec:.2f}s "
        f"输出目录={job_dir}"
    )
    if result.embedding_cache_hit is not None:
        console.print(f"CLIP 索引帧数={result.indexed_frames} "
                      f"本次编码帧数={result.online_encoded_frames} "
                      f"嵌入缓存{'命中' if result.embedding_cache_hit else '未命中'}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

_REQUIRED_DEPS = {
    "pydantic": "核心", "numpy": "核心", "typer": "核心", "rich": "核心",
    "PyYAML": "核心", "python-dotenv": "核心", "srt": "核心",
    "scenedetect": "核心", "opencv-python-headless": "核心",
    "rank-bm25": "核心", "httpx": "核心",
}
_OPTIONAL_DEPS = {
    "torch": "P2", "open-clip-torch": "P2", "faiss-cpu": "P2",
    "langgraph": "P3", "langgraph-checkpoint-sqlite": "P3", "streamlit": "P3",
    "faster-whisper": "P1(本地 ASR)",
}


@app.command()
def doctor(
    config_path: Annotated[Optional[Path], typer.Option("--config")] = None,
) -> None:
    """环境诊断：列出 Python/FFmpeg/依赖/配置/密钥状态/目录可写性。"""
    console.print(f"[bold]QuerySeek-VideoAgent[/bold] v{__version__} (pipeline v{PIPELINE_VERSION})")
    console.print(f"Python: {sys.version.split()[0]} @ {sys.executable}")

    import shutil

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        console.print(f"[green]FFmpeg[/green] {ffmpeg}")
        console.print(f"[green]FFprobe[/green] {ffprobe}")
    else:
        console.print("[red]FFmpeg/FFprobe 未安装[/red]（https://ffmpeg.org/download.html），依赖它的功能不可用")

    table = Table(title="Python 依赖")
    table.add_column("包", style="cyan")
    table.add_column("状态", style="green")
    table.add_column("版本")
    for dist, phase in {**_REQUIRED_DEPS, **_OPTIONAL_DEPS}.items():
        try:
            table.add_row(dist, f"[green]OK[/green]（{phase}）",
                          importlib.metadata.version(dist))
        except importlib.metadata.PackageNotFoundError:
            table.add_row(dist, f"[yellow]缺失（{phase}）[/yellow]", "-")
    console.print(table)

    try:
        cfg = _config(config_path)
        console.print(f"[green]配置[/green] {config_path or 'configs/default.yaml'} 加载成功")
        console.print(
            f"  cache_root={cfg.cache_root}\n  output_root={cfg.output_root}\n  data_root={cfg.data_root}"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]配置加载失败[/red]: {exc}")
        raise typer.Exit(code=1)

    key = os.environ.get("VLM_API_KEY", "")
    console.print(
        f"[green]VLM_API_KEY[/green] {'已设置' if key else '未设置（可用 --mock 运行）'}"
        f"  VLM_MODEL={os.environ.get('VLM_MODEL') or '(未设置)'}"
        f"  VLM_BASE_URL={os.environ.get('VLM_BASE_URL', 'https://api.openai.com/v1')}"
    )

    for p in (cfg.cache_root, cfg.output_root, cfg.data_root):
        try:
            probe = p / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            console.print(f"[green]可写[/green] {p}")
        except OSError as exc:
            console.print(f"[red]不可写[/red] {p}: {exc}")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command("ingest")
def ingest_cmd(
    video: Annotated[Path, typer.Option("--video", help="输入视频路径（MP4/MKV/MOV 等 FFmpeg 可解析格式）")],
    video_id: Annotated[Optional[str], typer.Option("--video-id", help="注册 ID（默认取文件名主干）")] = None,
    srt: Annotated[Optional[Path], typer.Option("--srt", help="可选 SRT 字幕；不提供则尝试本地 ASR")] = None,
    mock: Annotated[bool, typer.Option("--mock", help="跳过真实 OpenCLIP 嵌入（检索时用 Mock 后端）")] = False,
    config_path: Annotated[Optional[Path], typer.Option("--config")] = None,
) -> None:
    """解析视频：元信息、镜头、关键帧、字幕、（可选）帧嵌入，构建缓存索引并注册。"""
    cfg = _config(config_path)
    if mock:
        cfg = cfg.model_copy(deep=True)
        cfg.embedding.backend = "mock"
    trace = TraceWriter(cfg.output_root / "ingest" / (video_id or video.stem) / "trace.jsonl")
    t0 = time.perf_counter()
    try:
        index = ingest(video, video_id=video_id, srt_path=srt, config=cfg, trace=trace)
    except Exception as exc:  # noqa: BLE001
        trace.event("error", error=str(exc))
        console.print(f"[red]ingest 失败[/red]: {exc}")
        raise typer.Exit(code=1)
    finally:
        trace.close()

    console.print(
        f"[green]ingest 完成[/green]（{time.perf_counter() - t0:.1f}s，缓存{'命中' if index.cache_hit else '新建'}）"
    )
    t = Table(title=f"视频索引: {index.video_id}")
    t.add_column("字段"); t.add_column("值")
    a = index.asset
    t.add_row("video_id", a.video_id)
    t.add_row("sha256", a.sha256[:16] + "…")
    t.add_row("时长", f"{a.duration_sec:.2f}s")
    t.add_row("帧率/尺寸", f"{a.fps:.3f} fps / {a.width}x{a.height}")
    t.add_row("音轨", "有" if a.has_audio else "无")
    t.add_row("镜头数", str(len(index.scenes)))
    t.add_row("关键帧数", str(len(index.frames)))
    t.add_row("字幕段数", str(len(index.subtitles)))
    t.add_row("缓存目录", str(index.cache_dir))
    console.print(t)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command()
def search(
    video_id: Annotated[str, typer.Option("--video-id", help="ingest 时注册的视频 ID")],
    query: Annotated[str, typer.Option("--query", help="自然语言查询")],
    strategy: Annotated[str, typer.Option("--strategy", help=" | ".join(STRATEGIES))] = "uniform",
    budget_frames: Annotated[Optional[int], typer.Option("--budget-frames", help="VLM 帧预算；curve 使用全视频索引，不受此项限制")] = None,
    mock: Annotated[bool, typer.Option("--mock", help="使用确定性 Mock VLM（无密钥可跑）")] = False,
    config_path: Annotated[Optional[Path], typer.Option("--config")] = None,
) -> None:
    """按自然语言查询检索相关时间窗口，输出 result.json 与 trace.jsonl。"""
    cfg = _config(config_path)
    index = load_index(video_id, cfg)
    job_dir = cfg.output_root / "search" / new_job_id()
    trace = TraceWriter(job_dir / "trace.jsonl")

    t0 = time.perf_counter()
    try:
        result = _execute_search(cfg, index, video_id, query, strategy, budget_frames, mock, trace)
    except Exception as exc:  # noqa: BLE001
        trace.event("error", error=str(exc))
        console.print(f"[red]search 失败[/red]: {exc}")
        raise typer.Exit(code=1)
    finally:
        trace.close()

    _print_search_result(result, job_dir, elapsed_sec=time.perf_counter() - t0)
    # 证据不足是合法结果（开发说明书 §12.2），不是错误


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    video_id: Annotated[str, typer.Option("--video-id")],
    query: Annotated[str, typer.Option("--query")],
    strategy: Annotated[str, typer.Option("--strategy", help=" | ".join(STRATEGIES))] = "uniform",
    target_seconds: Annotated[Optional[float], typer.Option("--target-seconds", help="期望导出的总时长（秒）")] = None,
    budget_frames: Annotated[Optional[int], typer.Option("--budget-frames")] = None,
    mock: Annotated[bool, typer.Option("--mock")] = False,
    render: Annotated[bool, typer.Option("--render", help="执行 FFmpeg 导出")] = False,
    config_path: Annotated[Optional[Path], typer.Option("--config")] = None,
) -> None:
    """检索 + 可选导出：输出 clip.mp4、manifest.json、metrics.json、trace.jsonl。"""
    cfg = _config(config_path)
    index = load_index(video_id, cfg)
    job_dir = cfg.output_root / new_job_id()
    trace = TraceWriter(job_dir / "trace.jsonl")
    t0 = time.perf_counter()
    status = "ok"

    try:
        result = _execute_search(cfg, index, video_id, query, strategy, budget_frames, mock, trace)

        if not result.windows:
            status = "no_result"
            console.print("[yellow]未定位到相关窗口，跳过导出[/yellow]")

        manifest = None
        if render and result.windows:
            try:
                _out, manifest = export_clip(
                    index.asset, result.windows, cfg.output_root, cfg,
                    job_id=job_dir.name, target_seconds=target_seconds, trace=trace,
                )
                manifest = manifest.model_copy(update={"query": query, "strategy": strategy})
                (job_dir / "manifest.json").write_text(
                    manifest.model_dump_json(indent=2), encoding="utf-8"
                )
                trace.event("export_done", output=manifest.output_file,
                            duration_sec=manifest.output_duration_sec)
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                trace.event("error", error=f"导出失败: {exc}")
                console.print(f"[red]导出失败[/red]: {exc}")

        summary = RunSummary(
            job_id=job_dir.name,
            status=status,
            strategy=strategy,
            vlm_calls=result.vlm_calls,
            used_frames=len(result.used_frame_ids),
            indexed_frames=result.indexed_frames,
            online_encoded_frames=result.online_encoded_frames,
            embedding_cache_hit=result.embedding_cache_hit,
            elapsed_sec=round(time.perf_counter() - t0, 3),
            target_seconds=target_seconds,
            output_duration_sec=manifest.output_duration_sec if manifest else None,
            exported_windows=len(result.windows) if (render and status == "ok") else 0,
            budget_exhausted=result.budget_exhausted,
            warnings=result.warnings,
            output_dir=str(job_dir),
        )
        (job_dir / "metrics.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        trace.event("error", error=str(exc))
        console.print(f"[red]run 失败[/red]: {exc}")
        raise typer.Exit(code=1)
    finally:
        trace.close()

    _print_search_result(result, job_dir, elapsed_sec=time.perf_counter() - t0)
    console.print(f"[bold]状态[/bold] {status}  输出目录: {job_dir}")
    if status == "failed":
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# agent（P3）
# ---------------------------------------------------------------------------

@app.command()
def agent(
    video_id: Annotated[str, typer.Option("--video-id")],
    query: Annotated[str, typer.Option("--query")],
    strategy: Annotated[str, typer.Option("--strategy", help=" | ".join(AGENT_STRATEGIES))] = "adaptive",
    target_seconds: Annotated[Optional[float], typer.Option("--target-seconds")] = None,
    budget_frames: Annotated[Optional[int], typer.Option("--budget-frames", help="视觉帧预算上限")] = None,
    mock: Annotated[bool, typer.Option("--mock")] = False,
    render: Annotated[bool, typer.Option("--render", help="对选中窗口执行 FFmpeg 导出")] = False,
    thread_id: Annotated[Optional[str], typer.Option("--thread-id", help="任务线程 ID（默认自动生成；相同 ID 可恢复）")] = None,
    config_path: Annotated[Optional[Path], typer.Option("--config")] = None,
) -> None:
    """LangGraph Agent：plan → initial_retrieve → verify ⇄ ground → finalize → validate。"""
    if strategy not in AGENT_STRATEGIES:
        raise typer.BadParameter("Agent 仅支持 subtitle/uniform/semantic/adaptive；E4 请使用 search 或 run --strategy curve")
    cfg = _config(config_path)
    load_index(video_id, cfg)  # 提前校验索引可用
    job_dir = cfg.output_root / "agent" / new_job_id()
    trace_path = job_dir / "trace.jsonl"
    thread = thread_id or job_dir.name
    budget = default_budget({
        "max_iterations": cfg.agent.max_iterations,
        "max_vlm_calls": cfg.agent.max_vlm_calls,
        "max_frames": budget_frames or cfg.retrieval.frame_budget * 2,
        "wall_time_limit_sec": cfg.agent.wall_time_limit_sec,
    })
    state = initial_state(
        video_id, query, strategy=strategy,
        requested_duration_sec=target_seconds, budget=budget,
    )

    t0 = time.perf_counter()
    try:
        from .agent.graph import run_agent

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            raise LangGraphUnavailable("langgraph-checkpoint-sqlite 未安装：pip install -e '.[agent]'")

        ckpt_dir = cfg.output_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(ckpt_dir / f"agent_{video_id}.sqlite")) as saver:
            final = run_agent(
                state,
                {"thread_id": thread, "cfg": cfg.model_dump(), "mock": mock,
                 "trace_path": str(trace_path)},
                checkpointer=saver,
            )
    except LangGraphUnavailable as exc:
        console.print(f"[red]agent 不可用[/red]: {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]agent 失败[/red]: {exc}")
        raise typer.Exit(code=1)
    finally:
        pass

    if render and final.get("status") == "ok" and final.get("selected_windows"):
        from .agent.state import dicts_to_windows

        try:
            index = load_index(video_id, cfg)
            _out, manifest = export_clip(
                index.asset, dicts_to_windows(final["selected_windows"]),
                job_dir.parent, cfg, job_id=job_dir.name,  # 输出到 outputs/agent/<job>/
                target_seconds=target_seconds,
            )
            manifest = manifest.model_copy(update={"query": query, "strategy": f"agent:{strategy}"})
            (job_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            final["artifacts"]["clip"] = manifest.output_file
        except Exception as exc:  # noqa: BLE001
            final["errors"] = list(final.get("errors", [])) + [f"导出失败: {exc}"]
            final["status"] = "failed"
            console.print(f"[red]导出失败[/red]: {exc}")

    (job_dir / "state.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    console.print(f"[bold]Agent 状态[/bold] {final.get('status')}  thread_id={thread}")
    if final.get("plan"):
        console.print(f"  plan: {final['plan'].get('query_type')} / {final['plan'].get('rationale')}")
    if final.get("selected_windows"):
        t = Table(title=f"选中窗口（{len(final['selected_windows'])}）")
        t.add_column("#"); t.add_column("start_sec"); t.add_column("end_sec")
        t.add_column("时长"); t.add_column("score")
        for i, w in enumerate(final["selected_windows"]):
            t.add_row(str(i + 1), f"{w['start_sec']:.2f}", f"{w['end_sec']:.2f}",
                      f"{w['end_sec'] - w['start_sec']:.2f}s",
                      f"{w.get('score') or 0:.2f}")
        console.print(t)
    t2 = Table(title="Verifier 轨迹")
    t2.add_column("窗口"); t2.add_column("verdict"); t2.add_column("conf"); t2.add_column("reason")
    for r in final.get("verifier_results", []):
        t2.add_row(f"[{r['start_sec']:.1f},{r['end_sec']:.1f})", r["verdict"],
                   f"{r.get('confidence', 0):.2f}", str(r.get("reason", ""))[:60])
    console.print(t2)
    for wmsg in final.get("warnings", []):
        console.print(f"[yellow]警告[/yellow] {wmsg}")
    for err in final.get("errors", []):
        console.print(f"[red]错误[/red] {err}")
    console.print(
        f"vlm_calls={final.get('vlm_calls', 0)} 观察帧数={len(final.get('used_frame_ids', []))} "
        f"iterations={final.get('iteration', 0)} 耗时={time.perf_counter() - t0:.2f}s "
        f"输出目录={job_dir}"
    )
    if final.get("status") == "failed":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
