"""ingest 编排：视频 -> 结构化时序索引（开发说明书 §7.1）。

核心索引缓存键 = video_sha256 + pipeline_version + 场景/抽帧/ASR 配置
（cache.py）；配置变化不会命中旧缓存（AGENTS.md 规则 7）。
帧嵌入是可选附加工件：由 embeddings/embedding_meta.json 单独记录后端与模型，
核心索引命中后仍会按当前配置补齐/更新嵌入（后端与模型单独校验）。
无音轨/无字幕时仍支持视觉搜索，不崩溃（说明书 §12.1）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .. import PIPELINE_VERSION
from ..config import Config
from ..logging_utils import TraceWriter
from ..schemas import Scene, SubtitleSegment, VideoAsset, VideoFrame
from .cache import cache_dir_for, cache_hit, compute_cache_key, read_json, write_json_atomic
from .ffprobe import asset_from_probe
from .frames import load_frames_index, save_frames_index, select_frames
from .registry import register_video, sanitize_video_id
from .scenes import detect_scenes
from .subtitles import load_srt, load_transcript, save_transcript, try_local_asr

log = logging.getLogger(__name__)


@dataclass
class VideoIndex:
    """一个视频的完整离线索引。"""

    asset: VideoAsset
    cache_dir: Path
    scenes: list[Scene] = field(default_factory=list)
    frames: list[VideoFrame] = field(default_factory=list)
    subtitles: list[SubtitleSegment] = field(default_factory=list)
    content_hints: list[dict] = field(default_factory=list)
    cache_key: str = ""
    cache_hit: bool = False

    @property
    def video_id(self) -> str:
        return self.asset.video_id


def _load_scenes(cache_dir: Path) -> list[Scene]:
    import json as _json

    raw = (cache_dir / "scenes.json").read_text(encoding="utf-8")
    return [Scene.model_validate(s) for s in _json.loads(raw)]


def _load_hints(cache_dir: Path) -> list[dict]:
    p = cache_dir / "content_hints.json"
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("segments", [])


def _write_index_meta(cache_dir: Path, meta: dict) -> None:
    write_json_atomic(cache_dir / "index_meta.json", meta)


def _ensure_embeddings(index: VideoIndex, config: Config, trace: TraceWriter | None) -> None:
    """按当前配置补齐/更新帧嵌入（openclip 后端）。

    - mock/none 后端：检索时按需计算，无需落盘。
    - openclip：embedding_meta.json 与当前 model/pretrained 一致则跳过，
      否则（首次 / 模型变化）重新编码并缓存。
    """
    if config.embedding.backend != "openclip":
        return
    existing = read_json(index.cache_dir / "embedding_meta.json")
    want = {
        "backend": "openclip",
        "model_name": config.embedding.model_name,
        "pretrained": config.embedding.pretrained,
        "encoder_version": "2",
    }
    if existing and all(existing.get(k) == v for k, v in want.items()):
        return  # 已缓存且配置一致
    from ..embeddings.cache import save_embeddings
    from ..embeddings.factory import build_embedder

    embedder = build_embedder(config, use_mock=False)
    if embedder is None:
        log.warning(
            "OpenCLIP 不可用（未安装或权重下载失败），跳过嵌入；"
            "semantic/adaptive 可用 --mock 运行"
        )
        return
    try:
        mat = embedder.encode_images([f.image_path for f in index.frames])
        save_embeddings(
            index.cache_dir, mat, [f.frame_id for f in index.frames],
            {
                "backend": "openclip",
                "model_id": embedder.model_id,
                "model_name": config.embedding.model_name,
                "pretrained": config.embedding.pretrained,
                "encoder_version": "2",
            },
        )
        if trace:
            trace.event("embeddings_done", n=len(index.frames), model=embedder.model_id)
    except Exception as exc:  # noqa: BLE001 — 嵌入失败不阻塞 ingest
        log.warning("OpenCLIP 嵌入失败，跳过（semantic 策略将不可用）: %s", exc)


def ingest(
    video_path: str | Path,
    video_id: str | None = None,
    srt_path: str | Path | None = None,
    config: Config | None = None,
    use_cache: bool = True,
    trace: TraceWriter | None = None,
) -> VideoIndex:
    config = (config or Config.load()).resolved()
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    video_id = sanitize_video_id(video_id or video_path.stem)
    asset = asset_from_probe(video_id, video_path)

    # 输入守卫（说明书 §12）：时长与文件大小上限
    if asset.duration_sec <= 0:
        raise RuntimeError(f"视频时长无效: {asset.duration_sec}")
    if asset.duration_sec > config.video.max_duration_sec:
        raise RuntimeError(
            f"视频时长 {asset.duration_sec:.0f}s 超过上限 {config.video.max_duration_sec}s"
        )
    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > config.video.max_file_size_mb:
        raise RuntimeError(
            f"视频大小 {size_mb:.0f}MB 超过上限 {config.video.max_file_size_mb}MB"
        )

    snapshot = config.snapshot_for_cache()
    cache_key = compute_cache_key(asset.sha256, snapshot)
    cache_dir = cache_dir_for(config.cache_root, asset.sha256)
    if trace:
        trace.event(
            "ingest_start", video_id=video_id, sha256=asset.sha256,
            duration_sec=asset.duration_sec, cache_key=cache_key[:12],
        )

    # 缓存命中：直接加载核心索引，再按当前配置补齐嵌入
    if use_cache and cache_hit(config.cache_root, asset.sha256, snapshot):
        if trace:
            trace.event("ingest_cache_hit", cache_key=cache_key[:12])
        index = VideoIndex(
            asset=asset,
            cache_dir=cache_dir,
            scenes=_load_scenes(cache_dir),
            frames=load_frames_index(cache_dir / "frames.jsonl"),
            subtitles=load_transcript(cache_dir / "transcript.jsonl"),
            content_hints=_load_hints(cache_dir),
            cache_key=cache_key,
            cache_hit=True,
        )
        _ensure_embeddings(index, config, trace)
        register_video(config.data_root, asset)
        return index

    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. 镜头切分（PySceneDetect）
    scenes = detect_scenes(
        video_path, threshold=config.video.scene_threshold, duration_sec=asset.duration_sec
    )
    if trace:
        trace.event("scenes_detected", n=len(scenes))

    # 2. 抽帧（均匀网格 + 无覆盖镜头的代表帧）
    frames = select_frames(
        video_path,
        cache_dir / "frames",
        scenes,
        asset.duration_sec,
        config.video.coarse_fps,
        mode=config.video.frame_sampling,
        max_side=config.video.max_frame_side_px,
    )
    save_frames_index(frames, cache_dir / "frames.jsonl")
    if trace:
        trace.event("frames_extracted", n=len(frames), coarse_fps=config.video.coarse_fps)

    # 3. 字幕：用户 SRT -> 本地 ASR（已安装且有音轨）-> 空
    subtitles: list[SubtitleSegment] = []
    if srt_path:
        subtitles = load_srt(srt_path, duration_sec=asset.duration_sec)
    elif config.asr.backend in ("auto", "whisper") and asset.has_audio:
        subtitles = try_local_asr(video_path, config.asr.model, config.asr.local_compute_type) or []
    save_transcript(subtitles, cache_dir / "transcript.jsonl")
    if trace:
        trace.event("subtitles_ready", n=len(subtitles), source=srt_path and "srt" or "asr")

    # 4. 内容提示（可选，供 Mock VLM 使用；来自 <视频同名>.labels.json）
    labels_path = video_path.parent / (video_path.stem + ".labels.json")
    hints: list[dict] = []
    if labels_path.is_file():
        try:
            raw = json.loads(labels_path.read_text(encoding="utf-8"))
            hints = raw.get("segments", [])
            write_json_atomic(cache_dir / "content_hints.json", raw)
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("labels 文件解析失败，忽略: %s", exc)

    # 5. 元数据与缓存键
    _write_index_meta(
        cache_dir,
        {
            "cache_key": cache_key,
            "pipeline_version": PIPELINE_VERSION,
            "video_id": video_id,
            "sha256": asset.sha256,
            "duration_sec": asset.duration_sec,
            "n_scenes": len(scenes),
            "n_frames": len(frames),
            "n_subtitles": len(subtitles),
            "config_snapshot": snapshot,
        },
    )
    write_json_atomic(cache_dir / "metadata.json", asset.model_dump())
    write_json_atomic(cache_dir / "scenes.json", [s.model_dump() for s in scenes])

    # 6. 注册
    register_video(config.data_root, asset)
    if trace:
        trace.event("ingest_done", cache_key=cache_key[:12], n_frames=len(frames))

    index = VideoIndex(
        asset=asset,
        cache_dir=cache_dir,
        scenes=scenes,
        frames=frames,
        subtitles=subtitles,
        content_hints=hints,
        cache_key=cache_key,
        cache_hit=False,
    )
    # 7. 帧嵌入（openclip 后端，失败不阻塞 ingest）
    _ensure_embeddings(index, config, trace)
    return index


def load_index(video_id: str, config: Config | None = None) -> VideoIndex:
    """从注册表 + 缓存加载已 ingest 的视频索引。"""
    config = (config or Config.load()).resolved()
    from .registry import resolve_video

    asset = resolve_video(config.data_root, video_id)
    snapshot = config.snapshot_for_cache()
    cache_key = compute_cache_key(asset.sha256, snapshot)
    cache_dir = cache_dir_for(config.cache_root, asset.sha256)
    if not cache_hit(config.cache_root, asset.sha256, snapshot):
        raise RuntimeError(
            f"视频 {video_id!r} 的缓存与当前配置不匹配或不存在，请重新 ingest"
        )
    return VideoIndex(
        asset=asset,
        cache_dir=cache_dir,
        scenes=_load_scenes(cache_dir),
        frames=load_frames_index(cache_dir / "frames.jsonl"),
        subtitles=load_transcript(cache_dir / "transcript.jsonl"),
        content_hints=_load_hints(cache_dir),
        cache_key=cache_key,
        cache_hit=True,
    )
