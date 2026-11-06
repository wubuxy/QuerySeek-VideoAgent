"""配置加载：configs/default.yaml + 环境变量覆盖。

密钥仅通过环境变量读取（VLM_*、ASR_*），从不写入配置对象之外的任何日志/文件。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from . import PIPELINE_VERSION

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class ProjectConfig(BaseModel):
    name: str = "QuerySeek-VideoAgent"
    cache_root: Path
    output_root: Path
    data_root: Path


class VideoConfig(BaseModel):
    max_duration_sec: float = 1800.0
    max_file_size_mb: float = 1500.0
    max_frame_side_px: int = 512
    coarse_fps: float = 0.5
    frame_sampling: str = "scene_or_uniform"
    scene_threshold: float = 27.0


class AsrConfig(BaseModel):
    backend: str = "auto"
    local_compute_type: str = "int8"
    model: str = "small"


class EmbeddingConfig(BaseModel):
    backend: str = "openclip"  # openclip | mock | none
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"  # open_clip 预训练权重名（权重许可按模型单独核对）
    device: str = "cpu"


class SemanticRetrievalConfig(BaseModel):
    top_k_frames: int = 24
    min_sim: float = 0.15
    cluster_gap_sec: float = 4.0
    window_pad_sec: float = 1.5


class FusionConfig(BaseModel):
    rrf_k: int = 60
    nms_iou: float = 0.5


class WindowGrowthConfig(BaseModel):
    """证据驱动的窗口生长（EXPERIMENTS.md §6 诊断 3b：62% 有重叠未命中 =
    "窗口在 GT 内但过窄"）。默认关——开启是 v2 实验变量，保持与已报告
    结果的可比性。"""
    enabled: bool = False
    max_steps_per_side: int = 2  # 每侧最多向外生长的相邻索引帧数
    min_relevant_rate: float = 0.5  # 粗观察相关率低于此值不生长


class AdaptiveRetrievalConfig(BaseModel):
    coarse_fps: float = 1.0  # 细化轮次的局部加密抽帧密度
    coarse_frames_per_window: int = 4  # 粗观察阶段每候选窗口的帧数上限
    uncertainty_threshold: float = 0.5
    min_observations_per_window: int = 1
    uncertainty_width_reference_sec: float = 12.0  # 宽度 >= 该值视为"边界待细化"
    grow: WindowGrowthConfig = Field(default_factory=WindowGrowthConfig)


class SubtitleRetrievalConfig(BaseModel):
    merge_gap_sec: float = 1.5
    min_score: float = 0.1
    merge_ratio: float = 0.5  # 相邻段分数低于组内最高分 * merge_ratio 时不合并（话题连续性启发式）


class UniformRetrievalConfig(BaseModel):
    cluster_gap_sec: float = 4.0
    window_pad_sec: float = 1.5
    relevant_score: float = 0.5


class RetrievalConfig(BaseModel):
    top_k_windows: int = 5
    min_window_sec: float = 3.0
    max_window_sec: float = 120.0
    frame_budget: int = 16
    max_refinement_rounds: int = 2
    fusion: str = "rrf"
    subtitle: SubtitleRetrievalConfig = Field(default_factory=SubtitleRetrievalConfig)
    uniform: UniformRetrievalConfig = Field(default_factory=UniformRetrievalConfig)
    semantic: SemanticRetrievalConfig = Field(default_factory=SemanticRetrievalConfig)
    fusion_config: FusionConfig = Field(default_factory=FusionConfig)
    adaptive: AdaptiveRetrievalConfig = Field(default_factory=AdaptiveRetrievalConfig)


class VlmConfig(BaseModel):
    timeout_sec: float = 60.0
    max_retries: int = 3
    max_frames_per_call: int = 8
    temperature: float = 0.0
    max_output_tokens: int = 512  # 输出上限：判定 JSON 足够，且约束 API 成本（DECISIONS #39）
    enable_thinking: bool | None = None  # Provider-specific; omit unless explicitly configured
    trust_env: bool = False  # False=API 直连不走系统代理（DECISIONS #40）
    # P1 网络诊断（DECISIONS #41）：分项超时 + 连接池收紧 + 请求级观测
    connect_timeout_sec: float = 8.0
    read_timeout_sec: float = 45.0
    write_timeout_sec: float = 20.0
    pool_timeout_sec: float = 5.0
    http2: bool = False
    max_keepalive_connections: int = 0
    max_connections: int = 1
    req_log_dir: str | None = None  # 请求日志目录（无密钥、无图片内容）


class AgentConfig(BaseModel):
    max_iterations: int = 4
    max_vlm_calls: int = 4
    wall_time_limit_sec: float = 180.0
    allow_retry: bool = True
    checkpoint: str = "sqlite"


class ExportConfig(BaseModel):
    output_format: str = "mp4"
    target_duration_tolerance_sec: float = 3.0
    fps: int = 30
    codec: str = "libx264"
    crf: int = 23
    audio_codec: str = "aac"


class LoggingConfig(BaseModel):
    redact_secrets: bool = True
    trace: bool = True


class Config(BaseModel):
    project: ProjectConfig
    video: VideoConfig = Field(default_factory=VideoConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def cache_root(self) -> Path:
        return self.project.cache_root

    @property
    def output_root(self) -> Path:
        return self.project.output_root

    @property
    def data_root(self) -> Path:
        return self.project.data_root

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    @classmethod
    def load(cls, yaml_path: Path | None = None, apply_env: bool = True) -> "Config":
        """加载默认配置并应用环境变量覆盖（VLM_*、ASR_*、QUERYSEEK_*）。"""
        path = yaml_path or _PACKAGE_ROOT / "configs" / "default.yaml"
        cfg = cls.from_yaml(path)
        if apply_env:
            cfg = _apply_env(cfg)
        return cfg

    def resolved(self) -> "Config":
        """把相对路径解析为基于仓库根目录的绝对路径。"""
        raw = self.model_dump()
        for key in ("cache_root", "output_root", "data_root"):
            p = Path(raw["project"][key])
            if not p.is_absolute():
                raw["project"][key] = str((_PACKAGE_ROOT / p).resolve())
        return Config.model_validate(raw)

    def snapshot_for_cache(self) -> dict[str, Any]:
        """参与核心索引缓存键的配置子集（不含路径）。

        注意：嵌入后端**不**在这里——嵌入是可选附加工件，由
        embeddings/embedding_meta.json 单独记录后端与模型，使用前单独校验
        （check_backend），避免"ingest --mock 与 search --mock"配置不一致
        导致整索引缓存失效（docs/DECISIONS.md）。
        """
        return {
            "pipeline": PIPELINE_VERSION,
            "scene_threshold": self.video.scene_threshold,
            "frame_sampling": self.video.frame_sampling,
            "coarse_fps": self.video.coarse_fps,
            "max_frame_side_px": self.video.max_frame_side_px,
            "asr_backend": self.asr.backend,
            "asr_model": self.asr.model,
        }


def _apply_env(cfg: Config) -> Config:
    raw = cfg.model_dump()
    env = os.environ
    for key in ("cache_root", "output_root", "data_root"):
        v = env.get(f"QUERYSEEK_{key.upper()}")
        if v:
            raw["project"][key] = v
    # 密钥不进入 config 对象：由模型工厂在需要时直接读环境变量。
    return Config.model_validate(raw)


