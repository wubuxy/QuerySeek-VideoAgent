"""核心数据结构（开发说明书 §6）。

时间契约：所有对外接口使用浮点秒；区间为 [start_sec, end_sec)；
必须满足 0 <= start < end <= duration（duration 在具体模型里以字段校验）。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "TimeWindow",
    "SubtitleSegment",
    "VideoFrame",
    "VideoAsset",
    "Evidence",
    "SearchResult",
    "ExportSegment",
    "ExportManifest",
    "RunSummary",
    "Scene",
]


class TimeWindow(BaseModel):
    """半开区间 [start_sec, end_sec)。"""

    start_sec: float
    end_sec: float
    score: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("start_sec", "end_sec")
    @classmethod
    def _finite_nonneg(cls, v: float) -> float:
        if not (v == v and v != float("inf") and v != float("-inf")):  # NaN/Inf 检查
            raise ValueError("时间必须是有限数值")
        if v < 0:
            raise ValueError(f"时间不能为负: {v}")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> "TimeWindow":
        if self.start_sec >= self.end_sec:
            raise ValueError(
                f"区间必须满足 start < end: [{self.start_sec}, {self.end_sec})"
            )
        return self

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def clamp_to(self, duration_sec: float) -> "TimeWindow":
        """裁剪到 [0, duration] 内；若裁剪后无效返回 None 由调用方处理。"""
        s = max(0.0, self.start_sec)
        e = min(duration_sec, self.end_sec)
        if s >= e:
            raise ValueError(f"窗口 [{s},{e}) 裁剪后为空（duration={duration_sec}）")
        return self.model_copy(update={"start_sec": s, "end_sec": e})


class SubtitleSegment(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    seg_id: str | None = None
    source: str = "srt"

    @model_validator(mode="after")
    def _valid(self) -> "SubtitleSegment":
        if self.start_sec >= self.end_sec:
            raise ValueError(f"字幕区间非法: [{self.start_sec}, {self.end_sec})")
        if not self.text.strip():
            raise ValueError("字幕文本不能为空")
        return self


class VideoFrame(BaseModel):
    frame_id: str
    timestamp_sec: float
    image_path: str
    shot_id: str | None = None
    source: str = "uniform"  # uniform | scene

    @field_validator("timestamp_sec")
    @classmethod
    def _finite_nonneg(cls, v: float) -> float:
        if not (v == v and v != float("inf") and v != float("-inf")) or v < 0:
            raise ValueError(f"帧时间戳非法: {v}")
        return v


class Scene(BaseModel):
    scene_id: str
    start_sec: float
    end_sec: float

    @model_validator(mode="after")
    def _valid(self) -> "Scene":
        if self.start_sec >= self.end_sec:
            raise ValueError(f"镜头区间非法: [{self.start_sec}, {self.end_sec})")
        return self


class VideoAsset(BaseModel):
    video_id: str
    path: str
    sha256: str
    duration_sec: float
    fps: float
    width: int
    height: int
    has_audio: bool = False
    rotation: float = 0.0

    @field_validator("path")
    @classmethod
    def _path_exists(cls, v: str) -> str:
        if not Path(v).is_file():
            raise ValueError(f"视频文件不存在: {v}")
        return v

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("sha256 必须是 64 位小写十六进制")
        return v


class Evidence(BaseModel):
    """统一证据结构：evidence_id/type/timestamp 或 window/content/source/model_id。"""

    evidence_id: str
    type: str  # asr | frame | vlm
    timestamp_sec: float | None = None
    window: TimeWindow | None = None
    content: str = ""
    source_file: str | None = None
    model_id: str | None = None
    score: float | None = None


class SearchResult(BaseModel):
    video_id: str
    query: str
    windows: list[TimeWindow] = Field(default_factory=list)
    method: str
    used_frame_ids: list[str] = Field(default_factory=list)
    vlm_calls: int = 0
    indexed_frames: int = 0  # E4 实际打分的全视频索引帧数，独立于 VLM 观察
    online_encoded_frames: int = 0  # E4 未命中嵌入缓存时，本次编码的图片数
    embedding_cache_hit: bool | None = None
    elapsed_ms: float = 0.0
    trace_path: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    budget_exhausted: bool = False
    # adaptive 策略的每轮细化日志（说明书 §7.3）：
    # {round, before_window:[s,e], sampled_frames:[ids], after_window:[s,e]|None,
    #  uncertainty, reason}
    refinement_rounds: list[dict] = Field(default_factory=list)

    @field_validator("windows")
    @classmethod
    def _windows_valid(cls, ws: list[TimeWindow]) -> list[TimeWindow]:
        for w in ws:
            w._ordered()  # 复用校验逻辑
        return ws


class ExportSegment(BaseModel):
    """manifest 中的一个导出片段：源区间与导出后区间的映射。"""

    index: int
    source_start_sec: float
    source_end_sec: float
    export_start_sec: float
    export_end_sec: float
    score: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    segment_file: str | None = None


class ExportManifest(BaseModel):
    job_id: str
    video_id: str
    source_path: str
    source_sha256: str
    query: str
    strategy: str
    created_at_utc: str
    output_file: str
    output_duration_sec: float
    segments: list[ExportSegment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    """run 命令的最终指标汇总（写入 metrics.json）。"""

    job_id: str
    status: str  # ok | no_result | failed
    strategy: str
    vlm_calls: int
    used_frames: int
    indexed_frames: int = 0
    online_encoded_frames: int = 0
    embedding_cache_hit: bool | None = None
    elapsed_sec: float
    target_seconds: float | None = None
    output_duration_sec: float | None = None
    exported_windows: int = 0
    budget_exhausted: bool = False
    warnings: list[str] = Field(default_factory=list)
    output_dir: str
