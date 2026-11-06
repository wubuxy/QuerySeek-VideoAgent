"""VLM Adapter 契约（开发说明书 §6.3）。

- API 与 Mock 使用同一契约；调用结果严格解析为结构化对象（Pydantic），
  不接受模糊自由文本作为时间区间。
- VLM 只看必要帧；所有请求记录帧数、耗时、用量（若 API 返回）及失败原因。
- 图片原文与密钥不写入 Trace。
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from ..schemas import TimeWindow, VideoFrame

__all__ = [
    "VLMError",
    "FrameJudgment",
    "InspectionResult",
    "WindowVerification",
    "VLMAdapter",
]


class VLMError(RuntimeError):
    """VLM 调用失败（超时、非 JSON、限流、缺密钥等）统一错误类型。"""


class FrameJudgment(BaseModel):
    frame_id: str
    relevant: bool
    reason: str = ""


class InspectionResult(BaseModel):
    """inspect_frames 的严格结构化结果。"""

    relevant_frame_ids: list[str] = Field(default_factory=list)
    judgments: list[FrameJudgment] = Field(default_factory=list)
    summary: str = ""


class WindowVerification(BaseModel):
    """verify_window 的严格结构化结果（P3 使用，先定义契约）。"""

    verdict: str = Field(
        description="confirmed | refuted | insufficient"
    )
    confidence: float = 0.0
    reason: str = ""


class VLMAdapter(Protocol):
    """所有 VLM 实现（真实 API / Mock）必须满足的契约。"""

    model_id: str
    max_frames_per_call: int

    def inspect_frames(
        self, query: str, frames: list[VideoFrame], context: str = ""
    ) -> InspectionResult: ...

    def verify_window(
        self, query: str, window: TimeWindow, evidence: list[dict]
    ) -> WindowVerification: ...

    def verify_span(
        self, query: str, window: TimeWindow, frames: list[VideoFrame],
        envelope: TimeWindow,
    ) -> dict: ...
