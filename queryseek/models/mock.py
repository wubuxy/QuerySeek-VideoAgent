"""确定性 Mock VLM（开发说明书 §5：无密钥/无 GPU 时必须可用 --mock 跑通全流程）。

Mock 不做任何真实视觉理解：它根据可选的"内容提示"（content hints，由合成样本
生成脚本提供，形如 [start_sec, end_sec) -> label）做关键词匹配，模拟"帧所属片段
与查询相关"的判断。这使单测与 CI 完全确定、可离线复现，同时保证集成测试能检验
聚簇/合并等真实算法逻辑。

没有 hints 时一律判为不相关并给出说明——Mock 绝不假装看懂真实视频内容。
Mock 结果只用于验证流程，不当作真实算法准确率（AGENTS.md 规则 5）。
"""

from __future__ import annotations

from typing import Any

from ..hints import ContentHint, hint_at
from ..tokenize import tokenize_query
from ..schemas import TimeWindow, VideoFrame
from .interface import (
    FrameJudgment,
    InspectionResult,
    VLMAdapter,
    WindowVerification,
)


def load_hints_file(path) -> list[ContentHint]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    segments = raw.get("segments", raw if isinstance(raw, list) else [])
    hints = []
    for s in segments:
        hints.append(
            {
                "start_sec": float(s["start_sec"]),
                "end_sec": float(s["end_sec"]),
                "label": str(s.get("label", "")).lower(),
            }
        )
    return hints


class MockVLM:
    """关键词匹配式确定性 Mock。"""

    model_id = "mock-vlm-1"
    max_frames_per_call = 8

    def __init__(self, hints: list[ContentHint] | None = None):
        self.hints = hints or []
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _matches(self, query: str, label: str) -> bool:
        tokens = tokenize_query(query)
        if not tokens:
            return False
        return any(t in label for t in tokens)

    def inspect_frames(
        self, query: str, frames: list[VideoFrame], context: str = ""
    ) -> InspectionResult:
        if not self.hints:
            return InspectionResult(
                summary="mock: 无内容提示（content hints），所有帧判为不相关；Mock 不模拟真实视觉理解"
            )
        relevant: list[str] = []
        judgments: list[FrameJudgment] = []
        for f in frames:
            label = hint_at(self.hints, f.timestamp_sec)
            if label and self._matches(query, label):
                relevant.append(f.frame_id)
                judgments.append(
                    FrameJudgment(frame_id=f.frame_id, relevant=True, reason=f"mock hint 命中: {label}")
                )
            else:
                judgments.append(
                    FrameJudgment(
                        frame_id=f.frame_id,
                        relevant=False,
                        reason="mock hint 未命中" if self.hints else "mock: 无内容提示",
                    )
                )
        return InspectionResult(
            relevant_frame_ids=relevant,
            judgments=judgments,
            summary=f"mock: {len(relevant)}/{len(frames)} 帧相关（基于内容提示关键词匹配，非真实视觉）",
        )

    def verify_span(self, query, window, frames, envelope):
        """Mock checks plumbing only; never invent visual boundaries."""
        return {"verdict": "insufficient", "confidence": 0.0,
                "start_sec": None, "end_sec": None,
                "evidence_frame_ids": [], "reason": "Mock: boundaries unmeasured"}

    def verify_window(
        self, query: str, window: TimeWindow, evidence: list[dict]
    ) -> WindowVerification:
        if not self.hints:
            return WindowVerification(verdict="insufficient", confidence=0.0, reason="mock: 无内容提示")
        # 窗口内命中 hint 的时长占比作为确定性"置信度"
        total = window.end_sec - window.start_sec
        hit = 0.0
        step = 0.5
        t = window.start_sec
        while t < window.end_sec:
            label = hint_at(self.hints, t)
            if label and self._matches(query, label):
                hit += step
            t += step
        ratio = min(hit / total, 1.0) if total > 0 else 0.0
        verdict = "confirmed" if ratio >= 0.5 else ("insufficient" if ratio > 0 else "refuted")
        return WindowVerification(
            verdict=verdict,
            confidence=round(ratio, 3),
            reason=f"mock: 窗口内内容提示命中占比 {ratio:.2f}",
        )
