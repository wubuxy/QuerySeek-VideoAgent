"""VLM Adapter 层：统一契约（interface）+ 真实 API（compatible_api）+ 确定性 Mock。"""

from .factory import build_vlm
from .interface import (
    FrameJudgment,
    InspectionResult,
    VLMAdapter,
    VLMError,
    WindowVerification,
)
from .mock import ContentHint, MockVLM, load_hints_file

__all__ = [
    "build_vlm",
    "ContentHint",
    "FrameJudgment",
    "InspectionResult",
    "MockVLM",
    "VLMAdapter",
    "VLMError",
    "WindowVerification",
    "load_hints_file",
]
