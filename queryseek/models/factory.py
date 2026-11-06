"""VLM 工厂：--mock 或缺少密钥时用确定性 Mock，否则用 OpenAI 兼容 API。"""

from __future__ import annotations

import os

from ..config import Config
from .compatible_api import CompatibleAPIVLM
from .interface import VLMAdapter
from .mock import ContentHint, MockVLM


def build_vlm(
    config: Config,
    use_mock: bool = False,
    hints: list[ContentHint] | None = None,
) -> VLMAdapter:
    key = os.environ.get("VLM_API_KEY", "")
    if use_mock or not key:
        return MockVLM(hints)
    return CompatibleAPIVLM(
        api_key=key,
        base_url=os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("VLM_MODEL", ""),
        timeout_sec=config.vlm.timeout_sec,
        max_retries=config.vlm.max_retries,
        temperature=config.vlm.temperature,
        max_frames_per_call=config.vlm.max_frames_per_call,
        max_output_tokens=config.vlm.max_output_tokens,
        enable_thinking=config.vlm.enable_thinking,
        trust_env=config.vlm.trust_env,
        connect_timeout_sec=config.vlm.connect_timeout_sec,
        read_timeout_sec=config.vlm.read_timeout_sec,
        write_timeout_sec=config.vlm.write_timeout_sec,
        pool_timeout_sec=config.vlm.pool_timeout_sec,
        http2=config.vlm.http2,
        max_keepalive_connections=config.vlm.max_keepalive_connections,
        max_connections=config.vlm.max_connections,
        req_log_dir=config.vlm.req_log_dir,
    )
