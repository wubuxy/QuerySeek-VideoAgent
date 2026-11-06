"""VLM Adapter 测试：Mock 确定性 + OpenAI 兼容 API（httpx MockTransport，无真实调用）。"""

from __future__ import annotations

import json

import httpx
import pytest

from queryseek.models.compatible_api import CompatibleAPIVLM
from queryseek.models.interface import VLMError
from queryseek.models.mock import MockVLM
from queryseek.schemas import TimeWindow, VideoFrame

import pytest as _pytest


@_pytest.fixture()
def frames(tmp_path) -> list[VideoFrame]:
    """真实图片文件（Adapter 会读取字节；MockTransport 不校验内容）。"""
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    b.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    return [
        VideoFrame(frame_id="frame_000001", timestamp_sec=0.0, image_path=str(a)),
        VideoFrame(frame_id="frame_000002", timestamp_sec=12.0, image_path=str(b)),
    ]


HINTS = [
    {"start_sec": 0.0, "end_sec": 12.0, "label": "intro welcome 欢迎"},
    {"start_sec": 12.0, "end_sec": 24.0, "label": "architecture 系统架构 讲解"},
]


class TestMockVLM:
    def test_deterministic_match(self, frames):
        vlm = MockVLM(HINTS)
        r1 = vlm.inspect_frames("寻找讲解系统架构的片段", frames)
        r2 = vlm.inspect_frames("寻找讲解系统架构的片段", frames)
        assert r1.model_dump() == r2.model_dump()  # 确定性
        assert r1.relevant_frame_ids == ["frame_000002"]

    def test_no_hints_never_relevant(self, frames):
        vlm = MockVLM(None)
        r = vlm.inspect_frames("任何查询", frames)
        assert r.relevant_frame_ids == []

    def test_verify_window(self):
        vlm = MockVLM(HINTS)
        v = vlm.verify_window("寻找讲解系统架构的片段", TimeWindow(start_sec=12.0, end_sec=24.0), [])
        assert v.verdict == "confirmed"
        v2 = vlm.verify_window("寻找讲解系统架构的片段", TimeWindow(start_sec=0.0, end_sec=10.0), [])
        assert v2.verdict in ("refuted", "insufficient")


def _transport_with(response_payload: dict, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=response_payload)

    return httpx.MockTransport(handler)


def _ok_response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestCompatibleAPIVLM:
    def _vlm(self, transport) -> CompatibleAPIVLM:
        return CompatibleAPIVLM(
            api_key="test-key", base_url="https://api.example.com/v1",
            model="test-vlm", client=httpx.Client(transport=transport),
        )

    def test_inspect_frames_parses_json(self, frames):
        payload = json.dumps({"relevant_frame_ids": ["frame_000002"], "reasons": {}, "summary": "s"})
        vlm = self._vlm(_transport_with(_ok_response(payload)))
        r = vlm.inspect_frames("query", frames)
        assert r.relevant_frame_ids == ["frame_000002"]
        assert vlm.total_usage["prompt_tokens"] == 10

    def test_invalid_json_raises_vlm_error(self, frames):
        vlm = self._vlm(_transport_with(_ok_response("这不是 JSON")))
        with pytest.raises(VLMError):
            vlm.inspect_frames("query", frames)

    def test_retry_then_success(self, frames):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json=_ok_response('{"relevant_frame_ids": [], "summary": "ok"}'))

        vlm = self._vlm(httpx.MockTransport(handler))
        r = vlm.inspect_frames("query", frames)
        assert r.relevant_frame_ids == []
        assert calls["n"] == 3  # 2 次失败后第 3 次成功

    def test_exhaust_retries_raises(self, frames):
        vlm = self._vlm(_transport_with({"error": "boom"}, status=500))
        vlm.max_retries = 1
        with pytest.raises(VLMError):
            vlm.inspect_frames("query", frames)

    def test_missing_key_raises(self):
        with pytest.raises(VLMError):
            CompatibleAPIVLM(api_key="", base_url="https://x/v1", model="m")

    def test_verify_window_json(self):
        payload = json.dumps({"verdict": "confirmed", "confidence": 0.9, "reason": "yes"})
        vlm = self._vlm(_transport_with(_ok_response(payload)))
        v = vlm.verify_window("q", TimeWindow(start_sec=0.0, end_sec=5.0), [{"k": "v"}])
        assert v.verdict == "confirmed" and v.confidence == 0.9

    # ---- 响应结构防御（DECISIONS #43：'list' object has no attribute 'get'）----

    def test_content_as_parts_list_extracted(self, frames):
        """多模态兼容实现可能把 message.content 组织为 parts 列表。"""
        payload = {
            "choices": [{"message": {"content": [
                {"type": "text", "text": '{"verdict": "refuted", "confidence": 0.2, "reason": "x"}'},
            ]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        vlm = self._vlm(_transport_with(payload))
        v = vlm.verify_window("q", TimeWindow(start_sec=0.0, end_sec=5.0), [])
        assert v.verdict == "refuted" and v.confidence == 0.2

    def test_message_as_list_handled(self):
        """部分兼容实现把 message 直接组织为 parts 列表：不崩溃、按空处理。"""
        payload = {"choices": [{"message": [{"type": "text", "text": "not-json"}]}]}
        vlm = self._vlm(_transport_with(payload))
        with pytest.raises(VLMError):
            vlm.verify_window("q", TimeWindow(start_sec=0.0, end_sec=5.0), [])

    def test_top_level_list_raises_vlm_error(self):
        """网关偶发返回 JSON 数组顶层：转 VLMError 而非裸 AttributeError。"""
        vlm = self._vlm(_transport_with([]))
        with pytest.raises(VLMError, match="结构异常"):
            vlm.verify_window("q", TimeWindow(start_sec=0.0, end_sec=5.0), [])

    def test_usage_as_list_tolerated(self, frames):
        """usage 字段为非 dict（如 list）时按 0 计，不崩溃。"""
        payload = json.dumps({"relevant_frame_ids": ["frame_000001"], "reasons": {}, "summary": "s"})
        resp = _ok_response(payload)
        resp["usage"] = [1, 2, 3]
        vlm = self._vlm(_transport_with(resp))
        r = vlm.inspect_frames("query", frames)
        assert r.relevant_frame_ids == ["frame_000001"]
        assert vlm.total_usage["total_tokens"] == 0

    def test_reasons_as_list_tolerated(self, frames):
        """模型偶发把 reasons 写成 list（E3 崩溃根因候选）：忽略而非崩溃。"""
        payload = json.dumps({"relevant_frame_ids": ["frame_000002"],
                              "reasons": ["frame_000002 与查询相关"],
                              "summary": "s"})
        vlm = self._vlm(_transport_with(_ok_response(payload)))
        r = vlm.inspect_frames("query", frames)
        assert r.relevant_frame_ids == ["frame_000002"]
        assert r.judgments[1].relevant is True
        assert r.judgments[1].reason == ""  # 非 dict reasons 按无理由处理

    def test_relevant_ids_non_list_tolerated(self, frames):
        """relevant_frame_ids 为非列表（字符串/数字）时按无相关帧处理。"""
        payload = json.dumps({"relevant_frame_ids": "frame_000002", "reasons": {}, "summary": "s"})
        vlm = self._vlm(_transport_with(_ok_response(payload)))
        r = vlm.inspect_frames("query", frames)
        assert r.relevant_frame_ids == []

    def test_relevant_ids_int_items_ignored(self, frames):
        """列表内的非字符串项（模型回传数字序号）不参与匹配、不崩溃。"""
        payload = json.dumps({"relevant_frame_ids": [1, "frame_000002"], "reasons": {}, "summary": "s"})
        vlm = self._vlm(_transport_with(_ok_response(payload)))
        r = vlm.inspect_frames("query", frames)
        assert r.relevant_frame_ids == ["frame_000002"]
