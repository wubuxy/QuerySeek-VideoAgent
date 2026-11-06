"""OpenAI 兼容 Chat Completions 视觉模型 Adapter（开发说明书 §6.3）。

- 密钥仅从环境变量读取（VLM_API_KEY / VLM_BASE_URL / VLM_MODEL），绝不打印、不入 Trace。
- 第一版只传图片帧（base64 data URL）+ 文字，不假定原生视频流输入（说明书 §6.3）。
- 超时、有限指数退避重试、严格 JSON 解析；图片原文与密钥不写入 Trace。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from ..schemas import TimeWindow, VideoFrame
from .interface import (
    FrameJudgment,
    InspectionResult,
    VLMError,
    WindowVerification,
)

# ---- 请求级网络观测（P1，DECISIONS #41）------------------------------------
# httpcore 在 TRACE(5) 级别记录网络事件（send_request_body.complete /
# receive_response_headers.started 等）；本 handler 只采集"当前线程正在执行的
# 请求"的事件（_trace_collector 线程局部），避免跨请求混杂，且不记录敏感内容。
_TRACE_LEVEL = 5
_trace_collector = threading.local()


class _HttpcoreTraceHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        events = getattr(_trace_collector, "events", None)
        if events is not None:
            events.append({
                "t_ms": round(record.created * 1000),
                "event": record.getMessage(),
            })


_httpcore_logger = logging.getLogger("httpcore")
_httpcore_logger.setLevel(_TRACE_LEVEL)
_httpcore_logger.propagate = False
if not any(isinstance(h, _HttpcoreTraceHandler) for h in _httpcore_logger.handlers):
    _httpcore_logger.addHandler(_HttpcoreTraceHandler())

_SYSTEM_PROMPT = (
    "你是视频时间定位系统的视觉判定模块。你的输出必须是合法 JSON，"
    "不输出任何解释文字、代码块标记或多余内容。"
)

_INSPECT_PROMPT = (
    "查询：{query}\n"
    "下面是按时间顺序排列的视频帧，每帧带有 frame_id 与时间戳（秒）。\n"
    "{frame_lines}\n"
    "判断哪些帧的画面内容与查询相关（画面能作为回答/定位查询的证据）。\n"
    '只输出 JSON 对象，格式：{{"relevant_frame_ids": ["frame_000001", ...], '
    '"reasons": {{"frame_000001": "一句话理由"}}, "summary": "总体一句话判断"}}'
)

_VERIFY_PROMPT = (
    "查询：{query}\n"
    "候选时间窗口：[{start:.2f}, {end:.2f}) 秒\n"
    "证据：{evidence}\n"
    "判断该窗口内容是否确实与查询相关。\n"
    '只输出 JSON 对象，格式：{{"verdict": "confirmed|refuted|insufficient", '
    '"confidence": 0.0 到 1.0 之间的小数, "reason": "一句话理由"}}'
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_content(data: Any) -> str:
    """从 OpenAI 兼容响应中稳健提取文本内容。

    - choices/message/content 可能为 list（多模态 parts 格式），全部兼容；
    - 结构异常返回空串（由调用方按 VLMError 处理），绝不抛 AttributeError。
    """
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    msg = first.get("message") if isinstance(first, dict) else None
    if isinstance(msg, list):  # 部分兼容实现把 message 直接组织为 parts 列表
        parts = msg
    elif isinstance(msg, dict):
        content = msg.get("content")
        parts = content if isinstance(content, list) else [content]
    else:
        return ""
    texts: list[str] = []
    for p in parts:
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, dict) and p.get("type") == "text":
            texts.append(str(p.get("text", "")))
    return "".join(texts)


def _response_types(data: Any) -> dict[str, str]:
    """响应关键字段的类型名（仅供观测日志诊断，不含任何响应文本）。"""
    out: dict[str, str] = {"data": type(data).__name__}
    if isinstance(data, dict):
        choices = data.get("choices")
        out["choices"] = type(choices).__name__
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            out["message"] = type(msg).__name__
            if isinstance(msg, dict):
                out["content"] = type(msg.get("content")).__name__
            if isinstance(msg, list):
                out["content"] = "list(parts)"
    return out


def _parse_json_strict(text: str) -> dict[str, Any]:
    """严格解析模型输出 JSON；容忍被代码块包裹的情况，其余一律报错。"""
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if not m:
            raise VLMError(f"VLM 输出不是 JSON: {text[:200]!r}")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise VLMError(f"VLM 输出 JSON 解析失败: {exc}; 原文片段: {text[:200]!r}")
    if not isinstance(obj, dict):
        raise VLMError("VLM 输出 JSON 不是对象")
    return obj


def _image_data_url(path: str) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class CompatibleAPIVLM:
    """兼容 OpenAI Chat Completions 的视觉模型服务。"""

    model_id: str
    max_frames_per_call: int

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_sec: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_frames_per_call: int = 8,
        max_output_tokens: int | None = None,
        trust_env: bool = False,
        connect_timeout_sec: float | None = None,
        read_timeout_sec: float | None = None,
        write_timeout_sec: float | None = None,
        pool_timeout_sec: float | None = None,
        http2: bool = False,
        max_keepalive_connections: int = 0,
        max_connections: int = 1,
        req_log_dir: str | None = None,
        client: httpx.Client | None = None,
        json_mode: bool = True,
        enable_thinking: bool | None = None,
    ):
        if not api_key:
            raise VLMError("VLM_API_KEY 未设置：请配置环境变量或使用 --mock")
        if not model:
            raise VLMError("VLM_MODEL 未设置：请配置环境变量或使用 --mock")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.temperature = temperature
        self.model_id = model
        self.max_frames_per_call = max_frames_per_call
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.enable_thinking = enable_thinking
        # 分项超时（P1，DECISIONS #41）：未显式给定时退回单一 timeout_sec
        self._timeout = httpx.Timeout(
            connect=connect_timeout_sec if connect_timeout_sec is not None else timeout_sec,
            read=read_timeout_sec if read_timeout_sec is not None else timeout_sec,
            write=write_timeout_sec if write_timeout_sec is not None else timeout_sec,
            pool=pool_timeout_sec if pool_timeout_sec is not None else timeout_sec,
        )
        self._timeout_info = {
            "connect": self._timeout.connect,
            "read": self._timeout.read,
            "write": self._timeout.write,
            "pool": self._timeout.pool,
        }
        self.req_log_dir = Path(req_log_dir) if req_log_dir else None
        if self.req_log_dir:
            self.req_log_dir.mkdir(parents=True, exist_ok=True)
        # trust_env=False：不读系统代理（WinINET/环境变量）。百炼为国内服务应直连；
        # 本地代理节点不稳时会把请求挂起至超时（评测曾因此"卡死"，DECISIONS #40）。
        self._client = client or httpx.Client(
            timeout=self._timeout,
            trust_env=trust_env,
            http2=http2,
            limits=httpx.Limits(
                max_keepalive_connections=max_keepalive_connections,
                max_connections=max_connections,
            ),
        )
        self.total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_latency_ms: float = 0.0

    # -- 内部 ----------------------------------------------------------------

    def _chat(self, user_parts: list[dict[str, Any]], max_retries: int | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_parts},
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.max_output_tokens:
            payload["max_tokens"] = self.max_output_tokens
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        retries = self.max_retries if max_retries is None else max_retries
        n_images = sum(1 for p in user_parts
                       if isinstance(p, dict) and p.get("type") == "image_url")
        body_bytes = len(json.dumps(payload))
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            request_id = uuid.uuid4().hex[:12]
            events: list[dict[str, Any]] = []
            _trace_collector.events = events
            t0 = time.monotonic()
            status: int | None = None
            exc_type: str | None = None
            resp_types: dict[str, str] | None = None
            try:
                resp = self._client.post(url, json=payload, headers=headers)
                status = resp.status_code
                resp.raise_for_status()
                data = resp.json()
                # 响应结构防御（P4，DECISIONS #43）：网关偶发返回非对象顶层 /
                # list 形态的 message/content / 非 dict 的 usage，一律按结构异常
                # 处理，绝不抛 'list' object has no attribute 'get' 一类裸异常。
                if not isinstance(data, dict):
                    raise VLMError(f"VLM 响应结构异常：顶层为 {type(data).__name__}")
                resp_types = _response_types(data)
                usage = data.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                self.last_usage = {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                }
                for k in self.total_usage:
                    self.total_usage[k] += self.last_usage[k]
                content = _extract_content(data)
                if not content:
                    raise VLMError("VLM 响应为空")
                self.last_latency_ms = (time.monotonic() - t0) * 1000.0
                return content
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                exc_type = type(exc).__name__
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                exc_type = type(exc).__name__
                code = exc.response.status_code
                if code not in (429, 500, 502, 503, 504):
                    raise VLMError(f"VLM API 错误 {code}: {exc.response.text[:300]}") from exc
            except VLMError as exc:
                last_exc = exc
                exc_type = "VLMError"
                raise
            finally:
                _trace_collector.events = None
                self._log_request(request_id=request_id, attempt=attempt, events=events,
                                  t0=t0, status=status, exc_type=exc_type,
                                  n_images=n_images, body_bytes=body_bytes,
                                  resp_types=resp_types)
            if attempt < retries:
                time.sleep(min(0.5 * (2**attempt), 8.0))
        raise VLMError(f"VLM API 调用失败（重试 {retries} 次后）: {last_exc}") from last_exc

    def _log_request(
        self,
        request_id: str,
        attempt: int,
        events: list[dict[str, Any]],
        t0: float,
        status: int | None,
        exc_type: str | None,
        n_images: int,
        body_bytes: int,
        resp_types: dict[str, str] | None = None,
    ) -> None:
        """请求级观测落盘（P1）：只记元信息与网络事件，不含密钥与图片内容。

        resp_types：响应字段的类型名（choices/message/content），仅用于诊断
        网关响应结构漂移，不含任何响应文本。
        """
        if not self.req_log_dir:
            return
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request_id,
            "attempt": attempt,
            "model": self.model,
            "n_images": n_images,
            "body_bytes": body_bytes,
            "timeout": self._timeout_info,
            "duration_ms": round((time.monotonic() - t0) * 1000, 1),
            "status": status,
            "exception": exc_type,
            "resp_types": resp_types,
            "trace": events,
        }
        with (self.req_log_dir / "requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- 契约实现 ------------------------------------------------------------

    def inspect_frames(
        self, query: str, frames: list[VideoFrame], context: str = ""
    ) -> InspectionResult:
        if not frames:
            raise VLMError("inspect_frames: 帧列表为空")
        relevant: set[str] = set()
        judgments: list[FrameJudgment] = []
        summaries: list[str] = []
        for i in range(0, len(frames), self.max_frames_per_call):
            chunk = frames[i : i + self.max_frames_per_call]
            frame_lines = "\n".join(f"- {f.frame_id} @ {f.timestamp_sec:.2f}s" for f in chunk)
            parts: list[dict[str, Any]] = [
                {"type": "text", "text": _INSPECT_PROMPT.format(query=query, frame_lines=frame_lines)}
            ]
            for f in chunk:
                parts.append({"type": "image_url", "image_url": {"url": _image_data_url(f.image_path)}})
            obj = _parse_json_strict(self._chat(parts))
            ids = obj.get("relevant_frame_ids") or []
            # 结构防御（P4，DECISIONS #43）：模型偶发把相关帧/理由字段写成
            # 非预期类型（reasons 写成 list 曾致 E3 崩溃 'list' object has no
            # attribute 'get'）。非预期类型一律按空处理，不崩溃。
            if not isinstance(ids, list):
                ids = []
            valid_ids = {f.frame_id for f in chunk}
            for fid in ids:
                if isinstance(fid, str) and fid in valid_ids:
                    relevant.add(fid)
            reasons = obj.get("reasons") or {}
            if not isinstance(reasons, dict):
                reasons = {}
            for f in chunk:
                judgments.append(
                    FrameJudgment(
                        frame_id=f.frame_id,
                        relevant=f.frame_id in relevant,
                        reason=str(reasons.get(f.frame_id, "")),
                    )
                )
            if obj.get("summary"):
                summaries.append(str(obj["summary"]))
        return InspectionResult(
            relevant_frame_ids=sorted(relevant),
            judgments=judgments,
            summary=" | ".join(summaries),
        )

    def verify_window(
        self, query: str, window: TimeWindow, evidence: list[dict]
    ) -> WindowVerification:
        text = _VERIFY_PROMPT.format(
            query=query,
            start=window.start_sec,
            end=window.end_sec,
            evidence=json.dumps(evidence, ensure_ascii=False)[:2000],
        )
        obj = _parse_json_strict(self._chat([{"type": "text", "text": text}]))
        verdict = str(obj.get("verdict", "insufficient"))
        if verdict not in ("confirmed", "refuted", "insufficient"):
            verdict = "insufficient"
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return WindowVerification(
            verdict=verdict,
            confidence=max(0.0, min(1.0, conf)),
            reason=str(obj.get("reason", "")),
        )

    def verify_span(self, query: str, window: TimeWindow,
                    frames: list[VideoFrame], envelope: TimeWindow) -> dict[str, Any]:
        """Visual boundary verification; images and timestamps share exact order."""
        if not frames or len(frames) > self.max_frames_per_call:
            raise ValueError("Invalid visual span frame budget")
        prompt = (
            f"Query: {query}\nCandidate: [{window.start_sec}, {window.end_sec}) seconds. "
            f"Observed envelope: [{envelope.start_sec}, {envelope.end_sec}) seconds.\n"
            "Images below are chronological samples, NOT a continuous video. "
            "Distinguish the requested action from object/person presence. Do not invent "
            "motion across gaps. You may expand OR contract the candidate within the "
            "observed envelope. If the action or its boundaries are not supported, "
            "return insufficient. Return JSON with verdict (confirmed/refuted/insufficient), "
            "confidence (0..1), start_sec, end_sec (null when uncertain), "
            "evidence_frame_ids (visible supporting frame IDs), reason.\n"
            + "\n".join(f"{f.frame_id} @ {f.timestamp_sec:.3f}s" for f in frames)
        )
        parts = [{"type": "text", "text": prompt}]
        parts.extend({"type": "image_url", "image_url": {"url": _image_data_url(f.image_path)}}
                     for f in frames)
        return _parse_json_strict(self._chat(parts))

    def close(self) -> None:
        self._client.close()


def env_vlm_kwargs() -> dict[str, Any]:
    """从环境变量读取 VLM 配置（不读取密钥内容之外的任何东西）。"""
    return {
        "api_key": os.environ.get("VLM_API_KEY", ""),
        "base_url": os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1"),
        "model": os.environ.get("VLM_MODEL", ""),
    }
