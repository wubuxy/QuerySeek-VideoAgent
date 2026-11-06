"""trace.jsonl 写入与密钥脱敏。

Trace 记录事件流（ingest/search/vlm/export/validate/error），用于复现与面试展示。
密钥与图片原文绝不写入 trace（开发说明书 §6.3）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_SECRET_ENV_KEYS = ("VLM_API_KEY", "ASR_API_KEY")


def secret_values() -> list[str]:
    vals = [os.environ.get(k, "") for k in _SECRET_ENV_KEYS]
    return [v for v in vals if v]


def redact(obj: Any, secrets: list[str] | None = None) -> Any:
    """递归替换对象中的密钥子串为 '<redacted>'。"""
    secrets = secrets or secret_values()
    if not secrets:
        return obj
    if isinstance(obj, str):
        for s in secrets:
            obj = obj.replace(s, "<redacted>")
        return obj
    if isinstance(obj, dict):
        return {k: redact(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v, secrets) for v in obj]
    return obj


class TraceWriter:
    """线程安全的 JSONL trace 写入器。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def event(self, name: str, **payload: Any) -> None:
        entry = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": name,
            **redact(payload),
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def read_trace(path: Path) -> list[dict[str, Any]]:
    """读取 trace 文件为事件列表（用于测试与回放）。"""
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events
