""".env 加载（CLI/脚本入口用）。

测试通过 QUERYSEEK_LOAD_DOTENV=0 禁用，避免 CI 意外读到真实密钥、消耗真实 API。
"""

from __future__ import annotations

import os


def ensure_dotenv() -> None:
    if os.environ.get("QUERYSEEK_LOAD_DOTENV", "1") == "0":
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # pragma: no cover
        pass
