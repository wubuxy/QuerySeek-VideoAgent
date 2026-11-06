"""查询分词：中英混合的轻量 tokenizer。

中文按"整词 + 字符二元组"切分（BM25 默认空白分词对中文无效，二元组是
不引入分词器依赖的最小可用方案）；生成二元组前先剔除停用词（先多字词
再单字），避免产生"构的/的片"这类跨越停用词的噪声 token；英文按词切分。
"""

from __future__ import annotations

import re

_CJK_RUN = re.compile(r"[一-鿿]+")
_WORD_RUN = re.compile(r"[a-z0-9]+")

# 查询指令词 + 常见停用词（英文小写）
_STOPWORDS = {
    # 英文
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "in", "on", "at",
    "to", "for", "and", "or", "with", "this", "that", "it", "its", "where",
    "what", "which", "how", "when", "why", "who", "show", "me", "find",
    "search", "locate", "clip", "segment", "moment", "video", "please", "about",
    # 中文指令词/停用词
    "的", "了", "是", "在", "有", "和", "与", "中", "里", "个", "一", "就", "都",
    "片段", "视频", "寻找", "查找", "找", "什么", "哪里", "哪个", "如何", "怎么",
    "请", "帮我", "部分", "内容", "相关",
}

# 多字停用词（先从 CJK run 中整段剔除，再做单字剔除）
_MULTI_STOPWORDS = sorted(
    (w for w in _STOPWORDS if len(w) > 1), key=len, reverse=True
)
_SINGLE_STOPWORDS = {w for w in _STOPWORDS if len(w) == 1}


def _clean_cjk_run(run: str) -> str:
    for w in _MULTI_STOPWORDS:
        run = run.replace(w, "")
    return "".join(ch for ch in run if ch not in _SINGLE_STOPWORDS)


def tokenize_query(text: str) -> list[str]:
    """返回去停用词、去重（保持顺序）的 token 列表。"""
    text = text.lower()
    tokens: list[str] = []
    pos = 0
    for m in _CJK_RUN.finditer(text):
        for w in _WORD_RUN.findall(text[pos : m.start()]):
            tokens.append(w)
        run = _clean_cjk_run(m.group(0))
        if not run:
            pos = m.end()
            continue
        tokens.append(run)  # 整词
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # 字符二元组
        pos = m.end()
    for w in _WORD_RUN.findall(text[pos:]):
        tokens.append(w)
    out: list[str] = []
    for t in tokens:
        if t not in _STOPWORDS and t not in out:
            out.append(t)
    return out
