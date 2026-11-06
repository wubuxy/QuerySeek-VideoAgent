"""字幕检索 baseline：BM25（复用 rank-bm25，Apache-2.0）+ 时间邻接合并。

每个命中的字幕段映射回可播放区间（开发说明书 §7.2），相邻命中段按
merge_gap_sec 合并成窗口；不将未校准的 BM25 分数与其他模态直接相加。
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from ..config import Config
from ..schemas import Evidence, SearchResult, SubtitleSegment, TimeWindow
from ..tools.validate import normalize_windows
from ..tokenize import tokenize_query

__all__ = ["subtitle_search"]


def _segment_tokens(seg: SubtitleSegment) -> list[str]:
    return tokenize_query(seg.text)


def subtitle_search(
    video_id: str,
    query: str,
    segments: list[SubtitleSegment],
    duration_sec: float,
    config: Config,
    top_k: int | None = None,
    method: str = "subtitle",
) -> SearchResult:
    import time

    t0 = time.perf_counter()
    rcfg = config.retrieval
    top_k = top_k or rcfg.top_k_windows
    warnings: list[str] = []
    evidence: list[Evidence] = []
    windows: list[TimeWindow] = []

    if not segments:
        warnings.append("无字幕数据（未提供 SRT 且未启用 ASR），字幕检索不可用")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0, warnings=warnings,
        )

    corpus = [_segment_tokens(s) for s in segments]
    corpus = [toks if toks else ["<empty>"] for toks in corpus]
    bm25 = BM25Okapi(corpus)
    q_tokens = tokenize_query(query)
    if not q_tokens:
        warnings.append("查询分词后为空（全部是停用词）")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0, warnings=warnings,
        )
    scores = bm25.get_scores(q_tokens)

    # 分数 > 0 的命中段（BM25 负分=不相关），按分数排序
    scored = sorted(
        ((s, seg) for s, seg in zip(scores, segments) if s > 0),
        key=lambda x: x[0],
        reverse=True,
    )
    if not scored:
        warnings.append("未找到与查询匹配的字幕（证据不足）")
        return SearchResult(
            video_id=video_id, query=query, method=method,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0, warnings=warnings,
        )

    # 取前 N 个命中段，按时间排序后合并相邻段。
    # 话题连续性启发式：相邻段分数低于组内最高分 * merge_ratio 时不合并，
    # 避免把低分邻段拖进窗口（如 SRT 相邻但话题不同的一句）。
    score_of = {seg.seg_id: sc for sc, seg in scored}
    top_segs = sorted((seg for _, seg in scored[: max(top_k * 3, 10)]), key=lambda s: s.start_sec)
    merged: list[list[tuple[float, SubtitleSegment]]] = []
    for seg in top_segs:
        sc = score_of[seg.seg_id]
        if merged:
            best = max(score_of[g[1].seg_id] for g in merged[-1])
            gap = seg.start_sec - merged[-1][-1][0]
            # 较弱一方分数须 >= 较强一方 * merge_ratio 才合并
            if (
                gap <= rcfg.subtitle.merge_gap_sec
                and min(sc, best) >= rcfg.subtitle.merge_ratio * max(sc, best)
            ):
                merged[-1].append((seg.end_sec, seg))
                continue
        merged.append([(seg.end_sec, seg)])

    raw_windows = []
    for group in merged:
        s = group[0][1].start_sec
        e = group[-1][0]
        score = sum(score_of[g[1].seg_id] for g in group)
        raw_windows.append(
            TimeWindow(start_sec=s, end_sec=e, score=score, evidence_ids=[g[1].seg_id for g in group])
        )
    windows = normalize_windows(
        raw_windows,
        duration_sec=duration_sec,
        min_sec=rcfg.min_window_sec,
        max_sec=rcfg.max_window_sec,
    )
    windows = sorted(windows, key=lambda w: (w.score or 0.0), reverse=True)[:top_k]

    for group in merged:
        for _end, seg in group:
            evidence.append(
                Evidence(
                    evidence_id=f"asr_{seg.seg_id}",
                    type="asr",
                    window=TimeWindow(start_sec=seg.start_sec, end_sec=seg.end_sec),
                    content=seg.text,
                    source_file=seg.source,
                    score=next((sc for sc, sg in scored if sg.seg_id == seg.seg_id), 0.0),
                )
            )

    return SearchResult(
        video_id=video_id,
        query=query,
        windows=windows,
        method=method,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        evidence=evidence,
        warnings=warnings,
    )
