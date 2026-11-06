"""检索层。

- subtitle（BM25 基线，P1）
- uniform（E0 视觉基线，P1）
- semantic（E1 组件：CLIP/Mock 图文检索，P2）
- fusion（RRF + IoU-NMS，P2）
- adaptive（E2：Query-Adaptive 粗到细 + 预算守卫，P2）
"""

from .adaptive import adaptive_search, estimate_uncertainty
from .fusion import nms_dedupe, rrf_fuse
from .semantic import semantic_search
from .subtitle import subtitle_search
from .uniform import select_budget_frames, uniform_search

__all__ = [
    "adaptive_search",
    "estimate_uncertainty",
    "nms_dedupe",
    "rrf_fuse",
    "select_budget_frames",
    "semantic_search",
    "subtitle_search",
    "uniform_search",
]
