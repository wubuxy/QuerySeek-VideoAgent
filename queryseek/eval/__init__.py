"""评测层：QVHighlights 官方协议适配 + 公平对照 runner。"""

from .qvhighlights import (
    compute_boundary_error,
    compute_miou,
    compute_mr_ap,
    compute_mr_r1,
    eval_mr_brief,
    evaluate_submission,
    load_qv_gt,
    load_submission,
    save_submission,
    submission_from_windows,
)
from .runner import (
    EvalSample,
    MethodResult,
    dataset_from_labels,
    load_dataset,
    run_benchmark,
)

__all__ = [
    "EvalSample",
    "MethodResult",
    "compute_boundary_error",
    "compute_miou",
    "compute_mr_ap",
    "compute_mr_r1",
    "dataset_from_labels",
    "eval_mr_brief",
    "evaluate_submission",
    "load_dataset",
    "load_qv_gt",
    "load_submission",
    "run_benchmark",
    "save_submission",
    "submission_from_windows",
]
