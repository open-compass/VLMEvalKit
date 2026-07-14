"""Structured matching entrypoints used by datasets and smoke scripts."""

from .match import (
    compute_edit_distance_matrix_new,
    get_gt_pred_lines,
    get_pred_category_type,
    match_gt2pred_no_split,
    match_gt2pred_simple,
    match_gt2pred_timeout_safe,
)
from .match_quick import (
    match_gt2pred_quick,
    normalize_logic_brace_formula,
    sort_by_position_skip_inline,
    split_equation_arrays,
    split_gt_equation_arrays,
)

__all__ = [
    "compute_edit_distance_matrix_new",
    "get_gt_pred_lines",
    "get_pred_category_type",
    "match_gt2pred_no_split",
    "match_gt2pred_quick",
    "match_gt2pred_simple",
    "match_gt2pred_timeout_safe",
    "normalize_logic_brace_formula",
    "sort_by_position_skip_inline",
    "split_equation_arrays",
    "split_gt_equation_arrays",
]
