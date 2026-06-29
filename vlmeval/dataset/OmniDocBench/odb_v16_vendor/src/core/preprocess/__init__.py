"""Structured preprocessing entrypoints used by the public evaluation runtime."""

from .data_preprocess import (
    formula_to_text,
    normalized_formula,
    normalized_html_table,
    normalized_latex_table,
    normalized_table,
    normalized_text,
    remove_markdown_fences,
    replace_repeated_chars,
    strip_formula_delimiters,
    strip_formula_tags,
    textblock2unicode,
    textblock_with_norm_formula,
)
from .extract import md_tex_filter, _sanitize_formula_candidate
from .formula_cdm import build_matrix_cdm_variants, sanitize_formula_for_cdm
from .read_files import read_md_file
from .table_postprocess import table_content_post_process
from .text_postprocess import latex_timeout_context, likely_bad_latex, safe_latex_to_text

__all__ = [
    "build_matrix_cdm_variants",
    "formula_to_text",
    "latex_timeout_context",
    "likely_bad_latex",
    "md_tex_filter",
    "normalized_formula",
    "normalized_html_table",
    "normalized_latex_table",
    "normalized_table",
    "normalized_text",
    "read_md_file",
    "remove_markdown_fences",
    "replace_repeated_chars",
    "sanitize_formula_for_cdm",
    "safe_latex_to_text",
    "strip_formula_delimiters",
    "strip_formula_tags",
    "table_content_post_process",
    "textblock2unicode",
    "textblock_with_norm_formula",
    "_sanitize_formula_candidate",
]
