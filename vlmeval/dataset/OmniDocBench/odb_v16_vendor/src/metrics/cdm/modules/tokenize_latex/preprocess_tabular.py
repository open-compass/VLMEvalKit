"""
Pure-Python replacement for the original `preprocess_tabular.js`.

It provides two modes:
  - tokenize: KaTeX global_str-like token stream (used by the old pipeline)
  - normalize: KaTeX parseTree -> normalized LaTeX tokens
"""

from __future__ import annotations

import re
from typing import Literal

from .katex_renderer import KaTeXRenderer
from .options import Options
from .parse_guard import likely_bad_latex, suppress_pylatexenc_warnings
from .pylatexenc_to_katex import parse_latex_to_katex_ast
from .pylatexenc_tokenizer import latex_to_tokens

Mode = Literal["tokenize", "normalize"]


_LABEL_RE = re.compile(r"\\label\{.*?\}")
_GT_RE = re.compile(r"\\>")


def preprocess_tabular_line(line: str) -> str:
    """
    Match the original JS preprocessing behavior for tabular:
    - Strip leading '%' (one char)
    - Do NOT split on '%'
    - Replace '\\~' with a space
    - Remove '\\label{...}'
    - Do NOT remove '$' (tabular uses $...$ a lot)
    - Replace '\\\\' with '\\,' when not in matrix/cases/array/begin contexts
    """
    if not line:
        return line

    if line.startswith("%"):
        line = line[1:]

    line = line.replace(r"\~", " ")
    line = _GT_RE.sub(" ", line)
    line = _LABEL_RE.sub("", line)

    if all(key not in line for key in ("matrix", "cases", "array", "begin")):
        line = line.replace(r"\\", r"\,")

    return line


def normalize_tabular(line: str, mode: Mode = "tokenize") -> str:
    """
    Normalize or tokenize a tabular expression.

    Returns an empty string on parse failure.
    """
    try:
        pre = preprocess_tabular_line(line)
        is_bad, _ = likely_bad_latex(pre, latex_type="tabular")
        if is_bad:
            return ""

        with suppress_pylatexenc_warnings():
            if mode == "tokenize":
                return " ".join(latex_to_tokens(pre)).strip()

            ast = parse_latex_to_katex_ast(pre)
            return KaTeXRenderer(array_mode="tabular").render(ast, Options()).strip()
    except Exception:
        return ""
