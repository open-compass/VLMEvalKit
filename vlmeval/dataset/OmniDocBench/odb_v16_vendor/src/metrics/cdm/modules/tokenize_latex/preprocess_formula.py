"""
Pure-Python replacement for the original `preprocess_formula.js`.

It provides two modes:
  - tokenize: KaTeX global_str-like token stream
  - normalize: KaTeX parseTree -> normalized LaTeX tokens (used by CDM)
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
_CJK_AFTER_MACRO_RE = re.compile(r"(\\[A-Za-z]+)(?=[\u4e00-\u9fff])")


def preprocess_line(line: str, keep_dollar: bool = False) -> str:
    """
    Match the original JS preprocessing behavior:
    - Strip leading '%' (one char) and inline comments after '%'
    - Replace '\\~' with a space
    - Remove '\\label{...}'
    - Replace '$' with a space (formula mode)
    - Replace '\\\\' with '\\,' when not in matrix/cases/array/begin contexts
    """
    if not line:
        return line

    if line.startswith("%"):
        line = line[1:]

    line = line.split("%", 1)[0]
    line = line.replace(r"\~", " ")
    line = _CJK_AFTER_MACRO_RE.sub(r"\1 ", line)

    line = _GT_RE.sub(" ", line)
    if not keep_dollar:
        line = line.replace("$", " ")
    line = _LABEL_RE.sub("", line)

    if all(key not in line for key in ("matrix", "cases", "array", "begin")):
        line = line.replace(r"\\", r"\,")

    return line


def _append_trailing_backslash(out: str, trailing_backslash: bool) -> str:
    if not trailing_backslash or not out:
        return out
    tokens = out.split()
    if tokens and tokens[-1] == "\\":
        return out
    return f"{out} \\"


def normalize_rm(line: str) -> str:
    return (
        line.replace(r"{\rm", r"\mathrm{")
        .replace(r"{ \rm", r"\mathrm{")
        .replace(r"\rm{", r"\mathrm{")
    )


def postprocess_norm_str(norm: str) -> str:
    norm = norm.replace("SSSSSS", "$").replace(" S S S S S S", "$")
    norm = re.sub(r"\\label \{ .*? \}", "", norm)
    return norm


def normalize_formula(line: str, mode: Mode = "normalize") -> str:
    """
    Normalize a LaTeX math formula into a space-separated token string.

    Returns an empty string on parse failure.
    """
    try:
        guard_input = preprocess_line(line, keep_dollar=True)
        is_bad, _ = likely_bad_latex(guard_input, latex_type="formula")
        if is_bad:
            return ""

        pre = guard_input.replace("$", " ")
        trailing_backslash = pre.endswith("\\")

        with suppress_pylatexenc_warnings():
            if mode == "tokenize":
                out = " ".join(latex_to_tokens(pre)).strip()
                return _append_trailing_backslash(out, trailing_backslash)

            pre = normalize_rm(pre)
            ast = parse_latex_to_katex_ast(pre)
            out = KaTeXRenderer(array_mode="formula").render(ast, Options())
        out = postprocess_norm_str(out).strip()
        if not out and pre.strip() == "\\":
            out = "\\"
        return _append_trailing_backslash(out, trailing_backslash)
    except Exception:
        return ""
