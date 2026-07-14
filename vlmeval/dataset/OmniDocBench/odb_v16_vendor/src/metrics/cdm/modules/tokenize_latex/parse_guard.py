from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import re


_ENV_RE = re.compile(r"\\(begin|end)\{([^{}]+)\}")
_LEFT_RE = re.compile(r"\\left\b")
_RIGHT_RE = re.compile(r"\\right\b")
_INLINE_OPEN_RE = re.compile(r"\\\(")
_INLINE_CLOSE_RE = re.compile(r"\\\)")
_DISPLAY_OPEN_RE = re.compile(r"\\\[")
_DISPLAY_CLOSE_RE = re.compile(r"\\\]")
_UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")

_PYLATEXENC_LOGGER_NAMES = [
    "pylatexenc",
    "pylatexenc.latexwalker",
    "pylatexenc.latexwalker._walker",
    "pylatexenc.macrospec",
    "pylatexenc.macrospec._environmentbodyparser",
]


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def braces_balanced(line: str) -> bool:
    depth = 0
    idx = 0
    while idx < len(line):
        ch = line[idx]
        if ch == "\\" and idx + 1 < len(line) and line[idx + 1] in "{}[]()":
            idx += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        idx += 1
    return depth == 0


def env_balanced(line: str) -> bool:
    stack: list[str] = []
    for match in _ENV_RE.finditer(line):
        kind, name = match.group(1), match.group(2)
        if kind == "begin":
            stack.append(name)
            continue
        if not stack or stack[-1] != name:
            return False
        stack.pop()
    return not stack


def _math_delims_balanced(line: str) -> bool:
    if len(_INLINE_OPEN_RE.findall(line)) != len(_INLINE_CLOSE_RE.findall(line)):
        return False
    if len(_DISPLAY_OPEN_RE.findall(line)) != len(_DISPLAY_CLOSE_RE.findall(line)):
        return False
    dollar_count = len(_UNESCAPED_DOLLAR_RE.findall(line))
    if dollar_count % 2 != 0:
        return False
    return True


def _left_right_balanced(line: str) -> bool:
    return len(_LEFT_RE.findall(line)) == len(_RIGHT_RE.findall(line))


def likely_bad_latex(line: str, latex_type: str = "formula") -> tuple[bool, str]:
    stripped = (line or "").strip()
    if not stripped:
        return False, ""

    max_len = _read_int_env("CDM_MAX_LATEX_INPUT_LEN", 20000)
    if max_len > 0 and len(stripped) > max_len:
        return True, f"too_long:{len(stripped)}"

    if not braces_balanced(stripped):
        return True, "unbalanced_braces"

    if not env_balanced(stripped):
        return True, "unbalanced_env"

    if latex_type == "formula":
        if not _left_right_balanced(stripped):
            return True, "unbalanced_left_right"
        if not _math_delims_balanced(stripped):
            return True, "unbalanced_math_delims"

    return False, ""


@contextmanager
def suppress_pylatexenc_warnings():
    logger_states = []
    for name in _PYLATEXENC_LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger_states.append((logger, logger.level))
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in logger_states:
            logger.setLevel(level)
