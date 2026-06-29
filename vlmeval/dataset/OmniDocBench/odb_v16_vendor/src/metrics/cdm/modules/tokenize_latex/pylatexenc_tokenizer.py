from __future__ import annotations

from typing import Iterable


def latex_to_tokens(latex: str) -> list[str]:
    """
    Convert LaTeX to a KaTeX-global_str-like token stream (space-separated).

    This is used to replace the old Node/KaTeX tokenizer path, especially for
    tabular inputs where the original implementation used KaTeX's `global_str`
    ("tokenize" mode).

    Notes:
    - Whitespace is ignored (matches KaTeX lexer in math mode + downstream `.split()`).
    - `$...$` is preserved by re-inserting `$` around `LatexMathNode` content,
      because pylatexenc parses `$...$` as a math node while KaTeX treats `$` as
      a literal token in this project.
    """
    try:
        from pylatexenc.latexwalker import LatexWalker
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pylatexenc is required for the Python tokenizer. Install with: pip install pylatexenc"
        ) from e

    nodes, _, _ = LatexWalker(latex).get_latex_nodes()
    out: list[str] = []
    _emit_tokens_from_nodes(out, nodes)
    return out


def _emit_tokens_from_nodes(out: list[str], nodes: Iterable[object]) -> None:
    try:
        from pylatexenc.latexwalker import (
            LatexCharsNode,
            LatexCommentNode,
            LatexEnvironmentNode,
            LatexGroupNode,
            LatexMacroNode,
            LatexMathNode,
            LatexSpecialsNode,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError("pylatexenc is required. Install with: pip install pylatexenc") from e

    for node in list(nodes or []):
        if isinstance(node, LatexCommentNode):
            continue

        if isinstance(node, LatexCharsNode):
            for ch in node.chars or "":
                if ch.isspace():
                    continue
                out.append(ch)
            continue

        if isinstance(node, LatexSpecialsNode):
            for ch in getattr(node, "specials_chars", "") or "":
                if ch.isspace():
                    continue
                out.append(ch)
            continue

        if isinstance(node, LatexGroupNode):
            left, right = getattr(node, "delimiters", ("{", "}"))
            out.append(left)
            _emit_tokens_from_nodes(out, getattr(node, "nodelist", None))
            out.append(right)
            continue

        if isinstance(node, LatexMathNode):
            out.append("$")
            _emit_tokens_from_nodes(out, getattr(node, "nodelist", None))
            out.append("$")
            continue

        if isinstance(node, LatexEnvironmentNode):
            env_name = getattr(node, "environmentname", "")
            out.append(r"\begin")
            out.append("{" + env_name + "}")

            # Emit environment arguments (e.g. {cc} in \begin{tabular}{cc}).
            argd = getattr(node, "nodeargd", None)
            for arg in list(getattr(argd, "argnlist", []) or []):
                if arg is None:
                    continue
                _emit_tokens_from_nodes(out, [arg])

            _emit_tokens_from_nodes(out, getattr(node, "nodelist", None))

            out.append(r"\end")
            out.append("{" + env_name + "}")
            continue

        if isinstance(node, LatexMacroNode):
            name = getattr(node, "macroname", "")
            out.append("\\" + name)

            # If pylatexenc attached arguments to this macro, emit them as well.
            argd = getattr(node, "nodeargd", None)
            argnlist = list(getattr(argd, "argnlist", []) or [])
            if argnlist:
                for arg in argnlist:
                    if arg is None:
                        continue
                    _emit_tokens_from_nodes(out, [arg])
            else:
                for arg in list(getattr(node, "nodeargs", []) or []):
                    _emit_tokens_from_nodes(out, [arg])
            continue

        # Unknown node: ignore.
        continue

