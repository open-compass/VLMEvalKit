from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .katex_nodes import KaTeXNode
from .options import Options


@dataclass
class KaTeXRenderer:
    """
    Render a minimal KaTeX-like AST into a space-separated LaTeX string.

    This mirrors the original JS renderer in `preprocess_formula.js` /
    `preprocess_tabular.js` (KaTeX parseTree -> normalized LaTeX tokens).
    """

    array_mode: str = "formula"  # "formula" | "tabular"

    def __post_init__(self) -> None:
        if self.array_mode not in {"formula", "tabular"}:
            raise ValueError(f"array_mode must be 'formula' or 'tabular', got: {self.array_mode}")
        self._tokens: list[str] = []

    def render(self, expression: Iterable[KaTeXNode], options: Optional[Options] = None) -> str:
        self._tokens = []
        self._build_expression(list(expression), options or Options())
        return " ".join(self._tokens).strip()

    def _emit(self, *tokens: str) -> None:
        for tok in tokens:
            if tok is None or tok == "":
                continue
            self._tokens.append(tok)

    @staticmethod
    def _fmt_number(val: float) -> str:
        # Match KaTeX/JS printing: avoid trailing ".0" for integers.
        try:
            if float(val).is_integer():
                return str(int(val))
        except Exception:
            pass
        return str(val)

    def _build_expression(self, expression: list[KaTeXNode], options: Options) -> None:
        for group in expression:
            self._build_group(group, options)

    def _build_group(self, group: KaTeXNode, options: Options) -> None:
        handler = getattr(self, f"_group_{group.type}", None)
        if handler is None:
            raise ValueError(f"Got group of unknown type: '{group.type}'")
        handler(group, options)

    def _group_mathord(self, group: KaTeXNode, options: Options) -> None:
        val = str(group.value)
        if options and options.font == "mathrm" and val:
            if val.startswith("\\"):
                self._emit(val)
                return
            for ch in val:
                if ch == " ":
                    self._emit(r"\;")
                else:
                    self._emit(ch)
        else:
            self._emit(val)

    def _group_textord(self, group: KaTeXNode, options: Options) -> None:
        self._emit(str(group.value))

    def _group_bin(self, group: KaTeXNode, options: Options) -> None:
        self._emit(str(group.value))

    def _group_rel(self, group: KaTeXNode, options: Options) -> None:
        self._emit(str(group.value))

    def _group_open(self, group: KaTeXNode, options: Options) -> None:
        self._emit(str(group.value))

    def _group_close(self, group: KaTeXNode, options: Options) -> None:
        self._emit(str(group.value))

    def _group_inner(self, group: KaTeXNode, options: Options) -> None:
        self._emit(str(group.value))

    def _group_punct(self, group: KaTeXNode, options: Options) -> None:
        self._emit(str(group.value))

    def _group_ordgroup(self, group: KaTeXNode, options: Options) -> None:
        self._emit("{")
        self._build_expression(list(group.value or []), options)
        self._emit("}")

    def _group_text(self, group: KaTeXNode, options: Options) -> None:
        self._emit(r"\mathrm", "{")
        self._build_expression(list(group.value.body or []), options)
        self._emit("}")

    def _group_color(self, group: KaTeXNode, options: Options) -> None:
        # The original JS renderer wraps output in a MathML node; for our
        # tokenizer, we only need the inner expression.
        self._build_expression(list(group.value.value or []), options)

    def _group_supsub(self, group: KaTeXNode, options: Options) -> None:
        self._build_group(group.value.base, options)

        if group.value.sub is not None:
            self._emit("_")
            if getattr(group.value.sub, "type", None) != "ordgroup":
                self._emit("{")
                self._build_group(group.value.sub, options)
                self._emit("}")
            else:
                self._build_group(group.value.sub, options)

        if group.value.sup is not None:
            self._emit("^")
            if getattr(group.value.sup, "type", None) != "ordgroup":
                self._emit("{")
                self._build_group(group.value.sup, options)
                self._emit("}")
            else:
                self._build_group(group.value.sup, options)

    def _group_genfrac(self, group: KaTeXNode, options: Options) -> None:
        self._emit(r"\binom" if not group.value.hasBarLine else r"\frac")
        self._build_group(group.value.numer, options)
        self._build_group(group.value.denom, options)

    def _group_array(self, group: KaTeXNode, options: Options) -> None:
        if self.array_mode == "formula":
            self._emit(r"\begin{array}", "{")
            cols = group.value.cols
            if cols is not None:
                for start in cols:
                    if start is not None and getattr(start, "align", None):
                        self._emit(start.align)
            else:
                # Fallback: infer number of columns from the first row.
                if group.value.body and len(group.value.body) > 0:
                    ncols = len(group.value.body[0])
                    style = getattr(group.value, "style", "") or ""
                    if style.startswith("aligned"):
                        for col_idx in range(ncols):
                            self._emit("r" if (col_idx % 2 == 0) else "l")
                    else:
                        for _ in range(ncols):
                            self._emit("l")
            self._emit("}")

            for row in group.value.body or []:
                if not row:
                    continue
                if any(getattr(cell, "value", None) for cell in row):
                    for cell in row:
                        self._build_group(cell, options)
                        # Remove empty "{ }" cells (match JS behavior).
                        if len(self._tokens) >= 2 and self._tokens[-2:] == ["{", "}"]:
                            self._tokens = self._tokens[:-2]
                        self._emit("&")
                    if self._tokens and self._tokens[-1] == "&":
                        self._tokens.pop()
                    self._emit(r"\\")

            self._emit(r"\end{array}")
            return

        # tabular mode (matches preprocess_tabular.js)
        style = group.value.style
        self._emit(r"\begin{" + style + "}")

        if style in {"array", "tabular", "tabularx"}:
            self._emit("{")
            cols = group.value.cols
            if cols is not None:
                for start in cols:
                    if start is None:
                        continue
                    if getattr(start, "type", None) == "align" and getattr(start, "align", None):
                        self._emit(start.align)
                    elif getattr(start, "type", None) == "separator" and getattr(start, "separator", None):
                        self._emit(start.separator)
            else:
                if group.value.body and len(group.value.body) > 0:
                    for _ in group.value.body[0]:
                        self._emit("c")
            self._emit("}")

        for row in group.value.body or []:
            if not row:
                continue
            # Skip empty rows.
            if len(row) == 1 and (not getattr(row[0], "value", None)):
                continue

            # \hline handling: if first cell starts with "\hline", emit it and remove it.
            if (
                row
                and getattr(row[0], "type", None) == "ordgroup"
                and isinstance(getattr(row[0], "value", None), list)
                and row[0].value
                and getattr(row[0].value[0], "value", None) == r"\hline"
            ):
                self._emit(r"\hline")
                row[0].value = row[0].value[1:]

            for cell in row:
                self._build_group(cell, options)
                self._emit("&")

            if self._tokens and self._tokens[-1] == "&":
                self._tokens.pop()
            self._emit(r"\\")

        self._emit(r"\end{" + style + "}")

    def _group_sqrt(self, group: KaTeXNode, options: Options) -> None:
        if group.value.index is not None:
            self._emit(r"\sqrt", "[")
            # index is usually an ordgroup; render its inner value without braces
            self._build_expression(list(group.value.index.value or []), options)
            self._emit("]")
            self._build_group(group.value.body, options)
        else:
            self._emit(r"\sqrt")
            self._build_group(group.value.body, options)

    def _group_leftright(self, group: KaTeXNode, options: Options) -> None:
        self._emit(r"\left" + group.value.left)
        self._build_expression(list(group.value.body or []), options)
        self._emit(r"\right" + group.value.right)

    def _group_accent(self, group: KaTeXNode, options: Options) -> None:
        base = group.value.base
        if getattr(base, "type", None) != "ordgroup":
            self._emit(group.value.accent, "{")
            self._build_group(base, options)
            self._emit("}")
        else:
            self._emit(group.value.accent)
            self._build_group(base, options)

    def _group_spacing(self, group: KaTeXNode, options: Options) -> None:
        self._emit("~" if group.value == " " else str(group.value))

    def _group_op(self, group: KaTeXNode, options: Options) -> None:
        if group.value.symbol:
            self._emit(group.value.body)
            return

        self._emit(r"\operatorname*" if group.value.limits else r"\operatorname", "{")
        body = str(group.value.body or "")
        # JS skips the first character (typically "\" for things like "\sin").
        for ch in body[1:]:
            self._emit(ch)
        self._emit("}")

    def _group_katex(self, group: KaTeXNode, options: Options) -> None:
        # Not needed for tokenization.
        return

    def _group_font(self, group: KaTeXNode, options: Options) -> None:
        font = group.value.font
        if font in {"mbox", "hbox"}:
            font = "mathrm"
        self._emit("\\" + font)
        self._build_group(group.value.body, options.with_font(font))

    def _group_delimsizing(self, group: KaTeXNode, options: Options) -> None:
        self._emit(group.value.funcName, group.value.value)

    def _group_styling(self, group: KaTeXNode, options: Options) -> None:
        self._emit(group.value.original)
        self._build_expression(list(group.value.value or []), options)

    def _group_sizing(self, group: KaTeXNode, options: Options) -> None:
        if group.value.original == r"\rm":
            self._emit(r"\mathrm", "{")
            self._build_expression(list(group.value.value or []), options.with_font("mathrm"))
            self._emit("}")
        else:
            self._emit(group.value.original)
            self._build_expression(list(group.value.value or []), options)

    def _group_overline(self, group: KaTeXNode, options: Options) -> None:
        self._emit(r"\overline", "{")
        self._build_group(group.value.body, options)
        self._emit("}")

    def _group_underline(self, group: KaTeXNode, options: Options) -> None:
        self._emit(r"\underline", "{")
        self._build_group(group.value.body, options)
        self._emit("}")

    def _group_rule(self, group: KaTeXNode, options: Options) -> None:
        self._emit(
            r"\rule",
            "{",
            self._fmt_number(group.value.width.number),
            group.value.width.unit,
            "}",
            "{",
            self._fmt_number(group.value.height.number),
            group.value.height.unit,
            "}",
        )

    def _group_llap(self, group: KaTeXNode, options: Options) -> None:
        self._emit(r"\llap")
        self._build_group(group.value.body, options)

    def _group_rlap(self, group: KaTeXNode, options: Options) -> None:
        self._emit(r"\rlap")
        self._build_group(group.value.body, options)

    def _group_phantom(self, group: KaTeXNode, options: Options) -> None:
        self._emit(r"\phantom", "{")
        self._build_expression(list(group.value.value or []), options)
        self._emit("}")
