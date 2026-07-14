from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional, Union

from .katex_nodes import (
    AccentValue,
    ArrayValue,
    ColAlign,
    ColSeparator,
    FontValue,
    GenFracValue,
    KaTeXNode,
    LeftRightValue,
    LapValue,
    OpValue,
    OverUnderValue,
    PhantomValue,
    RuleDim,
    RuleValue,
    SqrtValue,
    SupSubValue,
    TextValue,
)


_SIZE_RE = re.compile(r"^\s*(-?)\s*(\d+(?:\.\d*)?|\.\d+)\s*([a-z]{2})\s*$")

_OP_WITH_LIMITS = {
    "det",
    "gcd",
    "inf",
    "lim",
    "liminf",
    "limsup",
    "max",
    "min",
    "Pr",
    "sup",
}
_OP_WITHOUT_LIMITS = {
    "arccos",
    "tanh",
}

_FONT_MACROS = {
    "mathrm",
    "mathbf",
    "mathit",
    "mathsf",
    "mathtt",
    "mathcal",
    "mathbb",
    "mathfrak",
    "mathscr",
    "mbox",
    "hbox",
}

_ACCENT_MACROS = {
    "hat",
    "bar",
    "vec",
    "dot",
    "ddot",
    "tilde",
    "breve",
    "check",
    "acute",
    "grave",
}

_LITERAL_MACROS_WITH_ARG = {
    "widetilde",
    "widehat",
}

_OPERATORNAME_REWRITE = {
    "arccos",
    "arcsin",
    "arctan",
    "arg",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "deg",
    "det",
    "dim",
    "exp",
    "gcd",
    "hom",
    "inf",
    "injlim",
    "ker",
    "lg",
    "lim",
    "liminf",
    "limsup",
    "ln",
    "log",
    "max",
    "min",
    "Pr",
    "projlim",
    "sec",
    "sin",
    "sinh",
    "sup",
    "tan",
    "tanh",
}

_MATRIX_DELIMS = {
    "pmatrix": ("(", ")"),
    "bmatrix": ("[", "]"),
    "Bmatrix": (r"\{", r"\}"),
    "vmatrix": ("|", "|"),
    "Vmatrix": (r"\Vert", r"\Vert"),
}

_CASES_DELIMS = {
    "cases": (r"\{", "."),
    "dcases": (r"\{", "."),
    "rcases": (".", r"\}"),
}



@dataclass(frozen=True)
class _ScriptMarker:
    kind: str  # "^" or "_"


_NodeOrMarker = Union[KaTeXNode, _ScriptMarker]


def parse_latex_to_katex_ast(latex: str) -> list[KaTeXNode]:
    """
    Parse LaTeX using pylatexenc and convert it into a minimal KaTeX-like AST.

    The result is designed to be rendered by `KaTeXRenderer` (see `katex_renderer.py`).
    """
    try:
        from pylatexenc.latexwalker import LatexWalker
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pylatexenc is required for the Python tokenizer. "
            "Install with: pip install pylatexenc"
        ) from e

    walker = LatexWalker(latex)
    nodes, _, _ = walker.get_latex_nodes()
    return _convert_nodes_to_katex(nodes, in_text_mode=False)


def _convert_nodes_to_katex(
    nodes: Iterable[object],
    *,
    in_text_mode: bool,
    allow_prime_shorthand: bool = True,
) -> list[KaTeXNode]:
    """
    Convert a pylatexenc node list to KaTeX-like nodes (with script attachment).
    """
    items: list[_NodeOrMarker] = []

    node_list = list(nodes or [])
    i = 0
    while i < len(node_list):
        node = node_list[i]

        def skip_ws(j: int) -> int:
            # pylatexenc may insert whitespace-only LatexCharsNode between macro and its brace groups.
            while j < len(node_list):
                chars = getattr(node_list[j], "chars", None)
                specials = getattr(node_list[j], "specials_chars", None)
                if isinstance(chars, str) and chars.strip() == "":
                    j += 1
                    continue
                if isinstance(specials, str) and specials.strip() == "":
                    j += 1
                    continue
                break
            return j

        # Some macros aren't defined with argument specs in pylatexenc defaults
        # (e.g. \operatorname, \phantom, \llap/\rlap, \rule). We handle them with
        # lookahead to keep parity with the original KaTeX-based implementation.
        macro = getattr(node, "macroname", None)
        if macro in _ACCENT_MACROS and i + 1 < len(node_list):
            args = _macro_args(node)
            if len(args) == 1:
                arg0 = args[0]
                inner_macro = getattr(arg0, "macroname", None)
                if inner_macro in _FONT_MACROS and not _macro_args(arg0):
                    j1 = skip_ws(i + 1)
                    if j1 < len(node_list):
                        nxt = node_list[j1]
                    else:
                        nxt = None
                    if _is_group_node(nxt) and getattr(nxt, "delimiters", None) == ("{", "}"):
                        body = _convert_group_as_ordgroup(nxt, in_text_mode=in_text_mode)
                        font_node = KaTeXNode("font", FontValue(font=inner_macro, body=body))
                        items.append(KaTeXNode("accent", AccentValue(accent="\\" + macro, base=font_node)))
                        i = j1 + 1
                        continue
        if macro in _LITERAL_MACROS_WITH_ARG and i + 1 < len(node_list):
            args = _macro_args(node)
            if len(args) == 1:
                arg0 = args[0]
                inner_macro = getattr(arg0, "macroname", None)
                if inner_macro in _FONT_MACROS and not _macro_args(arg0):
                    j1 = skip_ws(i + 1)
                    if j1 < len(node_list):
                        nxt = node_list[j1]
                    else:
                        nxt = None
                    if _is_group_node(nxt) and getattr(nxt, "delimiters", None) == ("{", "}"):
                        body = _convert_group_as_ordgroup(nxt, in_text_mode=in_text_mode)
                        items.append(KaTeXNode("textord", "\\" + macro))
                        items.append(KaTeXNode("font", FontValue(font=inner_macro, body=body)))
                        i = j1 + 1
                        continue
        if macro == "operatorname" and i + 1 < len(node_list):
            j1 = skip_ws(i + 1)
            if j1 >= len(node_list):
                nxt = None
            else:
                nxt = node_list[j1]
            if _is_group_node(nxt):
                raw_text = _group_text(nxt)
                if raw_text is not None and raw_text == raw_text.strip():
                    compact = re.sub(r"\s+", "", raw_text)
                    if compact in _OPERATORNAME_REWRITE:
                        items.append(KaTeXNode("textord", "\\" + compact))
                        i = j1 + 1
                        continue
                # KaTeX parses \operatorname{foo} as `\operatorname` + ordgroup(foo)
                items.append(KaTeXNode("mathord", r"\operatorname"))
                items.append(_convert_group_as_ordgroup(nxt, in_text_mode=False))
                i = j1 + 1
                continue

        if macro in _FONT_MACROS and i + 1 < len(node_list):
            if not _macro_args(node):
                j1 = skip_ws(i + 1)
                if j1 < len(node_list):
                    nxt = node_list[j1]
                else:
                    nxt = None
                if nxt is not None:
                    if _is_group_node(nxt) and getattr(nxt, "delimiters", None) == ("{", "}"):
                        body = _convert_group_as_ordgroup(nxt, in_text_mode=in_text_mode)
                    else:
                        body_nodes = _convert_nodes_to_katex(_node_to_nodelist(nxt), in_text_mode=in_text_mode)
                        body = body_nodes[0] if len(body_nodes) == 1 else KaTeXNode("ordgroup", body_nodes)
                    items.append(KaTeXNode("font", FontValue(font=macro, body=body)))
                    i = j1 + 1
                    continue

        if macro in _OP_WITH_LIMITS or macro in _OP_WITHOUT_LIMITS:
            limits = macro in _OP_WITH_LIMITS
            items.append(KaTeXNode("op", OpValue(symbol=False, limits=limits, body="\\" + macro)))
            i += 1
            continue

        if macro in {"binom", "frac", "tfrac", "dfrac"} and i + 1 < len(node_list):
            # pylatexenc does not always attach args for some macros (notably \binom).
            # Keep parity with the original KaTeX-based tokenizer by binding the next
            # two brace-groups as arguments when needed.
            if not _macro_args(node):
                j1 = skip_ws(i + 1)
                j2 = skip_ws(j1 + 1)
                g1 = node_list[j1] if j1 < len(node_list) else None
                g2 = node_list[j2] if j2 < len(node_list) else None

                if macro == "binom" and g1 is not None:
                    chars = getattr(g1, "chars", None)
                    if isinstance(chars, str):
                        match = re.match(r"\s*([^\s]+)\s+([^\s]+)(.*)", chars, flags=re.S)
                        if match:
                            token1 = match.group(1)
                            token2 = match.group(2)
                            rest = match.group(3)
                            numer_nodes = _attach_scripts(_chars_to_nodes(token1, in_text_mode=in_text_mode))
                            denom_nodes = _attach_scripts(_chars_to_nodes(token2, in_text_mode=in_text_mode))
                            if numer_nodes and denom_nodes:
                                numer = (
                                    numer_nodes[0]
                                    if len(numer_nodes) == 1
                                    else KaTeXNode("ordgroup", numer_nodes)
                                )
                                denom = (
                                    denom_nodes[0]
                                    if len(denom_nodes) == 1
                                    else KaTeXNode("ordgroup", denom_nodes)
                                )
                                items.append(
                                    KaTeXNode(
                                        "genfrac",
                                        GenFracValue(hasBarLine=False, numer=numer, denom=denom),
                                    )
                                )
                                g1.chars = rest
                                i = j1
                                continue

                if _is_group_node(g1) and _is_group_node(g2):
                    numer_nodes = _convert_nodes_to_katex(getattr(g1, "nodelist", None), in_text_mode=in_text_mode)
                    denom_nodes = _convert_nodes_to_katex(getattr(g2, "nodelist", None), in_text_mode=in_text_mode)
                    numer = KaTeXNode("ordgroup", numer_nodes)
                    denom = KaTeXNode("ordgroup", denom_nodes)
                    items.append(
                        KaTeXNode(
                            "genfrac",
                            GenFracValue(hasBarLine=(macro != "binom"), numer=numer, denom=denom),
                        )
                    )
                    i = j2 + 1
                    continue
                if macro == "binom" and g1 is not None and g2 is not None:
                    special1 = getattr(g1, "specials_chars", None)
                    special2 = getattr(g2, "specials_chars", None)
                    macro2 = getattr(g2, "macroname", None)
                    if special1 not in {"&"} and special2 not in {"&"} and macro2 != "\\":
                        chars = getattr(g2, "chars", None)
                        if isinstance(chars, str):
                            match = re.match(r"\s*([^\s]+)(.*)", chars, flags=re.S)
                            if match:
                                token = match.group(1)
                                rest = match.group(2)
                                if rest.strip():
                                    numer_nodes = _convert_nodes_to_katex(_node_to_nodelist(g1), in_text_mode=in_text_mode)
                                    denom_nodes = _attach_scripts(_chars_to_nodes(token, in_text_mode=in_text_mode))
                                    if numer_nodes and denom_nodes:
                                        numer = numer_nodes[0] if len(numer_nodes) == 1 else KaTeXNode("ordgroup", numer_nodes)
                                        denom = denom_nodes[0] if len(denom_nodes) == 1 else KaTeXNode("ordgroup", denom_nodes)
                                        items.append(
                                            KaTeXNode(
                                                "genfrac",
                                                GenFracValue(hasBarLine=False, numer=numer, denom=denom),
                                            )
                                        )
                                        g2.chars = rest
                                        i = j2
                                        continue
                        numer_nodes = _convert_nodes_to_katex(_node_to_nodelist(g1), in_text_mode=in_text_mode)
                        denom_nodes = _convert_nodes_to_katex(_node_to_nodelist(g2), in_text_mode=in_text_mode)
                        if numer_nodes and denom_nodes:
                            numer = numer_nodes[0] if len(numer_nodes) == 1 else KaTeXNode("ordgroup", numer_nodes)
                            denom = denom_nodes[0] if len(denom_nodes) == 1 else KaTeXNode("ordgroup", denom_nodes)
                            items.append(
                                KaTeXNode(
                                    "genfrac",
                                    GenFracValue(hasBarLine=False, numer=numer, denom=denom),
                                )
                            )
                            i = j2 + 1
                            continue

        if macro in {"phantom", "llap", "rlap"} and i + 1 < len(node_list):
            j1 = skip_ws(i + 1)
            if j1 >= len(node_list):
                nxt = None
            else:
                nxt = node_list[j1]
            if _is_group_node(nxt):
                body_nodes = _convert_nodes_to_katex(getattr(nxt, "nodelist", None), in_text_mode=in_text_mode)
                body_group = KaTeXNode("ordgroup", body_nodes)
                if macro == "phantom":
                    items.append(KaTeXNode("phantom", PhantomValue(value=body_nodes)))
                elif macro == "llap":
                    items.append(KaTeXNode("llap", LapValue(body=body_group)))
                else:
                    items.append(KaTeXNode("rlap", LapValue(body=body_group)))
                i = j1 + 1
                continue

        if macro == "rule" and i + 1 < len(node_list):
            j1 = skip_ws(i + 1)
            j2 = skip_ws(j1 + 1)
            if j2 < len(node_list):
                g1 = node_list[j1]
                g2 = node_list[j2]
            else:
                g1 = None
                g2 = None
            if _is_group_node(g1) and _is_group_node(g2):
                w = _parse_rule_dim(_group_text(g1))
                h = _parse_rule_dim(_group_text(g2))
                if w and h:
                    items.append(KaTeXNode("rule", RuleValue(width=w, height=h)))
                    i = j2 + 1
                    continue

        node_allow_prime = allow_prime_shorthand
        if items and isinstance(items[-1], _ScriptMarker):
            node_allow_prime = False
        items.extend(_convert_single_node(node, in_text_mode=in_text_mode, allow_prime_shorthand=node_allow_prime))
        i += 1

    return _attach_scripts(items)


def _attach_scripts(items: list[_NodeOrMarker]) -> list[KaTeXNode]:
    out: list[KaTeXNode] = []
    i = 0
    while i < len(items):
        item = items[i]

        if isinstance(item, _ScriptMarker):
            # Stray script marker; keep as a literal token for robustness.
            out.append(KaTeXNode("textord", item.kind))
            i += 1
            continue

        base = item
        sup: Optional[KaTeXNode] = None
        sub: Optional[KaTeXNode] = None

        def prime_nodes(node: Optional[KaTeXNode]) -> Optional[list[KaTeXNode]]:
            if node is None:
                return None
            if getattr(node, "type", None) == "mathord" and getattr(node, "value", None) in {"'", r"\prime"}:
                return [node]
            if getattr(node, "type", None) == "ordgroup":
                value = list(getattr(node, "value", None) or [])
                if value and all(
                    getattr(child, "type", None) == "mathord"
                    and getattr(child, "value", None) in {"'", r"\prime"}
                    for child in value
                ):
                    return value
            return None

        def prime_group(primes: list[KaTeXNode]) -> KaTeXNode:
            if len(primes) == 1:
                return primes[0]
            return KaTeXNode("ordgroup", primes)

        def normalize_prime_script(script: Optional[KaTeXNode]) -> Optional[KaTeXNode]:
            if script is None:
                return None
            if getattr(script, "type", None) != "ordgroup":
                return script
            value = list(getattr(script, "value", None) or [])
            if not value:
                return script
            head = value[0]
            if getattr(head, "type", None) != "textord" or getattr(head, "value", None) != "^":
                return script
            primes: list[KaTeXNode] = []
            for node in value[1:]:
                nodes = prime_nodes(node)
                if nodes is None:
                    return script
                primes.extend(nodes)
            if not primes:
                return script
            return prime_group(primes)

        j = i + 1
        while j < len(items) and isinstance(items[j], _ScriptMarker):
            marker = items[j].kind
            script = None
            script_index = j + 1
            k = script_index
            while k < len(items):
                if isinstance(items[k], _ScriptMarker):
                    break
                if (
                    isinstance(items[k], KaTeXNode)
                    and getattr(items[k], "type", None) == "spacing"
                    and getattr(items[k], "value", None) == " "
                ):
                    k += 1
                    continue
                script = items[k]
                script_index = k + 1
                break
            if marker == "^":
                script = normalize_prime_script(script)
                if sup is None:
                    sup = script
                else:
                    existing = prime_nodes(sup)
                    added = prime_nodes(script)
                    if existing is not None and added is not None:
                        sup = prime_group([*existing, *added])
                    else:
                        sup = script
            else:
                sub = script
            j = script_index if script is not None else j + 1

        if sup is not None or sub is not None:
            out.append(KaTeXNode("supsub", SupSubValue(base=base, sup=sup, sub=sub)))
            i = j
        else:
            out.append(base)
            i += 1

    return out


def _convert_single_node(
    node: object,
    *,
    in_text_mode: bool,
    allow_prime_shorthand: bool = True,
) -> list[_NodeOrMarker]:
    # Import types lazily so importing this module does not fail in environments
    # where pylatexenc is not installed.
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

    if isinstance(node, LatexCommentNode):
        return []

    if isinstance(node, LatexCharsNode):
        return _chars_to_nodes(
            node.chars or "",
            in_text_mode=in_text_mode,
            allow_prime_shorthand=allow_prime_shorthand,
        )

    if isinstance(node, LatexSpecialsNode):
        return _chars_to_nodes(
            getattr(node, "specials_chars", "") or "",
            in_text_mode=in_text_mode,
            allow_prime_shorthand=allow_prime_shorthand,
        )

    if isinstance(node, LatexGroupNode):
        inner = _convert_nodes_to_katex(
            getattr(node, "nodelist", None),
            in_text_mode=in_text_mode,
            allow_prime_shorthand=allow_prime_shorthand,
        )
        return [KaTeXNode("ordgroup", inner)]

    if isinstance(node, LatexMathNode):
        # In the original KaTeX-based tokenizer, "$" is just another token in math
        # mode. pylatexenc parses it as a math node, so we re-add delimiters.
        inner = _convert_nodes_to_katex(
            getattr(node, "nodelist", None),
            in_text_mode=False,
            allow_prime_shorthand=allow_prime_shorthand,
        )
        return [KaTeXNode("mathord", "$"), *inner, KaTeXNode("mathord", "$")]

    if isinstance(node, LatexEnvironmentNode):
        return [_convert_environment(node)]

    if isinstance(node, LatexMacroNode):
        return _convert_macro(node, in_text_mode=in_text_mode)

    # Unknown node: drop it rather than failing hard.
    return []


def _convert_macro(node: object, *, in_text_mode: bool) -> list[_NodeOrMarker]:
    name = getattr(node, "macroname", "")
    if name in {"limits", "nolimits"}:
        return []

    def _add_post_space(nodes: list[_NodeOrMarker]) -> list[_NodeOrMarker]:
        if in_text_mode and getattr(node, "macro_post_space", ""):
            if not nodes or not (
                isinstance(nodes[-1], KaTeXNode)
                and getattr(nodes[-1], "type", None) == "spacing"
                and getattr(nodes[-1], "value", None) == " "
            ):
                nodes = list(nodes)
                nodes.append(KaTeXNode("spacing", " "))
        return nodes

    def arg_as_group_or_node(arg: Optional[object]) -> KaTeXNode:
        """
        Convert a macro argument into a single KaTeX node.

        - If the argument is a `{...}` group, return an `ordgroup` (so braces remain).
        - Otherwise, return a single node when possible (to match KaTeX behavior like
          `\\frac 1 { x }`, `\\mathcal Z`, etc).
        """
        if arg is None:
            return KaTeXNode("ordgroup", [])

        if _is_group_node(arg) and getattr(arg, "delimiters", None) == ("{", "}"):
            inner = _convert_nodes_to_katex(getattr(arg, "nodelist", None), in_text_mode=in_text_mode)
            return KaTeXNode("ordgroup", inner)

        nodes = _convert_nodes_to_katex(_node_to_nodelist(arg), in_text_mode=in_text_mode)
        if len(nodes) == 1:
            return nodes[0]
        return KaTeXNode("ordgroup", nodes)

    def arg_as_inline_nodes(arg: Optional[object]) -> list[KaTeXNode]:
        """
        Convert a macro argument into an inline node sequence.

        This is mainly used by the generic fallback to avoid dropping arguments.
        It also preserves `[...]` groups as `[` ... `]` tokens (KaTeX outputs them
        as open/close nodes for many control sequences like `\\xrightarrow`).
        """
        if arg is None:
            return []

        if _is_group_node(arg):
            delims = getattr(arg, "delimiters", None)
            inner = _convert_nodes_to_katex(getattr(arg, "nodelist", None), in_text_mode=in_text_mode)
            if delims == ("{", "}"):
                return [KaTeXNode("ordgroup", inner)]
            if delims == ("[", "]"):
                return [KaTeXNode("open", "["), *inner, KaTeXNode("close", "]")]
            return [KaTeXNode("ordgroup", inner)]

        return _convert_nodes_to_katex(_node_to_nodelist(arg), in_text_mode=in_text_mode)

    # Common structured macros (match KaTeX parseTree types used by the renderer)
    if name in {"frac", "tfrac", "dfrac", "binom"}:
        args = _macro_args(node)
        if len(args) >= 2:
            numer = arg_as_group_or_node(args[0])
            denom = arg_as_group_or_node(args[1])
        else:
            numer = KaTeXNode("ordgroup", [])
            denom = KaTeXNode("ordgroup", [])
        return _add_post_space(
            [KaTeXNode("genfrac", GenFracValue(hasBarLine=(name != "binom"), numer=numer, denom=denom))]
        )

    if name == "sqrt":
        # pylatexenc stores optional/mandatory args in nodeargd.argnlist
        argd = getattr(node, "nodeargd", None)
        argnlist = list(getattr(argd, "argnlist", []) or [])
        index_node = argnlist[0] if len(argnlist) >= 2 else None
        body_node = argnlist[1] if len(argnlist) >= 2 else (argnlist[0] if len(argnlist) == 1 else None)

        index = (
            KaTeXNode("ordgroup", _convert_nodes_to_katex(_node_to_nodelist(index_node), in_text_mode=in_text_mode))
            if index_node is not None
            else None
        )
        body = arg_as_group_or_node(body_node)
        return _add_post_space([KaTeXNode("sqrt", SqrtValue(body=body, index=index))])

    if name in _ACCENT_MACROS:
        base = _macro_args(node)[0] if _macro_args(node) else None
        base_group = arg_as_group_or_node(base)
        return _add_post_space([KaTeXNode("accent", AccentValue(accent="\\" + name, base=base_group))])

    if name in {"mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathbb", "mathfrak", "mathscr", "mbox", "hbox"}:
        body = _macro_args(node)[0] if _macro_args(node) else None
        body_group = arg_as_group_or_node(body)
        return _add_post_space([KaTeXNode("font", FontValue(font=name, body=body_group))])

    if name == "text":
        body = _macro_args(node)[0] if _macro_args(node) else None
        body_nodes = _convert_nodes_to_katex(_node_to_nodelist(body), in_text_mode=True)
        return _add_post_space([KaTeXNode("text", TextValue(body=body_nodes))])

    if name in {"overline", "underline"}:
        body = _macro_args(node)[0] if _macro_args(node) else None
        body_group = arg_as_group_or_node(body)
        return _add_post_space([KaTeXNode(name, OverUnderValue(body=body_group))])

    # Default: keep as a literal control sequence token, but DO NOT drop arguments.
    out: list[_NodeOrMarker] = [KaTeXNode("textord", "\\" + name)]
    for a in _macro_args(node):
        out.extend(arg_as_inline_nodes(a))
    return _add_post_space(out)


def _convert_environment(env_node: object) -> KaTeXNode:
    env_name = getattr(env_node, "environmentname", "")

    cols = _parse_env_cols(env_node, env_name)
    body_rows = _parse_env_body(env_node, env_name)
    if cols is None and env_name.startswith("aligned"):
        max_cols = max((len(row) for row in body_rows), default=0)
        if max_cols > 0:
            cols = [ColAlign(align="r" if (i % 2 == 0) else "l") for i in range(max_cols)]
    array_node = KaTeXNode("array", ArrayValue(style=env_name if env_name else "array", cols=cols, body=body_rows))
    base_env = env_name.rstrip("*")
    if base_env in _CASES_DELIMS:
        left, right = _CASES_DELIMS[base_env]
        return KaTeXNode("leftright", LeftRightValue(left=left, right=right, body=[array_node]))
    if base_env in _MATRIX_DELIMS:
        left, right = _MATRIX_DELIMS[base_env]
        return KaTeXNode("leftright", LeftRightValue(left=left, right=right, body=[array_node]))
    return array_node


def _parse_env_cols(env_node: object, env_name: str) -> Optional[list[ColAlign | ColSeparator]]:
    argd = getattr(env_node, "nodeargd", None)
    argnlist = list(getattr(argd, "argnlist", []) or [])
    base_env = env_name.rstrip("*")

    if base_env in _CASES_DELIMS:
        # KaTeX defines cases/dcases/rcases with two left-aligned columns.
        return [ColAlign(align="l"), ColAlign(align="l")]

    # array: argspec='[{' => [optional, colspec]
    # tabular/tabularx: argspec='{' => [colspec]
    colspec_node = None
    if env_name == "array" and len(argnlist) >= 2:
        colspec_node = argnlist[1]
    elif env_name in {"tabular", "tabularx"} and len(argnlist) >= 1:
        colspec_node = argnlist[0]
    elif len(argnlist) >= 1:
        # best-effort for other envs
        colspec_node = argnlist[-1]

    if colspec_node is None:
        return None

    spec = _group_text(colspec_node)
    if spec is None:
        return None
    if spec == "":
        return []

    cols: list[ColAlign | ColSeparator] = []
    for ch in spec:
        if ch in {"l", "c", "r"}:
            cols.append(ColAlign(align=ch))
        elif ch in {"|"}:
            cols.append(ColSeparator(separator=ch))
        else:
            # Keep unknown spec chars as separators so the renderer can output them.
            if not ch.isspace():
                cols.append(ColSeparator(separator=ch))
    return cols or None


def _parse_env_body(env_node: object, env_name: str) -> list[list[KaTeXNode]]:
    try:
        from pylatexenc.latexwalker import LatexCharsNode, LatexMacroNode, LatexSpecialsNode
    except ImportError as e:  # pragma: no cover
        raise ImportError("pylatexenc is required. Install with: pip install pylatexenc") from e

    rows: list[list[KaTeXNode]] = []
    current_row: list[KaTeXNode] = []
    current_cell: list[object] = []
    col_idx = 0

    is_aligned_like = env_name.startswith("aligned")

    def flush_cell() -> None:
        nonlocal current_cell, current_row, col_idx
        cell_nodes = _convert_nodes_to_katex(current_cell, in_text_mode=False)
        # Match KaTeX behavior for aligned-family envs: prefix odd columns with an
        # empty group (e.g., "{ { } = ... }", "{ { } \\log ... }").
        if is_aligned_like and col_idx % 2 == 1:
            cell_nodes = [KaTeXNode("ordgroup", []), *cell_nodes]
        current_row.append(KaTeXNode("ordgroup", cell_nodes))
        current_cell = []
        col_idx += 1

    def flush_row() -> None:
        nonlocal current_row, col_idx
        if current_row:
            rows.append(current_row)
        current_row = []
        col_idx = 0

    for n in getattr(env_node, "nodelist", []) or []:
        if isinstance(n, LatexSpecialsNode) and getattr(n, "specials_chars", None) == "&":
            flush_cell()
            continue

        # Row separator inside environments is the `\\` macro.
        # pylatexenc represents it as a LatexMacroNode with `macroname == '\\'` (len==1).
        if isinstance(n, LatexMacroNode) and getattr(n, "macroname", None) == "\\":
            star_node = None
            argd = getattr(n, "nodeargd", None)
            argnlist = list(getattr(argd, "argnlist", []) or [])
            if argnlist:
                first = argnlist[0]
                if isinstance(first, LatexCharsNode) and getattr(first, "chars", None) == "*":
                    star_node = first
            flush_cell()
            flush_row()
            if star_node is not None:
                current_cell.append(star_node)
            continue

        current_cell.append(n)

    flush_cell()
    flush_row()
    return rows


def _chars_to_nodes(
    chars: str,
    *,
    in_text_mode: bool,
    allow_prime_shorthand: bool = True,
) -> list[_NodeOrMarker]:
    out: list[_NodeOrMarker] = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isspace():
            if in_text_mode:
                # Collapse consecutive whitespace in text-mode to a single spacing token.
                if not (
                    out
                    and isinstance(out[-1], KaTeXNode)
                    and getattr(out[-1], "type", None) == "spacing"
                    and getattr(out[-1], "value", None) == " "
                ):
                    out.append(KaTeXNode("spacing", " "))
            i += 1
            continue

        if not in_text_mode and ch == "-" and i + 1 < len(chars) and chars[i + 1] == "-":
            out.append(KaTeXNode("mathord", "--"))
            i += 2
            continue

        if ch == "'" and not in_text_mode and allow_prime_shorthand:
            count = 1
            while i + count < len(chars) and chars[i + count] == "'":
                count += 1
            primes = [KaTeXNode("mathord", r"\prime") for _ in range(count)]
            out.append(_ScriptMarker(kind="^"))
            out.append(primes[0] if count == 1 else KaTeXNode("ordgroup", primes))
            i += count
            continue

        if ch == "^" or ch == "_":
            out.append(_ScriptMarker(kind=ch))
            i += 1
            continue

        out.append(KaTeXNode("mathord", ch))
        i += 1
    return out


def _macro_args(macro_node: object) -> list[object]:
    """
    Best-effort argument extraction for pylatexenc macro nodes.

    Prefer `nodeargd.argnlist` when present, fall back to `nodeargs`.
    """
    argd = getattr(macro_node, "nodeargd", None)
    argnlist = list(getattr(argd, "argnlist", []) or [])
    if argnlist:
        return [a for a in argnlist if a is not None]
    return list(getattr(macro_node, "nodeargs", []) or [])


def _node_to_nodelist(node: Optional[object]) -> list[object]:
    if node is None:
        return []
    nodelist = getattr(node, "nodelist", None)
    if nodelist is None:
        return [node]
    return list(nodelist or [])


def _is_group_node(node: object) -> bool:
    return hasattr(node, "nodelist") and hasattr(node, "delimiters")


def _convert_group_as_ordgroup(
    group_node: object,
    *,
    in_text_mode: bool,
    allow_prime_shorthand: bool = True,
) -> KaTeXNode:
    return KaTeXNode(
        "ordgroup",
        _convert_nodes_to_katex(
            getattr(group_node, "nodelist", None),
            in_text_mode=in_text_mode,
            allow_prime_shorthand=allow_prime_shorthand,
        ),
    )


def _group_text(group_node: object) -> str:
    # Extract a plain string from a group node (used for env colspec parsing).
    nodelist = getattr(group_node, "nodelist", None) or []
    text_parts: list[str] = []
    for n in nodelist:
        chars = getattr(n, "chars", None)
        if chars:
            text_parts.append(chars)
    return "".join(text_parts)


def _parse_rule_dim(text: str) -> Optional[RuleDim]:
    m = _SIZE_RE.match(text or "")
    if not m:
        return None
    sign, number, unit = m.group(1), m.group(2), m.group(3)
    try:
        val = float(f"{sign}{number}")
    except ValueError:
        return None
    return RuleDim(number=val, unit=unit)
