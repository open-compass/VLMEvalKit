import re
import sys
from typing import Tuple
from .preprocess_tabular import normalize_tabular
from .preprocess_formula import normalize_formula
# 正则表达式用于替换各种 LaTeX 环境
_ALIGN_ENV_REGEX = re.compile(
    r"\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}",
    flags=re.S,
)
_SMALLMATRIX_ENV_REGEX = re.compile(r"\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}", flags=re.S)
_HSKIP_REGEX = re.compile(r"hskip(.*?)(cm|in|pt|mm|em)")

# Operator names: keep parity with the original Node/KaTeX pipeline.
_OPERATOR_NAMES = {
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

# Match JS wrapper behavior: only rewrite \operatorname{...} (NOT \operatorname*{...}).
_OPERATORNAME_REGEX = re.compile(r"\\operatorname\s*\{([^{}]*?)\}")

def _collapse_ws(s: str) -> str:
    return " ".join(s.strip().split())

def _rewrite_operatorname(post: str) -> str:
    if "\\operatorname" not in post:
        return post

    def repl(match) -> str:
        raw = match.group(1)
        compact = re.sub(r"\s+", "", raw.strip())
        if compact in _OPERATOR_NAMES:
            return "\\" + compact
        return match.group(0)

    return _OPERATORNAME_REGEX.sub(repl, post)

def _merge_left_right(post: str) -> str:
    if "\\left" not in post and "\\right" not in post:
        return post
    tokens = post.strip().split()
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {r"\left", r"\right"} and i + 1 < len(tokens):
            merged.append(tok + tokens[i + 1])
            i += 2
            continue
        merged.append(tok)
        i += 1
    return " ".join(merged)

def _strip_left_right(post: str) -> str:
    if "\\left" not in post and "\\right" not in post:
        return post
    tokens = post.strip().split()
    stripped: list[str] = []
    for tok in tokens:
        if tok in {r"\left", r"\right"}:
            continue
        if tok.startswith(r"\left") or tok.startswith(r"\right"):
            if tok.startswith(r"\left"):
                delim = tok[len(r"\left"):]
            else:
                delim = tok[len(r"\right"):]
            if not delim or delim == ".":
                continue
            stripped.append(delim)
            continue
        stripped.append(tok)
    return " ".join(stripped)

def tokenize_latex(latex_code: str, latex_type: str = "", middle_file: str = "") -> Tuple[bool, str]:
    if not latex_code:
        return False, latex_code

    if not latex_type:
        latex_type = "tabular" if "tabular" in latex_code else "formula"

    if latex_type == "formula":
        
        prepre = latex_code
        # replace split, align with aligned
        prepre = _ALIGN_ENV_REGEX.sub(r"\\begin{aligned}\2\\end{aligned}", prepre)
        prepre = _SMALLMATRIX_ENV_REGEX.sub(r"\\begin{matrix}\2\\end{matrix}", prepre)

        post = normalize_formula(prepre, mode="normalize")
        if not post:
            return True, ""

        post = _collapse_ws(post)
        post = _rewrite_operatorname(post)
        post = _merge_left_right(post)
        # post = _strip_left_right(post)
        # Match original wrapper: remove trailing row-break before \end{array}
        post = post.replace(r"\\ \end{array}", r"\end{array}")
        
        return True, post

    elif latex_type == "tabular":
        
        code = latex_code.replace("\\\\%", "\\\\ %")
        code = code.replace(r"\%", "<PERCENTAGE_TOKEN>")
        code = code.split("%")[0]
        code = code.replace("<PERCENTAGE_TOKEN>", r"\%")
        if "\\end{tabular}" not in code:
            code += "\\end{tabular}"

        code = _HSKIP_REGEX.sub(r"hspace{\1\2}", code)

        # Original Node pipeline uses KaTeX `global_str` ("tokenize" mode) for tabular.
        post = normalize_tabular(code.replace("\r", " ").replace("\n", " "), mode="tokenize")
        if not post:
            print(f"[PYTOK] tokenize_latex(tabular): normalize_tabular returned empty", file=sys.stderr)
            return False, latex_code
        
        post = _collapse_ws(post)
        return True, post

    # 其他未识别类型返回原始内容
    print(f"[PYTOK] tokenize_latex: unrecognized latex_type '{latex_type}', returning original", file=sys.stderr)
    return False, latex_code
