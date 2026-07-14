import re

from src.core.preprocess.data_preprocess import strip_formula_delimiters, strip_formula_tags


_MATRIX_ENV_NAMES = (
    "matrix",
    "bmatrix",
    "pmatrix",
    "vmatrix",
    "Vmatrix",
    "smallmatrix",
)

_STYLE_WRAPPER_NAMES = (
    "boldsymbol",
    "mathbf",
    "mathrm",
    "mathit",
    "mathsf",
    "mathtt",
    "textbf",
)

_STYLE_WRAPPER_RE = re.compile(
    r'\\(?:' + '|'.join(_STYLE_WRAPPER_NAMES) + r')\s*\{([^{}]+)\}'
)
_OPERATORNAME_DET_RE = re.compile(r'\\operatorname\s*\{\s*det\s*\}')
_EMPTY_ARRAY_SPEC_RE = re.compile(r'\\begin\{array\}\s*\{\s*\}')
_SINGLE_COLUMN_ARRAY_RE = re.compile(
    r'^\s*\\begin\{array\}\s*\{\s*l\s*\}(?P<body>.*)\\end\{array\}\s*$',
    re.S,
)
_ALIGNED_MATRIX_RE = re.compile(
    r'^\s*\\begin\{aligned\*?\}\s*&?\s*'
    r'(?P<body>\\begin\{(?:' + '|'.join(_MATRIX_ENV_NAMES) + r')\}.*?\\end\{(?:' + '|'.join(_MATRIX_ENV_NAMES) + r')\})'
    r'\s*(?:\\\\\s*)?\\end\{aligned\*?\}\s*$',
    re.S,
)


def _needs_gt_cdm_fix(text: str) -> bool:
    formula = strip_formula_tags(strip_formula_delimiters(str(text or '')).strip())
    if not formula:
        return False
    if _EMPTY_ARRAY_SPEC_RE.search(formula):
        return True
    return False


def _contains_matrix_hint(text: str) -> bool:
    formula = str(text or '')
    if not formula:
        return False
    if any(f'\\begin{{{env}}}' in formula for env in _MATRIX_ENV_NAMES):
        return True
    if '=\\begin{bmatrix}' in formula or '=\\begin{pmatrix}' in formula:
        return True
    if '\\begin{array}' in formula and ('\\left[' in formula or '\\right]' in formula or '&' in formula):
        return True
    return False


def _is_balanced_brace_wrapper(text: str) -> bool:
    if not text or text[0] != '{' or text[-1] != '}':
        return False
    depth = 0
    for idx, ch in enumerate(text):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and idx != len(text) - 1:
                return False
    return depth == 0


def _strip_outer_braces(text: str) -> str:
    stripped = str(text or '').strip()
    while _is_balanced_brace_wrapper(stripped):
        candidate = stripped[1:-1].strip()
        if not candidate:
            break
        stripped = candidate
    return stripped


def _split_latex_rows(body: str) -> list[str]:
    begin_token = '\\begin{'
    end_token = '\\end{'
    row_sep = '\\\\'

    rows = []
    current = []
    brace_depth = 0
    env_depth = 0
    index = 0
    length = len(body)

    while index < length:
        if body.startswith(begin_token, index):
            end_index = body.find('}', index + len(begin_token))
            if end_index != -1:
                current.append(body[index:end_index + 1])
                env_depth += 1
                index = end_index + 1
                continue
        if body.startswith(end_token, index):
            end_index = body.find('}', index + len(end_token))
            if end_index != -1:
                current.append(body[index:end_index + 1])
                env_depth = max(0, env_depth - 1)
                index = end_index + 1
                continue

        if body.startswith(row_sep, index) and brace_depth == 0 and env_depth == 0:
            row = ''.join(current).strip()
            if row:
                rows.append(row)
            current = []
            index += len(row_sep)
            while index < length and body[index].isspace():
                index += 1
            if index < length and body[index] == '*':
                index += 1
            if index < length and body[index] == '[':
                bracket_end = body.find(']', index + 1)
                if bracket_end != -1:
                    index = bracket_end + 1
            continue

        char = body[index]
        current.append(char)
        if char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth = max(0, brace_depth - 1)
        index += 1

    tail = ''.join(current).strip()
    if tail:
        rows.append(tail)
    return rows


def _split_top_level_cells(text: str, separator: str) -> list[str]:
    parts = []
    current = []
    brace_depth = 0
    env_depth = 0
    index = 0
    length = len(text)
    multi_sep = separator == '\\\\'

    while index < length:
        if text.startswith('\\begin{', index):
            end_index = text.find('}', index + len('\\begin{'))
            if end_index != -1:
                current.append(text[index:end_index + 1])
                env_depth += 1
                index = end_index + 1
                continue
        if text.startswith('\\end{', index):
            end_index = text.find('}', index + len('\\end{'))
            if end_index != -1:
                current.append(text[index:end_index + 1])
                env_depth = max(0, env_depth - 1)
                index = end_index + 1
                continue

        if multi_sep:
            if text.startswith(separator, index) and brace_depth == 0 and env_depth == 0:
                parts.append(''.join(current).strip())
                current = []
                index += len(separator)
                continue
        elif text[index] == separator and brace_depth == 0 and env_depth == 0:
            parts.append(''.join(current).strip())
            current = []
            index += 1
            continue

        char = text[index]
        current.append(char)
        if char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth = max(0, brace_depth - 1)
        index += 1

    tail = ''.join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_matrix_env_cells(text: str) -> str:
    matrix_env_group = '|'.join(_MATRIX_ENV_NAMES)

    def _clean_body(body: str) -> str:
        rows = _split_top_level_cells(body, '\\\\')
        cleaned_rows = []
        for row in rows:
            cells = _split_top_level_cells(row, '&')
            cleaned_cells = [_strip_outer_braces(cell) for cell in cells]
            cleaned_rows.append('&'.join(cleaned_cells))
        return r'\\'.join(cleaned_rows)

    def _replace_array(match: re.Match) -> str:
        spec = match.group('spec')
        body = _clean_body(match.group('body'))
        return f'\\begin{{array}}{{{spec}}}{body}\\end{{array}}'

    def _replace_matrix(match: re.Match) -> str:
        env = match.group('env')
        body = _clean_body(match.group('body'))
        return f'\\begin{{{env}}}{body}\\end{{{env}}}'

    text = re.sub(
        r'\\begin\{array\}\s*\{(?P<spec>[^}]*)\}(?P<body>.*?)\\end\{array\}',
        _replace_array,
        text,
        flags=re.S,
    )
    text = re.sub(
        r'\\begin\{(?P<env>' + matrix_env_group + r')\}(?P<body>.*?)\\end\{(?P=env)\}',
        _replace_matrix,
        text,
        flags=re.S,
    )
    return text


def _unwrap_style_macros(text: str) -> str:
    previous = None
    current = str(text or '')
    while previous != current:
        previous = current
        current = _STYLE_WRAPPER_RE.sub(r'\1', current)
    current = _OPERATORNAME_DET_RE.sub(r'\\det', current)
    return current


def _collapse_simple_atom_braces(text: str) -> str:
    current = str(text or '')
    prefix_boundary = r'(^|(?<=[\s&=,\(\[\{])|(?<=\\\\))'
    tail_boundary = (
        r'(?=\s*(?:&|\\\\|,|\.|;|:|=|\)|\]|\\quad|\\qquad|\\to|\\sim|'
        r'\\end\{(?:array|' + '|'.join(_MATRIX_ENV_NAMES) + r'|aligned)\}|$))'
    )
    patterns = [
        (
            re.compile(prefix_boundary + r'\{+\s*([A-Za-z0-9.+\-]+)\s*\}+' + tail_boundary),
            r'\1\2',
        ),
        (
            re.compile(
                prefix_boundary
                + r'\{+\s*(\\(?!begin\b|end\b|left\b|right\b|frac\b|dfrac\b|text\b|operatorname\b)[A-Za-z]+)\s*\}+'
                + tail_boundary
            ),
            r'\1\2',
        ),
        (
            re.compile(prefix_boundary + r'\{+\s*([A-Za-z](?:_\{[^{}]+\})?)\s*\}+' + tail_boundary),
            r'\1\2',
        ),
        (
            re.compile(
                prefix_boundary
                + r'\{+\s*(\\(?!begin\b|end\b|left\b|right\b|frac\b|dfrac\b|text\b|operatorname\b)[A-Za-z]+(?:_\{[^{}]+\})?)\s*\}+'
                + tail_boundary
            ),
            r'\1\2',
        ),
    ]
    previous = None
    while previous != current:
        previous = current
        for pattern, replacement in patterns:
            current = pattern.sub(replacement, current)
    return current


def _sanitize_matrix_fragment(text: str) -> str:
    fragment = _strip_outer_braces(text)
    fragment = _EMPTY_ARRAY_SPEC_RE.sub(r'\\begin{array}{l}', fragment)
    fragment = fragment.replace(r'\dfrac', r'\frac')
    fragment = re.sub(r'_\{_\{([^{}]+)\}\}', r'_{\1}', fragment)
    fragment = re.sub(r'x_\{_\{([^{}]+)\}\}', r'x_{\1}', fragment)
    fragment = _unwrap_style_macros(fragment)
    fragment = _normalize_matrix_env_cells(fragment)

    matrix_env_group = '|'.join(_MATRIX_ENV_NAMES)
    fragment = re.sub(
        r'(\\begin\{(?:' + matrix_env_group + r')\})\s*\\\\',
        r'\1 ',
        fragment,
    )
    fragment = re.sub(
        r'\\\\\s*(\\end\{(?:' + matrix_env_group + r')\})',
        r' \1',
        fragment,
    )

    aligned_match = _ALIGNED_MATRIX_RE.match(fragment)
    if aligned_match:
        fragment = aligned_match.group('body').strip()
    return fragment.strip()


def sanitize_formula_for_cdm(text: str, gt_text: str = '') -> str:
    formula = strip_formula_tags(strip_formula_delimiters(str(text or '')).strip())
    if not formula:
        return ''

    matrix_context = _contains_matrix_hint(formula) or _contains_matrix_hint(strip_formula_tags(gt_text))
    if not matrix_context:
        return formula.strip()

    formula = _sanitize_matrix_fragment(formula)
    single_column_match = _SINGLE_COLUMN_ARRAY_RE.match(formula)
    if single_column_match:
        rows = [
            _sanitize_matrix_fragment(row)
            for row in _split_latex_rows(single_column_match.group('body'))
            if str(row or '').strip()
        ]
        if rows and any(_contains_matrix_hint(row) for row in rows):
            formula = r' \qquad '.join(rows)

    formula = _sanitize_matrix_fragment(formula)
    return formula.strip()


def build_matrix_cdm_variants(gt_text: str, pred_text: str) -> tuple[str, str]:
    raw_gt = strip_formula_tags(strip_formula_delimiters(str(gt_text or '')).strip())
    raw_pred = strip_formula_tags(strip_formula_delimiters(str(pred_text or '')).strip())

    if not raw_gt and not raw_pred:
        return '', ''
    if not (_contains_matrix_hint(raw_gt) or _contains_matrix_hint(raw_pred)):
        return raw_gt, ''

    sanitized_gt = sanitize_formula_for_cdm(raw_gt, raw_gt)
    gt_cdm = sanitized_gt if _needs_gt_cdm_fix(raw_gt) else raw_gt
    pred_cdm = sanitize_formula_for_cdm(raw_pred, sanitized_gt or raw_gt)
    pred_cdm_alt = pred_cdm if pred_cdm and pred_cdm != raw_pred else ''
    return gt_cdm or raw_gt, pred_cdm_alt
