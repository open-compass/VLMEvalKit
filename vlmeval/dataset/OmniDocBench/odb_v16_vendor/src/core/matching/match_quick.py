from scipy.optimize import linear_sum_assignment
# from rapidfuzz.distance import Levenshtein
import Levenshtein
from collections import defaultdict
import copy
from src.core.matching.match import (
    compute_edit_distance_matrix_new,
    get_gt_pred_lines,
    get_pred_category_type,
    match_gt2pred_timeout_safe,
)
import numpy as np
import evaluate
from collections import Counter
from Levenshtein import distance as Levenshtein_distance

import re
import time
from copy import deepcopy
from typing import List, Dict, Any
from src.core.preprocess.data_preprocess import strip_formula_delimiters, strip_formula_tags
from loguru import logger

MAX_TRUNCATED_PRED_MERGE = 160


class TruncatedMatchTimeout(RuntimeError):
    pass


# ARRAY_RE = re.compile(
#     r'\\begin\{array\}\{[^}]*\}(.*?)\\end\{array\}', re.S
# )

# def split_gt_equation_arrays(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     拆分带 \\begin{array} … \\end{array} 的 GT 字典条目。

#     - 仅针对 category_type == 'equation_isolated' 且 latex 含 array。
#     - 每行公式拆出一个新条目：
#         * 更新 'latex'
#         * 若存在 line_with_spans，则同步替换其内部 latex
#         * 'order' 由 7 --> 7.1, 7.2, …
#     """
#     output = []

#     for item in data:
#         # 只处理满足条件的字典
#         if (item.get("category_type") == "equation_isolated" and
#                 "\\begin{array" in item.get("latex", "")):

#             # 抽取 array 内部内容
#             match = ARRAY_RE.search(item["latex"])
#             if match:
#                 body = match.group(1)           # 去掉 array 外壳
#                 # 按 LaTeX 行分隔符 \\\\ 拆分
#                 lines = [ln.strip() for ln in re.split(r'\\\\', body) if ln.strip()]

#                 base_order = float(item["order"])  # 7 -> 7.0，可兼容 float/int

#                 for idx, line in enumerate(lines, start=1):
#                     new_item = deepcopy(item)
#                     new_item["latex"] = f"\\[{line}\\]"
#                     new_item["order"] = round(base_order + idx / 10, 1)
#                     output.append(new_item)
#                 continue  # 跳过把原 item 加入
#         # 其它情况不修改
#         output.append(item)

#     return output

# def _wrap(line: str) -> str:
#     """给单行公式重新包 \\[ ... \\]"""
#     return f"\\[{line.strip()}\\]"

# def split_equation_arrays(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     处理 category_type == 'equation_isolated' 且含 \\begin{array} … 的条目：
#     * 拆分多行公式
#     * 重新包装 content
#     * **重计算 position / positions**
#     """
#     out: List[Dict[str, Any]] = []

#     for item in data:
#         if (item.get("category_type") == "equation_isolated" and
#                 "\\begin{array" in item.get("content", "")):

#             content = item["content"]
#             m = ARRAY_RE.search(content)
#             if not m:
#                 out.append(item)
#                 continue

#             body = m.group(1)
#             lines = [ln.strip() for ln in re.split(r'\\\\', body) if ln.strip()]

#             # 全局起始字符索引
#             pos_key = "position" if "position" in item else "positions"
#             global_start = item[pos_key][0]

#             # array 正文在原 content 内的起点
#             body_start_in_content = m.start(1)

#             search_from = 0  # 在 body 中的游标
#             for ln in lines:
#                 # 在 body 中找到当前行的偏移
#                 idx_in_body = body.find(ln, search_from)
#                 if idx_in_body == -1:
#                     # 不太可能发生；保守处理
#                     idx_in_body = search_from
#                 search_from = idx_in_body + len(ln)  # 更新游标

#                 # 计算全局索引
#                 line_start_global = global_start + body_start_in_content + idx_in_body
#                 line_end_global   = line_start_global + len(ln) - 1

#                 new_item = deepcopy(item)
#                 new_item["content"] = _wrap(ln)
#                 new_item[pos_key]   = [line_start_global, line_end_global]

#                 out.append(new_item)

#             # 拆分完成，不保留原条目
#             continue

#         # 其它条目直接加入
#         out.append(item)

#     return out

ARRAY_RE = re.compile(
    r'\\\\begin\{array\}\{(?P<spec>[^}]*)\}(?P<body>.*?)\\\\end\{array\}',
    re.S
)
MULTILINE_ENV_NAMES = [
    'array', 'aligned', 'align', 'align*', 'alignedat', 'alignedat*',
    'split', 'gather', 'gather*', 'gathered', 'multline', 'multline*',
    'flalign', 'flalign*', 'eqnarray', 'eqnarray*', 'cases'
]
MULTILINE_ENV_RE = re.compile(
    r'^\\begin\{(?P<env>' + '|'.join(re.escape(name) for name in MULTILINE_ENV_NAMES) + r')\}'
    r'\s*(?P<extra>(?:\{[^{}]*\}\s*)*)',
    re.S,
)
MULTILINE_TRAILING_SPACING_RE = re.compile(r'(?:\\qquad|\\quad|\\,|\\;|\\!|\\>|\\:|~)+')
MULTILINE_LABEL_RE = re.compile(r'\\label\s*\{[^{}]*\}')
MULTILINE_TAG_RE = re.compile(r'\\tag\*?\s*\{[^{}]*\}')
ALIGNED_ENV_NAMES = {'aligned', 'align', 'align*', 'alignedat', 'alignedat*'}
CASES_LOGIC_MARKER_RE = re.compile(
    r'\\(?:Rightarrow|Leftrightarrow|Longrightarrow|Longleftrightarrow|'
    r'longrightarrow|longleftrightarrow|implies|therefore|because|'
    r'uparrow|downarrow|Updownarrow|forall|exists)\b|单调|否定'
)
NESTED_MULTILINE_ENV_RE = re.compile(
    r'\\begin\{(?:array|aligned|align\*?|alignedat\*?|split|gathered?|multline\*?|flalign\*?|eqnarray\*?|cases)\}'
)
ALIGNED_CASE_LIKE_LEAD_RE = re.compile(
    r'^(?:\(?\d+\)?|\\(?:Rightarrow|Leftrightarrow|Longrightarrow|Longleftrightarrow|implies|therefore|uparrow|downarrow|Updownarrow|leftarrow|rightarrow))'
)
ALIGNED_EXPLANATORY_TEXT_RE = re.compile(r'\\(?:text|mathrm|operatorname)\s*\{')


def is_all_l(spec: str) -> bool:
    """检查是否为适合按行拆分的 array 公式，放开常见的对齐型两列 array。"""
    spec = _normalize_array_spec(spec)
    if not spec or any(ch not in {'l', 'c', 'r'} for ch in spec):
        return False
    if len(spec) == 1:
        return True
    if len(spec) == 2 and ('l' in spec or 'r' in spec):
        return True
    return False


def _normalize_array_spec(spec: str) -> str:
    normalized = re.sub(r'\s+|\|', '', str(spec or ''))
    normalized = re.sub(r'@{[^}]*}', '', normalized)
    normalized = re.sub(r'!{[^}]*}', '', normalized)
    return normalized


def _find_matching_env_end(text: str, env_name: str, search_start: int) -> int:
    begin_token = f'\\begin{{{env_name}}}'
    end_token = f'\\end{{{env_name}}}'
    depth = 1
    index = search_start

    while index < len(text):
        next_begin = text.find(begin_token, index)
        next_end = text.find(end_token, index)
        if next_end < 0:
            return -1
        if 0 <= next_begin < next_end:
            depth += 1
            index = next_begin + len(begin_token)
            continue
        depth -= 1
        if depth == 0:
            return next_end
        index = next_end + len(end_token)

    return -1


def _extract_multiline_env_parts(formula_text: str):
    raw_formula = str(formula_text or '')
    stripped_formula = strip_formula_delimiters(raw_formula).strip()
    if not stripped_formula:
        return None

    prefix_match = re.match(r'^(?P<prefix>(?:\\left\s*\\\{|\\left\.))\s*', stripped_formula)
    candidate_formula = stripped_formula[prefix_match.end():].lstrip() if prefix_match else stripped_formula

    suffix_match = re.search(r'(?P<suffix>\s*\\right(?:\.|\\\}))\s*$', candidate_formula)
    if suffix_match:
        candidate_core = candidate_formula[:suffix_match.start()].rstrip()
    else:
        candidate_core = candidate_formula

    start_match = MULTILINE_ENV_RE.match(candidate_core)
    if not start_match:
        return None

    env_name = start_match.group('env')
    extra = start_match.group('extra')
    end_tag = f'\\end{{{env_name}}}'
    end_index = _find_matching_env_end(candidate_core, env_name, start_match.end())
    if end_index < start_match.end():
        return None

    array_spec = ''
    if env_name == 'array':
        extra_groups = re.findall(r'\{([^{}]*)\}', extra)
        if extra_groups:
            array_spec = extra_groups[0]

    body = candidate_core[start_match.end():end_index]
    raw_rows = _split_latex_rows(body)
    has_outer_brace_wrapper = bool(prefix_match or suffix_match)
    trailing = candidate_core[end_index + len(end_tag):]
    trailing_suffix = _strip_multiline_end_labels(trailing).strip()

    body_start_in_raw = raw_formula.find(body)
    if body_start_in_raw < 0:
        body_start_in_raw = 0

    trailing_suffix_start_in_raw = -1
    trailing_suffix_end_in_raw = -1
    if trailing_suffix:
        end_tag_in_raw = raw_formula.find(end_tag, body_start_in_raw + len(body))
        if end_tag_in_raw >= 0:
            search_start = end_tag_in_raw + len(end_tag)
            trailing_suffix_start_in_raw = raw_formula.find(trailing_suffix, search_start)
            if trailing_suffix_start_in_raw >= 0:
                trailing_suffix_end_in_raw = trailing_suffix_start_in_raw + len(trailing_suffix)

    return {
        'env': env_name,
        'extra': extra,
        'array_spec': array_spec,
        'body': body,
        'raw_rows': raw_rows,
        'has_outer_brace_wrapper': has_outer_brace_wrapper,
        'trailing': trailing,
        'trailing_suffix': trailing_suffix,
        'body_start_in_raw': body_start_in_raw,
        'trailing_suffix_start_in_raw': trailing_suffix_start_in_raw,
        'trailing_suffix_end_in_raw': trailing_suffix_end_in_raw,
    }


def _rstrip_multiline_spacing_tokens(text: str) -> str:
    working = str(text or '').rstrip()
    while True:
        updated = re.sub(r'(?:\\qquad|\\quad|\\,|\\;|\\!|\\>|\\:|~)\s*$', '', working)
        updated = updated.rstrip()
        if updated == working:
            return working
        working = updated


def _strip_multiline_end_labels(trailing: str) -> str:
    working = MULTILINE_LABEL_RE.sub('', str(trailing or ''))
    while True:
        stripped = working.rstrip()
        if not stripped:
            return ''

        tag_match = MULTILINE_TAG_RE.search(stripped)
        if tag_match and tag_match.end() == len(stripped):
            working = stripped[:tag_match.start()].rstrip()
            working = _rstrip_multiline_spacing_tokens(working)
            continue

        if stripped[-1] not in {')', ']'}:
            return stripped

        opening = '(' if stripped[-1] == ')' else '['
        start = stripped.rfind(opening)
        if start < 0:
            return stripped

        label_body = stripped[start + 1:-1].strip()
        prefix = stripped[:start]
        if not _is_supported_multiline_end_label(label_body, prefix):
            return stripped

        working = _rstrip_multiline_spacing_tokens(prefix)


def _is_supported_multiline_end_label(label_body: str, prefix: str) -> bool:
    label_body = str(label_body or '').strip()
    if not label_body:
        return False
    if prefix and not (prefix[-1].isspace() or MULTILINE_TRAILING_SPACING_RE.search(prefix[-10:])):
        return False
    if re.fullmatch(r'[a-z]{1,3}', label_body):
        return False
    if re.fullmatch(r'[A-Za-z0-9.\-#]+', label_body):
        return True
    if re.fullmatch(r'(?:\\[A-Za-z#]+)+', label_body):
        return True
    return False


def _is_escaped_char(text: str, index: int) -> bool:
    backslash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == '\\':
        backslash_count += 1
        cursor -= 1
    return backslash_count % 2 == 1


def _split_top_level_alignment_cells(line: str):
    cells = []
    current = []
    env_depth = 0
    index = 0
    length = len(line)

    while index < length:
        if line.startswith('\\begin{', index):
            end_index = line.find('}', index + len('\\begin{'))
            if end_index != -1:
                current.append(line[index:end_index + 1])
                env_depth += 1
                index = end_index + 1
                continue
        if line.startswith('\\end{', index):
            end_index = line.find('}', index + len('\\end{'))
            if end_index != -1:
                current.append(line[index:end_index + 1])
                env_depth = max(0, env_depth - 1)
                index = end_index + 1
                continue
        if line[index] == '&' and env_depth == 0 and not _is_escaped_char(line, index):
            cells.append(''.join(current).strip())
            current = []
            index += 1
            continue
        current.append(line[index])
        index += 1

    cells.append(''.join(current).strip())
    return cells


def _strip_leading_alignment_markers(line: str) -> str:
    stripped = str(line or '').lstrip()
    while stripped.startswith('&'):
        stripped = stripped[1:].lstrip()
    return stripped


def _aligned_row_has_independent_lhs(line: str) -> bool:
    cells = _split_top_level_alignment_cells(line)
    if len(cells) <= 1:
        return False
    return bool(strip_formula_tags(cells[0]).strip())


def _is_cases_like_aligned_row(line: str) -> bool:
    lead = strip_formula_tags(_strip_leading_alignment_markers(line)).strip()
    if not lead:
        return False
    return bool(ALIGNED_CASE_LIKE_LEAD_RE.match(lead))


def _aligned_row_has_explanatory_text(line: str) -> bool:
    return bool(ALIGNED_EXPLANATORY_TEXT_RE.search(_strip_leading_alignment_markers(line)))


def _cases_row_has_top_level_alignment(line: str) -> bool:
    return len(_split_top_level_alignment_cells(line)) > 1


def _cases_row_has_logic_marker(line: str) -> bool:
    stripped = _strip_leading_alignment_markers(line)
    return bool(CASES_LOGIC_MARKER_RE.search(stripped))


def _should_split_cases_block(body: str, raw_rows) -> bool:
    if len(raw_rows) <= 1:
        return False

    if any(_cases_row_has_top_level_alignment(row) for row in raw_rows):
        return False

    has_logic_marker = bool(CASES_LOGIC_MARKER_RE.search(str(body or '')))
    if not has_logic_marker:
        return False

    has_nested_multiline_env = bool(NESTED_MULTILINE_ENV_RE.search(str(body or '')))
    if has_nested_multiline_env:
        return True

    if any(_cases_row_has_logic_marker(row) for row in raw_rows[1:]):
        return True

    if any(_aligned_row_has_explanatory_text(row) for row in raw_rows[1:]):
        return True

    return _cases_row_has_logic_marker(raw_rows[0])


def _should_normalize_cases_block(body: str, raw_rows, trailing_context: str = '') -> bool:
    if len(raw_rows) <= 1:
        return False

    if any(_cases_row_has_top_level_alignment(row) for row in raw_rows):
        return False

    trailing_context = str(trailing_context or '').lstrip()
    has_logic_marker = (
        bool(CASES_LOGIC_MARKER_RE.search(str(body or '')))
        or bool(CASES_LOGIC_MARKER_RE.search(trailing_context))
    )
    has_following_multiline_env = bool(MULTILINE_ENV_RE.match(trailing_context))
    if not has_logic_marker and not has_following_multiline_env:
        return False

    if bool(NESTED_MULTILINE_ENV_RE.search(str(body or ''))):
        return True

    if any(_cases_row_has_logic_marker(row) for row in raw_rows):
        return True

    if any(_aligned_row_has_explanatory_text(row) for row in raw_rows[1:]):
        return True

    return has_logic_marker or has_following_multiline_env


def _should_keep_aligned_block_together(body: str, raw_rows, has_outer_brace_wrapper: bool) -> bool:
    has_nested_multiline_env = bool(NESTED_MULTILINE_ENV_RE.search(str(body or '')))
    if has_nested_multiline_env and any(_is_cases_like_aligned_row(row) for row in raw_rows[1:]):
        return True

    if not has_outer_brace_wrapper:
        return False

    if has_nested_multiline_env:
        return True

    independent_lhs_rows = sum(1 for row in raw_rows if _aligned_row_has_independent_lhs(row))
    if independent_lhs_rows >= 2:
        return True

    if any(_aligned_row_has_explanatory_text(row) for row in raw_rows[1:]):
        return True

    if any(_is_cases_like_aligned_row(row) for row in raw_rows[1:]):
        return True

    return False


def _should_keep_array_block_together(spec: str, body: str, raw_rows, has_outer_brace_wrapper: bool) -> bool:
    normalized_spec = _normalize_array_spec(spec)
    if not normalized_spec:
        return False

    has_nested_multiline_env = bool(NESTED_MULTILINE_ENV_RE.search(str(body or '')))
    if has_nested_multiline_env and any(_is_cases_like_aligned_row(row) for row in raw_rows[1:]):
        return True

    if has_outer_brace_wrapper and set(normalized_spec) <= {'l'} and len(raw_rows) > 1:
        return True

    return False


def _find_multiline_env_match(formula_text: str):
    match_info = _extract_multiline_env_parts(formula_text)
    if not match_info:
        return None

    env_name = match_info['env']
    extra = match_info['extra']
    array_spec = match_info['array_spec']
    body = match_info['body']
    raw_rows = match_info['raw_rows']
    has_outer_brace_wrapper = match_info['has_outer_brace_wrapper']
    if env_name == 'cases':
        if not _should_split_cases_block(body, raw_rows):
            return None
    if env_name in ALIGNED_ENV_NAMES:
        if _should_keep_aligned_block_together(body, raw_rows, has_outer_brace_wrapper):
            return None
    if env_name == 'array':
        if not array_spec or not is_all_l(array_spec):
            return None
        if _should_keep_array_block_together(array_spec, body, raw_rows, has_outer_brace_wrapper):
            return None

    trailing_suffix = match_info['trailing_suffix']
    if trailing_suffix and len(_split_latex_rows(trailing_suffix)) > 1:
        return None

    return {
        'env': env_name,
        'extra': extra,
        'body': body,
        'body_start_in_raw': match_info['body_start_in_raw'],
        'trailing_suffix': trailing_suffix,
        'trailing_suffix_start_in_raw': match_info['trailing_suffix_start_in_raw'],
        'trailing_suffix_end_in_raw': match_info['trailing_suffix_end_in_raw'],
    }


def _find_bare_multiline_match(formula_text: str):
    raw_formula = str(formula_text or '')
    stripped_formula = strip_formula_delimiters(raw_formula).strip()
    if not stripped_formula or '\\\\' not in stripped_formula:
        return None

    lines = _split_latex_rows(stripped_formula)
    if len(lines) <= 1:
        return None

    body_start_in_raw = raw_formula.find(stripped_formula)
    if body_start_in_raw < 0:
        body_start_in_raw = 0

    return {
        'env': 'bare_multiline',
        'extra': '',
        'body': stripped_formula,
        'body_start_in_raw': body_start_in_raw,
    }


def should_split_array_formula(formula_text: str) -> bool:
    """向后兼容旧接口：当前含义扩展为“是否应拆分整个多行公式环境”"""
    return _find_multiline_env_match(formula_text) is not None


def _split_latex_rows(body: str):
    begin_token = '\\begin{'
    end_token = '\\end{'
    row_sep = '\\\\'

    lines = []
    current = []
    brace_depth = 0
    env_depth = 0
    index = 0
    length = len(body)

    while index < length:
        if body.startswith(begin_token, index):
            end_index = body.find('}', index + len(begin_token))
            if end_index != -1:
                token = body[index:end_index + 1]
                current.append(token)
                env_depth += 1
                index = end_index + 1
                continue
        if body.startswith(end_token, index):
            end_index = body.find('}', index + len(end_token))
            if end_index != -1:
                token = body[index:end_index + 1]
                current.append(token)
                env_depth = max(0, env_depth - 1)
                index = end_index + 1
                continue

        if body.startswith(row_sep, index) and brace_depth == 0 and env_depth == 0:
            line = ''.join(current).strip()
            if line:
                lines.append(line)
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
        if char == '{' and not _is_escaped_char(body, index):
            brace_depth += 1
        elif char == '}' and not _is_escaped_char(body, index):
            brace_depth = max(0, brace_depth - 1)
        index += 1

    tail = ''.join(current).strip()
    if tail:
        lines.append(tail)
    return lines


def _split_multiline_formula_lines(formula_text: str):
    match_info = _find_multiline_env_match(formula_text)
    if not match_info:
        match_info = _find_bare_multiline_match(formula_text)
    if not match_info:
        return None
    body = match_info['body']
    line_infos = []
    raw_lines = _split_latex_rows(body)
    for raw_line in raw_lines:
        line = strip_formula_tags(raw_line).strip()
        if not line:
            continue
        line_infos.append({
            'raw': raw_line,
            'content': line,
        })
    if len(line_infos) <= 1:
        return None
    trailing_suffix = strip_formula_tags(match_info.get('trailing_suffix', '')).strip()
    if trailing_suffix:
        line_infos[-1]['content'] = f"{line_infos[-1]['content']} {trailing_suffix}".strip()
    return match_info, line_infos


def _normalize_logic_brace_blocks(text: str):
    working = str(text or '')
    if not working:
        return working, False

    pieces = []
    cursor = 0
    changed = False
    while cursor < len(working):
        begin_index = working.find('\\begin{', cursor)
        if begin_index < 0:
            pieces.append(working[cursor:])
            break

        pieces.append(working[cursor:begin_index])
        start_match = MULTILINE_ENV_RE.match(working[begin_index:])
        if not start_match:
            pieces.append(working[begin_index:begin_index + len('\\begin{')])
            cursor = begin_index + len('\\begin{')
            continue

        env_name = start_match.group('env')
        end_tag = f'\\end{{{env_name}}}'
        block_start = begin_index
        body_start = block_start + start_match.end()
        end_index = _find_matching_env_end(working, env_name, body_start)
        if end_index < body_start:
            pieces.append(working[begin_index:])
            break

        body = working[body_start:end_index]
        normalized_body, body_changed = _normalize_logic_brace_blocks(body)
        trailing_context = working[end_index + len(end_tag):end_index + len(end_tag) + 200]

        if env_name == 'cases':
            raw_rows = _split_latex_rows(normalized_body)
            if _should_normalize_cases_block(normalized_body, raw_rows, trailing_context=trailing_context):
                pieces.append(
                    '\\left\\{ \\begin{array}{l} '
                    + normalized_body.strip()
                    + ' \\end{array} \\right.'
                )
                changed = True
            else:
                pieces.append(working[block_start:body_start] + normalized_body + end_tag)
                changed = changed or body_changed
        else:
            pieces.append(working[block_start:body_start] + normalized_body + end_tag)
            changed = changed or body_changed

        cursor = end_index + len(end_tag)

    return ''.join(pieces), changed


def normalize_logic_brace_formula(formula_text: str) -> str:
    normalized_formula, changed = _normalize_logic_brace_blocks(formula_text)
    if not changed:
        return str(formula_text or '')
    return normalized_formula


def _formula_sub_order(base_order: float, idx: int) -> float:
    return float(base_order) + idx / 1000.0


def split_gt_equation_arrays(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将整个公式本体就是多行环境（array/aligned/align/split/...）的 GT 公式按行拆分。"""
    output = []

    for item in data:
        if item.get("category_type") != "equation_isolated":
            output.append(item)
            continue

        split_result = _split_multiline_formula_lines(item.get("latex", ""))
        if not split_result:
            output.append(item)
            continue

        _, line_infos = split_result
        base_order = float(item["order"])
        for idx, line_info in enumerate(line_infos, start=1):
            new_item = deepcopy(item)
            new_item["latex"] = f"\\[{line_info['content']}\\]"
            new_item["order"] = _formula_sub_order(base_order, idx)
            output.append(new_item)

    return output


def _wrap(line: str) -> str:
    """给单行公式重新包 \\[ ... \\]"""
    return f"\\[{line.strip()}\\]"


def split_equation_arrays(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将整个公式本体就是多行环境（array/aligned/align/split/...）的预测公式按行拆分，并重算位置。"""
    out: List[Dict[str, Any]] = []

    for item in data:
        if item.get("category_type") != "equation_isolated":
            out.append(item)
            continue

        content = item.get("content", "")
        split_result = _split_multiline_formula_lines(content)
        if not split_result:
            out.append(item)
            continue

        match_info, line_infos = split_result
        pos_key = "position" if "position" in item else "positions"
        global_start = item[pos_key][0]
        body = match_info['body']
        body_start_in_content = match_info['body_start_in_raw']

        search_from = 0
        for idx, line_info in enumerate(line_infos):
            raw_line = strip_formula_tags(line_info['raw']).strip()
            idx_in_body = body.find(raw_line, search_from)
            if idx_in_body == -1:
                idx_in_body = search_from
            search_from = idx_in_body + len(raw_line)

            line_start_global = global_start + body_start_in_content + idx_in_body
            line_end_global = line_start_global + len(raw_line) - 1
            if idx == len(line_infos) - 1 and match_info.get('trailing_suffix_end_in_raw', -1) > 0:
                line_end_global = global_start + match_info['trailing_suffix_end_in_raw'] - 1

            new_item = deepcopy(item)
            new_item["content"] = _wrap(line_info['content'])
            new_item[pos_key] = [line_start_global, line_end_global]
            out.append(new_item)

    return out

def sort_by_position_skip_inline(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    先按 position[0] 从小到大排序；
    若 fine_category_type == 'equation_inline'，则统一放到最后，
    并保持它们在原列表中的相对顺序（稳定排序）。
    """
    # enumerate 保留原始顺序索引，用于 equation_inline “并列时” 的稳定性
    return sorted(
        enumerate(items),
        key=lambda pair: (
            pair[1].get('fine_category_type') == 'equation_inline',  # False < True
            pair[1]['position'][0],                                   # 位置起点
            pair[0]                                                   # 原序号，确保稳定
        )
    )
def match_gt2pred_quick(gt_items, pred_items, line_type, img_name, truncated_timeout_sec=None, fallback_short_line_max_chars=None, fallback_target_chunk_chars=None, fallback_max_chunk_chars=None, fallback_max_chunk_span=8, fallback_order_window=None, fallback_order_penalty=0.08, split_pred_formula=True):

    gt_items = split_gt_equation_arrays(gt_items)
    
    # pred_items = sorted(pred_items, key=lambda x: x['position'][0])
    pred_items = [pair[1] for pair in sort_by_position_skip_inline(pred_items)]

    if split_pred_formula:
        pred_items = split_equation_arrays(pred_items)

    # gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines= get_gt_pred_lines(gt_items, pred_items, line_type)
    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines, gt_items, pred_items = get_gt_pred_lines(gt_items, pred_items, None)
    all_gt_indices = set(range(len(norm_gt_lines)))  
    all_pred_indices = set(range(len(norm_pred_lines)))  
    
    if not norm_gt_lines:
        match_list = []
        for pred_idx in range(len(norm_pred_lines)):
            match_list.append({
                'gt_idx': [""],
                'gt': "",
                'pred_idx': [pred_idx],
                'pred': pred_lines[pred_idx],
                'gt_position': [""],
                'pred_position': pred_items[pred_idx]['position'][0],
                'norm_gt': "",
                'norm_pred': norm_pred_lines[pred_idx],
                'gt_category_type': "",
                'pred_category_type': get_pred_category_type(pred_idx, pred_items),
                'gt_attribute': [{}],
                'edit': 1,
                'img_id': img_name
            })
        return match_list
    elif not norm_pred_lines:
        match_list = []
        for gt_idx in range(len(norm_gt_lines)):
            match_list.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
                'pred_position': "",
                'norm_gt': norm_gt_lines[gt_idx],
                'norm_pred': "",
                'gt_category_type': gt_cat_list[gt_idx],
                'pred_category_type': "",
                'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
                'edit': 1,
                'img_id': img_name
            })
        return match_list
    elif len(norm_gt_lines) == 1 and len(norm_pred_lines) == 1:
        edit_distance = Levenshtein_distance(norm_gt_lines[0], norm_pred_lines[0])
        normalized_edit_distance = edit_distance / max(len(norm_gt_lines[0]), len(norm_pred_lines[0]))
        return [{
            'gt_idx': [0],
            'gt': gt_lines[0],
            'pred_idx': [0],
            'pred': pred_lines[0],
            'gt_position': [gt_items[0].get('order') if gt_items[0].get('order') else gt_items[0].get('position', [""])[0]],
            'pred_position': pred_items[0]['position'][0],
            'norm_gt': norm_gt_lines[0],
            'norm_pred': norm_pred_lines[0],
            'gt_category_type': gt_cat_list[0],
            'pred_category_type': get_pred_category_type(0, pred_items),
            'gt_attribute': [gt_items[0].get("attribute", {})],
            'edit': normalized_edit_distance,
            'img_id': img_name
        }]
    
    # match category ignore first
    ignores = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 
               'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
    
    ignore_gt_lines = []
    ignores_ori_gt_lines= []
    ignores_gt_items = []
    ignore_gt_idxs = []
    ignores_gt_cat_list = []
    
    no_ignores_gt_lines = []
    no_ignores_ori_gt_lines = []
    no_ignores_gt_idxs = []
    no_ignores_gt_items = []
    no_ignores_gt_cat_list = []

    for i, line in enumerate(norm_gt_lines):
        if gt_cat_list[i] in ignores:
            ignore_gt_lines.append(line)
            ignores_ori_gt_lines.append(gt_lines[i])
            ignores_gt_items.append(gt_items[i])
            ignore_gt_idxs.append(i)
            ignores_gt_cat_list.append(gt_cat_list[i])
        else:
            no_ignores_gt_lines.append(line)
            no_ignores_ori_gt_lines.append(gt_lines[i])
            no_ignores_gt_items.append(gt_items[i])
            no_ignores_gt_cat_list.append(gt_cat_list[i])
            no_ignores_gt_idxs.append(i)

    # print("-------------ignore_gt_lines-------------------")
    # for idx, line in zip(ignore_idx,ignore_gt_lines):
    #     print(f"{gt_cat_list[idx]}: {line}")
    
    # print("-------------no_ignores_gt_lines-------------------")
    # for line in no_ignores_gt_lines:
    #     print(line)

    ignore_pred_idxs = []
    ignore_pred_lines = []
    ignores_pred_items = []
    ignores_ori_pred_lines = []

    merged_ignore_results = []

    if len(ignore_gt_lines) > 0:
        
        ignore_matches_dict = {}

        ignore_matrix = compute_edit_distance_matrix_new(ignore_gt_lines, norm_pred_lines)
        # print("-------------ignore_matrix-------------")
        # print(ignore_matrix)
        
        ignores_gt_indices = set(range(len(ignore_gt_lines)))  
        ignores_pred_indices = set(range(len(ignore_pred_lines)))

        ignore_matches = np.argwhere(ignore_matrix < 0.25) 
        # print("-------------ignore_matches-------------")
        # print(ignore_matches)
        if len(ignore_matches) > 0:
            ignore_pred_idxs = [_[1] for _ in ignore_matches]
            ignore_gt_matched_idxs = [ignore_gt_idxs[_[0]] for _ in ignore_matches]
            # print("-------------ignore_pred_idxs-------------")
            # print(ignore_pred_idxs)
            # print("-------------ignore_gt_matched_idxs-------------")
            # print(ignore_gt_matched_idxs)

            for i in ignore_pred_idxs:
                ignore_pred_lines.append(norm_pred_lines[i])
                ignores_ori_pred_lines.append(pred_lines[i])
                ignores_pred_items.append(pred_items[i])
            # print("-------------ignore_pred_lines-------------")
            # for i in ignore_pred_lines:
            #     print(i)

                ignores_gt_indices = set(range(len(ignore_gt_lines)))  
                ignores_pred_indices = set(range(len(ignore_pred_lines))) 

            for idx, i in enumerate(ignore_matches):
                ignore_matches_dict[i[0]] = {
                    'pred_indices': [idx],
                    'edit_distance': ignore_matrix[i[0]][i[1]]
                }
            # print("-------------ignore_matches_dict-------------")
            # print(ignore_matches_dict)

        ignore_final_matches = merge_matches(ignore_matches_dict, {})
        # print("-------------ignore_final_matches-------------")
        # print(ignore_final_matches)
        
        recalculate_edit_distances(ignore_final_matches, {}, ignore_gt_lines, ignore_pred_lines)
        # print("-------------recalculate_ignore_final_matches-------------")
        # print(ignore_final_matches)

        converted_ignore_results = convert_final_matches(ignore_final_matches, ignore_gt_lines, ignore_pred_lines)
        # print("-------------converted_ignore_results-------------")
        # for i in converted_ignore_results:
        #     print(i)
            
        merged_ignore_results = merge_duplicates_add_unmatched(converted_ignore_results, ignore_gt_lines, ignore_pred_lines, ignores_ori_gt_lines, ignores_ori_pred_lines, ignores_gt_indices, ignores_pred_indices)

        for entry in merged_ignore_results:
            entry['gt_idx'] = [entry['gt_idx']] if not isinstance(entry['gt_idx'], list) else entry['gt_idx']
            entry['pred_idx'] = [entry['pred_idx']] if not isinstance(entry['pred_idx'], list) else entry['pred_idx']
            entry['gt_position'] = [ignores_gt_items[_].get('order') if ignores_gt_items[_].get('order') else ignores_gt_items[_].get('position', [""])[0] for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [""]
            entry['pred_position'] = ignores_pred_items[entry['pred_idx'][0]]['position'][0] if entry['pred_idx'] != [""] else "" 
            entry['gt'] = ''.join([ignores_ori_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
            entry['pred'] = ''.join([ignores_ori_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""
            entry['norm_gt'] = ''.join([ignore_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
            entry['norm_pred'] = ''.join([ignore_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""

            if entry['gt_idx'] != [""]:
                ignore_type = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
                gt_cagegory_clean = [ignores_gt_cat_list[_] for _ in entry['gt_idx'] if ignores_gt_cat_list[_] not in ignore_type] 
                if gt_cagegory_clean:
                    entry['gt_category_type'] = Counter(gt_cagegory_clean).most_common(1)[0][0] 
                else:
                    entry['gt_category_type'] = Counter([ignores_gt_cat_list[_] for _ in entry['gt_idx']]).most_common(1)[0][0] 
            else:
                entry['gt_category_type'] = ""
                entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], ignores_pred_items) if entry['pred_idx'] != [""] else ""
                if entry['pred_category_type'] == 'equation_inline':
                    merged_ignore_results.remove(entry)
            entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], ignores_pred_items) if entry['pred_idx'] != [""] else ""
            entry['gt_attribute'] = [ignores_gt_items[_].get("attribute", {}) for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [{}] 
            entry['img_id'] = img_name
        
        for entry in merged_ignore_results:
            if isinstance(entry['gt_idx'], list) and entry['gt_idx'] != [""]:
                gt_idx = []
                for i in entry['gt_idx']:
                    gt_idx.append(ignore_gt_idxs[i])
                entry['gt_idx'] = gt_idx
            if isinstance(entry['pred_idx'], list) and entry['pred_idx'] != [""]:
                pred_idx = []
                for i in entry['pred_idx']:
                    pred_idx.append(int(ignore_pred_idxs[i]))
                entry['pred_idx'] = pred_idx

        # print("-------------merged_ignore_results-------------")
        # for i in merged_ignore_results:
        #     print(i)

    no_ignores_pred_lines = []
    no_ignores_ori_pred_lines = []
    no_ignores_pred_indices = []
    no_ignores_pred_items = []
    no_ignore_pred_idxs = []

    for idx, line in enumerate(norm_pred_lines):
        if not idx in ignore_pred_idxs:
            no_ignores_pred_lines.append(line)
            no_ignores_ori_pred_lines.append(pred_lines[idx])
            # no_ignores_pred_indices.append(idx)
            no_ignores_pred_items.append(pred_items[idx])
            no_ignore_pred_idxs.append(idx)
    
    # initialize new indices for lines without ignore categories
    no_ignores_gt_indices = set(range(len(no_ignores_gt_lines)))  
    no_ignores_pred_indices = set(range(len(no_ignores_pred_lines)))  
    
    # exclude ignore categories
    cost_matrix = compute_edit_distance_matrix_new(no_ignores_gt_lines, no_ignores_pred_lines)
    # print("-------------cost matrix-------------")
    # print(cost_matrix)

    try:
        matched_col_idx, row_ind, cost_list = cal_final_match(
            cost_matrix,
            no_ignores_gt_lines,
            no_ignores_pred_lines,
            truncated_timeout_sec=truncated_timeout_sec,
        )
    except TruncatedMatchTimeout:
        print(
            f"[quick-match-timeout] {img_name}: deal_with_truncated exceeded {truncated_timeout_sec}s, fallback to chunked Hungarian",
            flush=True,
        )
        return match_gt2pred_timeout_safe(
            gt_items,
            pred_items,
            img_name,
            short_line_max_chars=fallback_short_line_max_chars,
            target_chunk_chars=fallback_target_chunk_chars,
            max_chunk_chars=fallback_max_chunk_chars,
            max_chunk_span=fallback_max_chunk_span,
            order_window=fallback_order_window,
            order_penalty=fallback_order_penalty,
            fallback_reason='quick_match_timeout',
        )
    # print("-------------matched_col_idx-------------")
    # print(matched_col_idx)
    
    # print("-------------gt_row_ind-------------")
    # print(row_ind)

    # print("-------------cost_list-------------")
    # print(cost_list)
        
    gt_lens_dict, pred_lens_dict = initialize_indices(no_ignores_gt_lines, no_ignores_pred_lines)
    # print("-------------gt_lens_dict-------------")
    # print(gt_lens_dict)

    # print("-------------pred_lens_dict-------------")
    # print(pred_lens_dict)
    
    matches, unmatched_gt_indices, unmatched_pred_indices = process_matches(matched_col_idx, row_ind, cost_list, no_ignores_gt_lines, no_ignores_pred_lines, no_ignores_ori_pred_lines)

    # print("-------------matches-------------")
    # print(matches)

    # print("-------------unmatched_gt_indices-------------")
    # print(unmatched_gt_indices)

    # print("-------------unmatched_pred_indices-------------")
    # print(unmatched_pred_indices)
    
    matching_dict = fuzzy_match_unmatched_items(
        unmatched_gt_indices,
        no_ignores_gt_lines,
        no_ignores_pred_lines,
        gt_cat_list=no_ignores_gt_cat_list,
    )
    # print("-------------matching_dict-------------")
    # print(matching_dict)
    
    final_matches = merge_matches(matches, matching_dict)
    # print("-------------final_matches-------------")
    # print(final_matches)

    recalculate_edit_distances(final_matches, gt_lens_dict, no_ignores_gt_lines, no_ignores_pred_lines)
    # print("-------------recalculate_edit_distances-------------")
    # print(final_matches)
    
    converted_results = convert_final_matches(final_matches, no_ignores_gt_lines, no_ignores_pred_lines)
    # print("-------------converted_results-------------")
    # print(converted_results)
    
    merged_results = merge_duplicates_add_unmatched(converted_results, no_ignores_gt_lines, no_ignores_pred_lines, no_ignores_ori_gt_lines, no_ignores_ori_pred_lines, no_ignores_gt_indices, no_ignores_pred_indices)

    for entry in merged_results:
        if entry['gt_idx'] != [""]:
            ignore_type = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
            gt_cagegory_clean = [no_ignores_gt_cat_list[_] for _ in entry['gt_idx'] if no_ignores_gt_cat_list[_] not in ignore_type] 
            if gt_cagegory_clean:
                entry['gt_category_type'] = Counter(gt_cagegory_clean).most_common(1)[0][0] 
            else:
                entry['gt_category_type'] = Counter([no_ignores_gt_cat_list[_] for _ in entry['gt_idx']]).most_common(1)[0][0] 
        else:
            entry['gt_category_type'] = ""
            entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], no_ignores_pred_items) if entry['pred_idx'] != [""] else ""
            if entry['pred_category_type'] == 'equation_inline':
                merged_results.remove(entry)


        entry['gt_idx'] = [entry['gt_idx']] if not isinstance(entry['gt_idx'], list) else entry['gt_idx']
        entry['pred_idx'] = [entry['pred_idx']] if not isinstance(entry['pred_idx'], list) else entry['pred_idx']
        entry['gt_position'] = [no_ignores_gt_items[_].get('order') if no_ignores_gt_items[_].get('order') else no_ignores_gt_items[_].get('position', [""])[0] for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [""]
        entry['pred_position'] = no_ignores_pred_items[entry['pred_idx'][0]]['position'][0] if entry['pred_idx'] != [""] else "" 
        #  0507 多行公式拼接修改
        if entry['gt_category_type'] == 'equation_isolated' and len(entry['gt_idx']) > 1:
            mutli_formula  = ' \\\\ '.join(['{'+strip_formula_delimiters(no_ignores_ori_gt_lines[_])+'}' for _ in  entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
            mutli_formula = '\\begin{array}{l} ' + mutli_formula + ' \end{array}'
            entry['gt'] = mutli_formula
        else:
            entry['gt'] = ''.join([no_ignores_ori_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""

        entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], no_ignores_pred_items) if entry['pred_idx'] != [""] else ""
        entry['gt_attribute'] = [no_ignores_gt_items[_].get("attribute", {}) for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [{}] 
        entry['img_id'] = img_name
        
        #  0724 多行公式拼接修改pred
        if 'equation' in entry['pred_category_type'] and len(entry['pred_idx']) > 1:
            mutli_formula  = ' \\\\ '.join(['{'+strip_formula_delimiters(no_ignores_ori_pred_lines[_])+'}' for _ in  entry['pred_idx']]) if entry['pred_idx'] != [""] else ""
            mutli_formula = '\\begin{array}{l} ' + mutli_formula + ' \end{array}'
            entry['pred'] = mutli_formula
        else:
            entry['pred'] = ''.join([no_ignores_ori_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""

        entry['norm_gt'] = ''.join([no_ignores_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
        entry['norm_pred'] = ''.join([no_ignores_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""

    
    # print("-------------merged_results-------------")
    # for i in merged_results:
    #     print(i)
    for entry in merged_results:
        if isinstance(entry['gt_idx'], list) and entry['gt_idx'] != [""]:
            gt_idx = []
            for i in entry['gt_idx']:
                gt_idx.append(no_ignores_gt_idxs[i])
            entry['gt_idx'] = gt_idx
        if isinstance(entry['pred_idx'], list) and entry['pred_idx'] != [""]:
            pred_idx = []
            for i in entry['pred_idx']:
                pred_idx.append(int(no_ignore_pred_idxs[i]))
            entry['pred_idx'] = pred_idx

    if len(merged_ignore_results) > 0:
        merged_results.extend(merged_ignore_results)
        # for i in merged_ignore_results:
        #     merged_results.append(i)

    return merged_results

    # cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, norm_pred_lines)
    
    # matched_col_idx, row_ind, cost_list = cal_final_match(cost_matrix, norm_gt_lines, norm_pred_lines)
    
    # gt_lens_dict, pred_lens_dict = initialize_indices(norm_gt_lines, norm_pred_lines)
    
    # matches, unmatched_gt_indices, unmatched_pred_indices = process_matches(matched_col_idx, row_ind, cost_list, norm_gt_lines, norm_pred_lines, pred_lines)
    
    # matching_dict = fuzzy_match_unmatched_items(unmatched_gt_indices, norm_gt_lines, norm_pred_lines)
    
    # final_matches = merge_matches(matches, matching_dict)
    
    # recalculate_edit_distances(final_matches, gt_lens_dict, norm_gt_lines, norm_pred_lines)
    
    # converted_results = convert_final_matches(final_matches, norm_gt_lines, norm_pred_lines)
    
    # merged_results = merge_duplicates_add_unmatched(converted_results, norm_gt_lines, norm_pred_lines, gt_lines, pred_lines, all_gt_indices, all_pred_indices)

    # for entry in merged_results:
    #         entry['gt_idx'] = [entry['gt_idx']] if not isinstance(entry['gt_idx'], list) else entry['gt_idx']
    #         entry['pred_idx'] = [entry['pred_idx']] if not isinstance(entry['pred_idx'], list) else entry['pred_idx']
    #         entry['gt_position'] = [gt_items[_].get('order') if gt_items[_].get('order') else gt_items[_].get('position', [""])[0] for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [""]
    #         entry['pred_position'] = pred_items[entry['pred_idx'][0]]['position'][0] if entry['pred_idx'] != [""] else "" 
    #         entry['gt'] = ''.join([gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
    #         entry['pred'] = ''.join([pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""
    #         entry['norm_gt'] = ''.join([norm_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [""] else ""
    #         entry['norm_pred'] = ''.join([norm_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [""] else ""

    #         if entry['gt_idx'] != [""]:
    #             ignore_type = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
    #             gt_cagegory_clean = [gt_cat_list[_] for _ in entry['gt_idx'] if gt_cat_list[_] not in ignore_type] 
    #             if gt_cagegory_clean:
    #                 entry['gt_category_type'] = Counter(gt_cagegory_clean).most_common(1)[0][0] 
    #             else:
    #                 entry['gt_category_type'] = Counter([gt_cat_list[_] for _ in entry['gt_idx']]).most_common(1)[0][0] 
    #         else:
    #             entry['gt_category_type'] = ""
    #         entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], pred_items) if entry['pred_idx'] != [""] else ""
    #         entry['gt_attribute'] = [gt_items[_].get("attribute", {}) for _ in entry['gt_idx']] if entry['gt_idx'] != [""] else [{}] 
    #         entry['img_id'] = img_name
        
    # return merged_results


def merge_duplicates_add_unmatched(converted_results, norm_gt_lines, norm_pred_lines, gt_lines, pred_lines, all_gt_indices, all_pred_indices):
    merged_results = []
    processed_pred = set()
    processed_gt = set()

    for entry in converted_results:
        pred_idx = tuple(entry['pred_idx']) if isinstance(entry['pred_idx'], list) else (entry['pred_idx'],)
        if pred_idx not in processed_pred and pred_idx != ("",):
            merged_entry = {
                'gt_idx': [entry['gt_idx']],
                'gt': entry['gt'],
                'pred_idx': entry['pred_idx'],
                'pred': entry['pred'],
                'edit': entry['edit']
            }
            for other_entry in converted_results:
                other_pred_idx = tuple(other_entry['pred_idx']) if isinstance(other_entry['pred_idx'], list) else (other_entry['pred_idx'],)
                if other_pred_idx == pred_idx and other_entry is not entry:
                    merged_entry['gt_idx'].append(other_entry['gt_idx'])
                    merged_entry['gt'] += other_entry['gt']
                    processed_gt.add(other_entry['gt_idx'])
            merged_results.append(merged_entry)
            processed_pred.add(pred_idx)
            processed_gt.add(entry['gt_idx'])

    # for entry in converted_results:
    #     if entry['gt_idx'] not in processed_gt:
    #         merged_results.append(entry)

    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx not in processed_gt:
            merged_results.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'edit': 1
            })
    return merged_results




def formula_format(formula_matches, img_name):
    return [
        {
            "gt": item["gt"],
            "pred": item["pred"],
            "img_id": f"{img_name}_{i}"
        }
        for i, item in enumerate(formula_matches)
    ]


def merge_lists_with_sublists(main_list, sub_lists):
    main_list_final = list(copy.deepcopy(main_list))
    for sub_list in sub_lists:
        pop_idx = main_list_final.index(sub_list[0])
        for _ in sub_list: 
            main_list_final.pop(pop_idx)
        main_list_final.insert(pop_idx, sub_list) 
    return main_list_final   


def sub_pred_fuzzy_matching(gt, pred):
    
    min_d = float('inf')
    # pos = -1

    gt_len = len(gt)
    pred_len = len(pred)

    if gt_len >= pred_len and pred_len > 0:
        for i in range(gt_len - pred_len + 1):
            sub = gt[i:i + pred_len]
            dist = Levenshtein_distance(sub, pred)/pred_len
            if dist < min_d:
                min_d = dist
                pos = i

        return min_d
    else:
        return False
        
def sub_gt_fuzzy_matching(pred, gt):  
    
    min_d = float('inf')  
    pos = ""  
    matched_sub = ""  
    gt_len = len(gt)  
    pred_len = len(pred)  
    
    if pred_len >= gt_len and gt_len > 0:  
        for i in range(pred_len - gt_len + 1):  
            sub = pred[i:i + gt_len]  
            dist = Levenshtein.distance(sub, gt)  /gt_len
            if dist < min_d:  
                min_d = dist  
                pos = i  
                matched_sub = sub  
        return min_d, pos, gt_len, matched_sub  
    else:  
        return 1, "", gt_len, "" 
        
        
def get_final_subset(subset_certain, subset_certain_cost):
    if not subset_certain or not subset_certain_cost:
        return []  

    subset_turple = sorted([(a, b) for a, b in zip(subset_certain, subset_certain_cost)], key=lambda x: x[0][0])

    group_list = defaultdict(list)
    group_idx = 0
    group_list[group_idx].append(subset_turple[0])

    for item in subset_turple[1:]:
        overlap_flag = False
        for subset in group_list[group_idx]:
            for idx in item[0]:
                if idx in subset[0]:
                    overlap_flag = True
                    break
            if overlap_flag:
                break
        if overlap_flag:
            group_list[group_idx].append(item)
        else:
            group_idx += 1
            group_list[group_idx].append(item)

    final_subset = []
    for _, group in group_list.items():
        if len(group) == 1: 
            final_subset.append(group[0][0])
        else:
            path_dict = defaultdict(list)
            path_idx = 0
            path_dict[path_idx].append(group[0])
            
            for subset in group[1:]:
                new_path = True
                for path_idx_s, path_items in path_dict.items():
                    is_dup = False
                    is_same = False
                    for path_item in path_items:
                        if path_item[0] == subset[0]:
                            is_dup = True
                            is_same = True
                            if path_item[1] > subset[1]:
                                path_dict[path_idx_s].pop(path_dict[path_idx_s].index(path_item))
                                path_dict[path_idx_s].append(subset)
                        else:
                            for num_1 in path_item[0]:
                                for num_2 in subset[0]:
                                    if num_1 == num_2:
                                        is_dup = True
                    if not is_dup:
                        path_dict[path_idx_s].append(subset)
                        new_path = False
                    if is_same:
                        new_path = False
                if new_path:
                    path_idx = len(path_dict.keys())
                    path_dict[path_idx].append(subset)

            saved_cost = float('inf')
            saved_subset = []  
            for path_idx, path in path_dict.items():
                avg_cost = sum([i[1] for i in path]) / len(path)
                if avg_cost < saved_cost:
                    saved_subset = [i[0] for i in path]
                    saved_cost = avg_cost

            final_subset.extend(saved_subset)

    return final_subset

def judge_pred_merge(gt_list, pred_list, threshold=0.6):
    if len(pred_list) == 1:
        return False, False

    cur_pred = ' '.join(pred_list[:-1])
    merged_pred = ' '.join(pred_list)
    
    cur_dist = Levenshtein.distance(gt_list[0], cur_pred) / max(len(gt_list[0]), len(cur_pred))
    merged_dist = Levenshtein.distance(gt_list[0], merged_pred) / max(len(gt_list[0]), len(merged_pred))
    
    if merged_dist > cur_dist:
        return False, False

    cur_fuzzy_dists = [sub_pred_fuzzy_matching(gt_list[0], cur_pred) for cur_pred in pred_list[:-1]]
    if any(dist is False or dist > threshold for dist in cur_fuzzy_dists):
        return False, False

    add_fuzzy_dist = sub_pred_fuzzy_matching(gt_list[0], pred_list[-1])
    if add_fuzzy_dist is False:
        return False, False

    merged_pred_flag = add_fuzzy_dist < threshold
    continue_flag = len(merged_pred) <= len(gt_list[0])

    return merged_pred_flag, continue_flag
    
def deal_with_truncated(cost_matrix, norm_gt_lines, norm_pred_lines, deadline=None):
    matched_first = np.argwhere(cost_matrix < 0.25)
    masked_gt_idx = [i[0] for i in matched_first]
    unmasked_gt_idx = [i for i in range(cost_matrix.shape[0]) if i not in masked_gt_idx]
    masked_pred_idx = [i[1] for i in matched_first]
    unmasked_pred_idx = [i for i in range(cost_matrix.shape[1]) if i not in masked_pred_idx]

    merges_gt_dict = {}
    merges_pred_dict = {}
    merged_gt_subsets = []

    for gt_idx in unmasked_gt_idx:
        if deadline is not None and time.monotonic() >= deadline:
            raise TruncatedMatchTimeout()
        check_merge_subset = []
        merged_dist = []

        for pred_idx in unmasked_pred_idx: 
            if deadline is not None and time.monotonic() >= deadline:
                raise TruncatedMatchTimeout()
            step = 1
            merged_pred = [norm_pred_lines[pred_idx]]

            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TruncatedMatchTimeout()
                if step >= MAX_TRUNCATED_PRED_MERGE:
                    break
                if pred_idx + step in masked_pred_idx or pred_idx + step >= len(norm_pred_lines):
                    break
                else:
                    merged_pred.append(norm_pred_lines[pred_idx + step])
                    merged_pred_flag, continue_flag = judge_pred_merge([norm_gt_lines[gt_idx]], merged_pred) 
                    if not merged_pred_flag:
                        break
                    else:
                        step += 1
                    if not continue_flag:
                        break

            check_merge_subset.append(list(range(pred_idx, pred_idx + step)))
            matched_line = ' '.join([norm_pred_lines[i] for i in range(pred_idx, pred_idx + step)])
            dist = Levenshtein_distance(norm_gt_lines[gt_idx], matched_line) / max(len(matched_line), len(norm_gt_lines[gt_idx]))
            merged_dist.append(dist)

        if not merged_dist:
            subset_certain = []
            min_cost_idx = ""
            min_cost = float('inf')
        else:
            min_cost = min(merged_dist)
            min_cost_idx = merged_dist.index(min_cost)
            subset_certain = check_merge_subset[min_cost_idx]

        merges_gt_dict[gt_idx] = {
            'merge_subset': check_merge_subset,
            'merged_cost': merged_dist,
            'min_cost_idx': min_cost_idx,
            'subset_certain': subset_certain,
            'min_cost': min_cost
        }

    subset_certain = [merges_gt_dict[gt_idx]['subset_certain'] for gt_idx in unmasked_gt_idx if merges_gt_dict[gt_idx]['subset_certain']]
    subset_certain_cost = [merges_gt_dict[gt_idx]['min_cost'] for gt_idx in unmasked_gt_idx if merges_gt_dict[gt_idx]['subset_certain']]

    subset_certain_final = get_final_subset(subset_certain, subset_certain_cost)

    if not subset_certain_final:  
        return cost_matrix, norm_pred_lines, range(len(norm_pred_lines))

    final_pred_idx_list = merge_lists_with_sublists(range(len(norm_pred_lines)), subset_certain_final)
    final_norm_pred_lines = [' '.join(norm_pred_lines[idx_list[0]:idx_list[-1]+1]) if isinstance(idx_list, list) else norm_pred_lines[idx_list] for idx_list in final_pred_idx_list]

    new_cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, final_norm_pred_lines)

    return new_cost_matrix, final_norm_pred_lines, final_pred_idx_list
    
def cal_move_dist(gt, pred):
    assert len(gt) == len(pred), 'Not right length'
    step = 0
    for i, gt_c in enumerate(gt):
        if gt_c != pred[i]:
            step += abs(i - pred.index(gt_c))
            pred[i], pred[pred.index(gt_c)] = pred[pred.index(gt_c)], pred[i]
    return step / len(gt)

def cal_final_match(cost_matrix, norm_gt_lines, norm_pred_lines, truncated_timeout_sec=None):
    # min_indice = cost_matrix.argmax(axis=1)

    deadline = None
    if truncated_timeout_sec is not None:
        try:
            truncated_timeout_sec = float(truncated_timeout_sec)
        except (TypeError, ValueError):
            truncated_timeout_sec = None
        if truncated_timeout_sec is not None and truncated_timeout_sec > 0:
            deadline = time.monotonic() + truncated_timeout_sec

    new_cost_matrix, final_norm_pred_lines, final_pred_idx_list = deal_with_truncated(
        cost_matrix,
        norm_gt_lines,
        norm_pred_lines,
        deadline=deadline,
    )

    row_ind, col_ind = linear_sum_assignment(new_cost_matrix)

    cost_list = [new_cost_matrix[r][c] for r, c in zip(row_ind, col_ind)]
    matched_col_idx = [final_pred_idx_list[i] for i in col_ind]

    return matched_col_idx, row_ind, cost_list

def initialize_indices(norm_gt_lines, norm_pred_lines):
    gt_lens_dict = {idx: len(gt_line) for idx, gt_line in enumerate(norm_gt_lines)}
    pred_lens_dict = {idx: len(pred_line) for idx, pred_line in enumerate(norm_pred_lines)}
    return gt_lens_dict, pred_lens_dict

def process_matches(matched_col_idx, row_ind, cost_list, norm_gt_lines, norm_pred_lines, pred_lines):
    matches = {}
    unmatched_gt_indices = []
    unmatched_pred_indices = []

    for i in range(len(norm_gt_lines)):
        if i in row_ind:
            idx = list(row_ind).index(i)
            pred_idx = matched_col_idx[idx]

            if pred_idx is None or (isinstance(pred_idx, list) and None in pred_idx):
                unmatched_pred_indices.append(pred_idx)
                continue

            if isinstance(pred_idx, list):
                pred_line = ' | '.join(norm_pred_lines[pred_idx[0]:pred_idx[-1]+1])
                ori_pred_line = ' | '.join(pred_lines[pred_idx[0]:pred_idx[-1]+1])
                matched_pred_indices_range = list(range(pred_idx[0], pred_idx[-1]+1))
            else:
                pred_line = norm_pred_lines[pred_idx]
                ori_pred_line = pred_lines[pred_idx]
                matched_pred_indices_range = [pred_idx]

            edit = cost_list[idx]

            if edit > 0.7:
                unmatched_pred_indices.extend(matched_pred_indices_range)
                unmatched_gt_indices.append(i)
            else:
                matches[i] = {
                    'pred_indices': matched_pred_indices_range,
                    'edit_distance': edit,
                }
                for matched_pred_idx in matched_pred_indices_range:
                    if matched_pred_idx in unmatched_pred_indices:
                        unmatched_pred_indices.remove(matched_pred_idx)
        else:
            unmatched_gt_indices.append(i)

    return matches, unmatched_gt_indices, unmatched_pred_indices

def fuzzy_match_unmatched_items(unmatched_gt_indices, norm_gt_lines, norm_pred_lines, gt_cat_list=None):
    matching_dict = {}

    for pred_idx, pred_content in enumerate(norm_pred_lines):
        if isinstance(pred_idx, list):
            continue

        matching_indices = []

        for unmatched_gt_idx in unmatched_gt_indices:
            if gt_cat_list and gt_cat_list[unmatched_gt_idx] == 'equation_isolated':
                continue
            gt_content = norm_gt_lines[unmatched_gt_idx]
            cur_fuzzy_dist_unmatch, cur_pos, gt_lens, matched_field = sub_gt_fuzzy_matching(pred_content, gt_content)
            if cur_fuzzy_dist_unmatch < 0.4:
                matching_indices.append(unmatched_gt_idx)

        if matching_indices:
            matching_dict[pred_idx] = matching_indices

    return matching_dict

def merge_matches(matches, matching_dict):
    final_matches = {}
    processed_gt_indices = set() 

    for gt_idx, match_info in matches.items():
        pred_indices = match_info['pred_indices']
        edit_distance = match_info['edit_distance']

        pred_key = tuple(sorted(pred_indices))

        if pred_key in final_matches:
            if gt_idx not in processed_gt_indices:
                final_matches[pred_key]['gt_indices'].append(gt_idx)
                processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                'gt_indices': [gt_idx],
                'edit_distance': edit_distance
            }
            processed_gt_indices.add(gt_idx)

    for pred_idx, gt_indices in matching_dict.items():
        pred_key = (pred_idx,) if not isinstance(pred_idx, (list, tuple)) else tuple(sorted(pred_idx))

        if pred_key in final_matches:
            for gt_idx in gt_indices:
                if gt_idx not in processed_gt_indices:
                    final_matches[pred_key]['gt_indices'].append(gt_idx)
                    processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                'gt_indices': [gt_idx for gt_idx in gt_indices if gt_idx not in processed_gt_indices],
                'edit_distance': None
            }
            processed_gt_indices.update(final_matches[pred_key]['gt_indices'])

    return final_matches
    


def recalculate_edit_distances(final_matches, gt_lens_dict, norm_gt_lines, norm_pred_lines):
    for pred_key, info in final_matches.items():
        gt_indices = sorted(set(info['gt_indices']))

        if not gt_indices:
            info['edit_distance'] = 1
            continue

        if len(gt_indices) > 1:
            merged_gt_content = ''.join(norm_gt_lines[gt_idx] for gt_idx in gt_indices)
            pred_content = norm_pred_lines[pred_key[0]] if isinstance(pred_key[0], int) else ''

            try:
                edit_distance = Levenshtein_distance(merged_gt_content, pred_content)
                normalized_edit_distance = edit_distance / max(len(merged_gt_content), len(pred_content))
            except ZeroDivisionError:
                normalized_edit_distance = 1

            info['edit_distance'] = normalized_edit_distance
        else:
            gt_idx = gt_indices[0]
            pred_content = ' '.join(norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int))

            try:
                edit_distance = Levenshtein_distance(norm_gt_lines[gt_idx], pred_content)
                normalized_edit_distance = edit_distance / max(len(norm_gt_lines[gt_idx]), len(pred_content))
            except ZeroDivisionError:
                normalized_edit_distance = 1

            info['edit_distance'] = normalized_edit_distance
            info['pred_content'] = pred_content


def convert_final_matches(final_matches, norm_gt_lines, norm_pred_lines):
    converted_results = []

    all_gt_indices = set(range(len(norm_gt_lines)))
    all_pred_indices = set(range(len(norm_pred_lines)))

    for pred_key, info in final_matches.items():
        pred_content = ' '.join(norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int))
        
        for gt_idx in sorted(set(info['gt_indices'])):
            result_entry = {
                'gt_idx': int(gt_idx),
                'gt': norm_gt_lines[gt_idx],
                'pred_idx': list(pred_key),
                'pred': pred_content,
                'edit': info['edit_distance']
            }
            converted_results.append(result_entry)
    
    matched_gt_indices = set().union(*[set(info['gt_indices']) for info in final_matches.values()])
    unmatched_gt_indices = all_gt_indices - matched_gt_indices
    matched_pred_indices = set(idx for pred_key in final_matches.keys() for idx in pred_key if isinstance(idx, int))
    unmatched_pred_indices = all_pred_indices - matched_pred_indices

    if unmatched_pred_indices:
        if unmatched_gt_indices:
            distance_matrix = [
                # [Levenshtein_distance(norm_gt_lines[gt_idx], norm_pred_lines[pred_idx]) for pred_idx in unmatched_pred_indices]
                [Levenshtein_distance(norm_gt_lines[gt_idx], norm_pred_lines[pred_idx])/max(len(norm_gt_lines[gt_idx]), len(norm_pred_lines[pred_idx])) for pred_idx in unmatched_pred_indices]
                for gt_idx in unmatched_gt_indices
            ]

            row_ind, col_ind = linear_sum_assignment(distance_matrix)

            for i, j in zip(row_ind, col_ind):
                gt_idx = list(unmatched_gt_indices)[i]
                pred_idx = list(unmatched_pred_indices)[j]
                result_entry = {
                    'gt_idx': int(gt_idx),
                    'gt': norm_gt_lines[gt_idx],
                    'pred_idx': [pred_idx],
                    'pred': norm_pred_lines[pred_idx],
                    'edit': 1
                }
                converted_results.append(result_entry)

            matched_gt_indices.update(list(unmatched_gt_indices)[i] for i in row_ind)
        else:
            result_entry = {
                'gt_idx': "",
                'gt': '',
                'pred_idx': list(unmatched_pred_indices),
                'pred': ' '.join(norm_pred_lines[pred_idx] for pred_idx in unmatched_pred_indices),
                'edit': 1
            }
            converted_results.append(result_entry)
    else:
        for gt_idx in unmatched_gt_indices:
            result_entry = {
                'gt_idx': int(gt_idx),
                'gt': norm_gt_lines[gt_idx],
                'pred_idx': "",
                'pred': '',
                'edit': 1
            }
            converted_results.append(result_entry)

    return converted_results
