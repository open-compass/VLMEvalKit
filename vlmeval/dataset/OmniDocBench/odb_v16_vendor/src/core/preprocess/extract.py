import re
import os
import json
import copy
#from  modules.table_utils import convert_markdown_to_html #end
from src.core.preprocess.table_utils import convert_markdown_to_html
import re
import unicodedata
from bs4 import BeautifulSoup
from pylatexenc.latexencode import unicode_to_latex
from src.core.preprocess.text_postprocess import safe_latex_to_text
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexCharsNode, LatexGroupNode, LatexMacroNode, LatexSpecialsNode
from collections import defaultdict
from src.core.preprocess.data_preprocess import (
    normalized_latex_table,
    remove_markdown_fences,
    replace_repeated_chars,
    textblock2unicode,
    textblock_with_norm_formula,
)


def extract_tabular(text):
    begin_pattern = r'\\begin{tabular}'
    end_pattern = r'\\end{tabular}'

    tabulars = []
    positions = []
    current_pos = 0
    stack = []
    
    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])
        
        if not begin_match and not end_match:
            break
            
        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1
    
    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_tabular(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions

# math reg
    # r'\\begin{equation\*?}(.*?)\\end{equation\*?}|'
    # r'\\begin{align\*?}(.*?)\\end{align\*?}|'
    # r'\\begin{gather\*?}(.*?)\\end{gather\*?}|'
display_reg = re.compile(
    # r'\\begin{equation\*?}(.*?)\\end{equation\*?}|'
    # r'\\begin{align\*?}(.*?)\\end{align\*?}|'
    # r'\\begin{gather\*?}(.*?)\\end{gather\*?}|'
    # r'\\begin{array\*?}(.*?)\\end{array\*?}|'
    r'\$\$(.*?)\$\$|'
    r'\\\[(.*?)\\\]|'
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',  
    re.DOTALL
)

# inline_reg = re.compile(
#     r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)|'
#     r'\\\((.*?)\\\)',
# )
inline_reg = re.compile(
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',
)

# table 
table_reg = re.compile(
    r'\\begin{table\*?}(.*?)\\end{table\*?}|'
    r'\\begin{tabular\*?}(.*?)\\end{tabular\*?}',
    re.DOTALL 
)
md_table_reg = re.compile(
    r'\|\s*.*?\s*\|\n', 
    re.DOTALL)
html_table_reg = re.compile(
    r'(<table.*?</table>)',
    re.DOTALL
)

# title
title_reg = re.compile(
    r'^\s*#.*$', 
    re.MULTILINE)

# img
img_pattern = r'!\[.*?\]\(.*?\)'
image_reg = re.compile(r'!\[(.*?)\]\((.*?)\)')

# code block
code_block_reg = re.compile(
    r'```(\w+)\n(.*?)```',
    re.DOTALL
)

formula_token_reg = re.compile(
    r'(\\frac|\\sqrt|\\sum|\\int|\\prod|\\times|\\div|\\square|\\cdots|\\left|\\right|'
    r'\\approx|\\xrightarrow|\\rightarrow|\\longrightarrow|'
    r'=|<|>|≤|≥|≈|÷|×|±|∑|∫|√|→|\^|_|□|△|…|%|/|\+|-)'
)
cjk_reg = re.compile(r'[\u4e00-\u9fff]')
alpha_word_reg = re.compile(r'[A-Za-z]{4,}')

def _strip_formula_wrappers_keep_content(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f" {m.group(1)} ", text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: f" {m.group(1)} ", text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: f" {m.group(1)} ", text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', lambda m: f" {m.group(1)} ", text, flags=re.DOTALL)
    return text


def _formula_like_text(text):
    text = ' '.join(str(text or '').split())
    if not text or len(text) > 220:
        return False

    bare_text = _strip_formula_wrappers_keep_content(text)
    compact = re.sub(r'\s+', '', bare_text)
    if not compact:
        return False

    operator_hits = len(formula_token_reg.findall(bare_text))
    digit_hits = len(re.findall(r'\d', bare_text))
    inline_hits = len(list(inline_reg.finditer(text))) + len(list(re.finditer(r'\$\$(.*?)\$\$|\\\[(.*?)\\\]', text, re.DOTALL)))
    cjk_hits = len(cjk_reg.findall(bare_text))
    alpha_hits = len(alpha_word_reg.findall(bare_text))

    residual = _strip_formula_wrappers_keep_content(text)
    residual = re.sub(r'(\\[A-Za-z]+|[=<>≤≥÷×±∑∫√\^_□△…%/()+\-])', ' ', residual)
    residual = re.sub(r'\d', ' ', residual)
    residual = re.sub(r'\s+', '', residual)

    if inline_hits > 0 and len(residual) <= 18 and operator_hits >= 1:
        return True
    if ('□' in bare_text or '\\square' in bare_text) and operator_hits >= 1 and alpha_hits <= 1 and len(compact) <= 80:
        return True
    if ('\\xrightarrow' in bare_text or '\\longrightarrow' in bare_text or '→' in bare_text) and cjk_hits <= 24 and len(compact) <= 120:
        return True
    if operator_hits >= 2 and digit_hits >= 1 and alpha_hits <= 2 and cjk_hits <= 12 and len(compact) <= 100:
        return True
    if operator_hits >= 3 and alpha_hits == 0 and cjk_hits <= 18 and len(compact) <= 80:
        return True
    return False


def _sanitize_formula_candidate(text):
    text = ' '.join(str(text or '').replace('\u3000', ' ').split())
    if not text:
        return ''

    text = _strip_formula_wrappers_keep_content(text)
    replacements = [
        ('……', ' \\cdots \\cdots '),
        ('…', ' \\cdots '),
        ('×', ' \\times '),
        ('÷', ' \\div '),
        ('≤', ' \\leq '),
        ('≥', ' \\geq '),
        ('≠', ' \\neq '),
        ('≈', ' \\approx '),
        ('□', ' \\square '),
        ('△', ' \\triangle '),
        ('（', '('),
        ('）', ')'),
        ('［', '['),
        ('］', ']'),
        ('｛', '{'),
        ('｝', '}'),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    arrow_match = re.search(r'(?P<lhs>.*?)\\xrightarrow\s*\{(?P<label>[^{}]+)\}(?P<rhs>.*)', text)
    if arrow_match:
        lhs = arrow_match.group('lhs').strip()
        arrow_label = arrow_match.group('label').strip()
        rhs = arrow_match.group('rhs').strip()
        if arrow_label and '\\' not in arrow_label:
            arrow_label = f'\\text{{{arrow_label}}}'
        if rhs.endswith('氧化还原反应'):
            rhs_main = rhs[:-len('氧化还原反应')].strip()
            text = f'{lhs} \\underset{{氧化还原反应}}{{\\xrightarrow{{{arrow_label}}}}} {rhs_main}'.strip()
        else:
            text = f'{lhs} \\xrightarrow{{{arrow_label}}} {rhs}'.strip()

    text = re.sub(r'(?<!\\)%', r'\\%', text)
    text = re.sub(r'(?<!\\)&', r'\\&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ''
    if text.startswith('\\begin{') or text.startswith('$$') or text.startswith('\\['):
        return text
    return '\\[ ' + text + ' \\]'


def _looks_like_table_formula_cell(text):
    compact_text = ' '.join(str(text or '').split())
    if not compact_text or len(compact_text) > 120:
        return False
    if not _formula_like_text(compact_text):
        return False

    digit_hits = len(re.findall(r'\d', compact_text))
    fraction_hits = '\\frac' in compact_text or bool(re.search(r'\d+\s*/\s*\d+', compact_text))
    operator_hits = sum(token in compact_text for token in ['=', '+', '-', '×', '÷', '/', '≈', '\\times', '\\div', '\\approx'])

    if operator_hits < 1:
        return False
    if digit_hits >= 2:
        return True
    return fraction_hits and operator_hits >= 1


def _build_html_table_formula_rescue_items(html_table, table_position):
    rescue_items = []
    try:
        soup = BeautifulSoup(str(html_table or ''), 'html.parser')
    except Exception:
        return rescue_items

    pseudo_cursor = max(int(table_position[0]), 0)
    for cell in soup.find_all(['td', 'th']):
        cell_text = ' '.join(cell.get_text(separator=' ').split())
        if not _looks_like_table_formula_cell(cell_text):
            continue

        formula_content = _sanitize_formula_candidate(cell_text)
        if not formula_content:
            continue

        cell_start = pseudo_cursor
        cell_end = cell_start + max(len(cell_text), 1)
        rescue_items.append({
            'category_type': 'equation_isolated',
            'position': [cell_start, cell_end],
            'content': formula_content,
            'fine_category_type': 'table_formula_rescue',
        })
        pseudo_cursor = cell_end + 1

    return rescue_items


def _append_formula_rescue_candidate(pred_all, text, position, fine_category_type):
    if not _formula_like_text(text):
        return
    formula_content = _sanitize_formula_candidate(text)
    if not formula_content:
        return
    category_type = 'formula_rescue'
    if '\\xrightarrow' in formula_content or '\\longrightarrow' in formula_content:
        category_type = 'equation_isolated'
    pred_all.append({
        'category_type': category_type,
        'position': position,
        'content': formula_content,
        'fine_category_type': fine_category_type,
    })


def _unwrap_formula_envelope(text):
    text = ' '.join(str(text or '').split())
    wrappers = (
        ('\\[', '\\]'),
        ('\\(', '\\)'),
        ('$$', '$$'),
        ('$', '$'),
    )
    for left, right in wrappers:
        if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
            return left, right, text[len(left):len(text) - len(right)].strip()
    return '', '', text


def _wrap_formula_envelope(inner_text, left_wrapper, right_wrapper):
    inner_text = str(inner_text or '').strip()
    if not inner_text:
        return ''
    if left_wrapper and right_wrapper:
        return f'{left_wrapper} {inner_text} {right_wrapper}'
    return inner_text


def _latex_fragment_to_plain_text(text):
    text = str(text or '').strip()
    if not text:
        return ''
    try:
        plain_text = safe_latex_to_text(text, fallback=text, latex_type='formula')
    except Exception:
        plain_text = text
    plain_text = plain_text.replace('\xa0', ' ')
    plain_text = re.sub(r'\s+', ' ', plain_text)
    return plain_text.strip()


def _looks_like_equation_caption(text):
    raw_text = str(text or '').strip()
    if not raw_text:
        return False
    plain_text = _latex_fragment_to_plain_text(raw_text)
    if not plain_text:
        return False

    letter_hits = len(re.findall(r'[A-Za-z]', plain_text))
    word_hits = re.findall(r'[A-Za-z]{2,}', plain_text)
    math_hits = len(re.findall(r'(\\[A-Za-z]+|[=<>≤≥÷×±∑∫√_^{}])', raw_text))
    digit_hits = len(re.findall(r'\d', plain_text))

    if plain_text.startswith('(') and plain_text.endswith(')'):
        plain_core = plain_text[1:-1].strip()
    else:
        plain_core = plain_text

    if re.fullmatch(r'[0-9IVXivx]+', plain_core.replace(' ', '')):
        return False
    if letter_hits >= 10 and len(word_hits) >= 2 and math_hits <= 6:
        return True
    if letter_hits >= 14 and digit_hits == 0 and math_hits <= 3:
        return True
    return False


def _make_equation_margin_text_item(text, position):
    plain_text = _latex_fragment_to_plain_text(text)
    plain_text = plain_text.strip() if plain_text else ''
    if not plain_text:
        return None
    return {
        'category_type': 'text_all',
        'position': position,
        'content': plain_text,
        'fine_category_type': 'equation_caption'
    }


def _split_display_formula_margin_text(formula_text, position):
    compact_text = ' '.join(str(formula_text or '').split())
    if not compact_text:
        return compact_text, []

    left_wrapper, right_wrapper, inner_text = _unwrap_formula_envelope(compact_text)
    if not inner_text:
        return compact_text, []

    extra_items = []
    working_text = inner_text

    leading_number_pattern = re.compile(
        r'^\s*(?P<prefix>(?:\\left\(\s*[0-9IVXivx]+\s*\\right\)|\(\s*[0-9IVXivx]+\s*\)))\s*(?:(?:\\qquad|\\quad|\\enspace|\\,|\\;|\\:)\s*)*'
    )
    leading_match = leading_number_pattern.match(working_text)
    if leading_match:
        prefix_raw = leading_match.group('prefix').strip()
        prefix_end = min(position[1], position[0] + max(len(prefix_raw), 1))
        prefix_item = _make_equation_margin_text_item(prefix_raw, [position[0], prefix_end])
        if prefix_item:
            extra_items.append(prefix_item)
        working_text = working_text[leading_match.end():].strip()

    trailing_patterns = [
        re.compile(r'(?P<sep>(?:\\qquad|\\quad|\\enspace|\\,|\\;|\\:)\s*)(?P<caption>\\text\s*\{[^{}]*\}\.?)\s*$'),
        re.compile(r'(?P<sep>(?:\\qquad|\\quad|\\enspace|\\,|\\;|\\:)\s*)(?P<caption>\([^)]*[A-Za-z][^)]*\)\.?)\s*$'),
    ]
    while working_text:
        matched = False
        for trailing_pattern in trailing_patterns:
            trailing_match = trailing_pattern.search(working_text)
            if not trailing_match:
                continue
            caption_raw = trailing_match.group('caption').strip()
            if not _looks_like_equation_caption(caption_raw):
                continue
            caption_start = max(position[0], position[1] - max(len(caption_raw), 1))
            caption_item = _make_equation_margin_text_item(caption_raw, [caption_start, position[1]])
            if caption_item:
                extra_items.append(caption_item)
            working_text = working_text[:trailing_match.start('sep')].rstrip()
            matched = True
            break
        if not matched:
            break

    working_text = re.sub(r'(?:(?:\\qquad|\\quad|\\enspace|\\,|\\;|\\:)\s*)+$', '', working_text).strip()
    if not working_text:
        return compact_text, []
    return _wrap_formula_envelope(working_text, left_wrapper, right_wrapper), extra_items


def _should_split_display_formula_margin_text():
    flag = str(
        os.environ.get(
            'OMNIDOCBENCH_FORMULA_MARGIN_SPLIT',
            os.environ.get('FORMULA_MARGIN_SPLIT', '0'),
        )
    ).strip().lower()
    return flag not in {'0', 'false', 'off', 'no'}


def _has_display_formula_margin_text(single_line):
    compact_text = ' '.join(str(single_line or '').split())
    if not compact_text:
        return False

    _, _, inner_text = _unwrap_formula_envelope(compact_text)
    if not inner_text:
        return False

    leading_number_pattern = re.compile(
        r'^\s*(?:\\left\(\s*[0-9IVXivx]+\s*\\right\)|\(\s*[0-9IVXivx]+\s*\))\s*(?:(?:\\qquad|\\quad|\\enspace|\\,|\\;|\\:)\s*)*'
    )
    if leading_number_pattern.match(inner_text):
        return True

    trailing_patterns = [
        re.compile(r'(?P<sep>(?:\\qquad|\\quad|\\enspace|\\,|\\;|\\:)\s*)(?P<caption>\\text\s*\{[^{}]*\}\.?)\s*$'),
        re.compile(r'(?P<sep>(?:\\qquad|\\quad|\\enspace|\\,|\\;|\\:)\s*)(?P<caption>\([^)]*[A-Za-z][^)]*\)\.?)\s*$'),
    ]
    for trailing_pattern in trailing_patterns:
        trailing_match = trailing_pattern.search(inner_text)
        if trailing_match and _looks_like_equation_caption(trailing_match.group('caption')):
            return True
    return False


def _looks_like_compound_arithmetic_formula(segment_text):
    compact_text = ' '.join(str(segment_text or '').split())
    if not compact_text or len(compact_text) > 160:
        return False
    if '\\begin{' in compact_text or '\\end{' in compact_text:
        return False
    if re.search(r'\\(mathbf|left|right|cdot|lambda|Gamma|varGamma|text|matrix|pmatrix|bmatrix)', compact_text):
        return False

    digit_hits = len(re.findall(r'\d', compact_text))
    fraction_hits = '\\frac' in compact_text or bool(re.search(r'\d+\s*/\s*\d+', compact_text))
    relation_hits = sum(token in compact_text for token in ['=', '\\approx', '≈', '\\neq', '≠', '\\leq', '≤', '\\geq', '≥'])

    if relation_hits < 1:
        return False
    if digit_hits >= 2:
        return True
    return fraction_hits and relation_hits >= 1


def _split_compound_display_formula(single_line, position):
    compact_text = ' '.join(str(single_line or '').split())
    if not compact_text:
        return []

    left_wrapper, right_wrapper, inner_text = _unwrap_formula_envelope(compact_text)
    if not inner_text or ('\\quad' not in inner_text and '\\qquad' not in inner_text):
        return []

    segments = [segment.strip() for segment in re.split(r'\s*(?:\\qquad|\\quad)\s*', inner_text) if segment.strip()]
    if len(segments) <= 1:
        return []
    if not all(_looks_like_compound_arithmetic_formula(segment) for segment in segments):
        return []

    items = []
    total_span = max(position[1] - position[0], len(segments))
    per_span = max(total_span // len(segments), 1)
    for idx, segment in enumerate(segments):
        segment_start = position[0] + idx * per_span
        segment_end = position[1] if idx == len(segments) - 1 else min(position[1], segment_start + per_span)
        items.append({
            'category_type': 'equation_isolated',
            'position': [segment_start, segment_end],
            'content': _wrap_formula_envelope(segment, left_wrapper, right_wrapper),
        })
    return items


def _append_display_formula_item(pred_all, single_line, position, fine_category_type=None):
    if _should_split_display_formula_margin_text() or _has_display_formula_margin_text(single_line):
        normalized_formula, extra_items = _split_display_formula_margin_text(single_line, position)
    else:
        normalized_formula, extra_items = single_line, []

    split_items = _split_compound_display_formula(normalized_formula, position)
    if split_items:
        for split_item in split_items:
            if fine_category_type:
                split_item['fine_category_type'] = fine_category_type
            pred_all.append(split_item)
    else:
        formula_item = {
            'category_type': 'equation_isolated',
            'position': position,
            'content': normalized_formula,
        }
        if fine_category_type:
            formula_item['fine_category_type'] = fine_category_type
        pred_all.append(formula_item)
    pred_all.extend(extra_items)


def _normalize_html_text_fragment(text):
    if not isinstance(text, str):
        return text

    stripped_text = text.strip()
    if not stripped_text:
        return stripped_text

    lowered_text = stripped_text.lower()
    if '<table' in lowered_text:
        return stripped_text
    if not re.match(r'^<(div|dir)\b', stripped_text, flags=re.IGNORECASE):
        return stripped_text

    soup = BeautifulSoup(stripped_text, 'html.parser')
    for removable_tag in soup.find_all(['style', 'script']):
        removable_tag.decompose()
    for inline_tag in soup.find_all(['span', 'font', 'b', 'strong', 'i', 'em', 'u', 'a', 'sup', 'sub']):
        inline_tag.unwrap()
    for br_tag in soup.find_all('br'):
        br_tag.replace_with('\n')
    for block_tag in soup.find_all(['div', 'dir', 'p', 'li']):
        if not block_tag.contents or block_tag.contents[-1] != '\n':
            block_tag.append('\n')

    extracted_text = soup.get_text(separator='')
    extracted_text = extracted_text.replace('\xa0', ' ')
    extracted_text = re.sub(r'[ 	]*\n[ 	]*', '\n', extracted_text)
    extracted_lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
    normalized_text = '\n'.join(extracted_lines).strip()
    return normalized_text or stripped_text


def _looks_like_image_description_block(text):
    compact_text = ' '.join(str(text or '').split()).strip().lower()
    if len(compact_text) < 24:
        return False
    prefixes = (
        'the image ',
        'the image displays',
        'the figure ',
        'the figure shows',
        'the diagram ',
        'diagram:',
        '[diagram:',
        '[image:',
        'qr code ',
    )
    return any(compact_text.startswith(prefix) for prefix in prefixes)


def _strip_inline_formula_delimiters(text):
    text = re.sub(r'\\\((.*?)\\\)', lambda m: m.group(1), str(text or ''), flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', lambda m: m.group(1), text, flags=re.DOTALL)
    return text


def _suppress_formula_delimiters_in_image_descriptions(content):
    blocks = re.split(r'(\n\s*\n)', str(content or ''))
    for idx in range(0, len(blocks), 2):
        block = blocks[idx]
        if not _looks_like_image_description_block(block):
            continue
        blocks[idx] = _strip_inline_formula_delimiters(block)
    return ''.join(blocks)


def _should_drop_short_inline_formula_item(item):
    if item.get('category_type') != 'equation_isolated':
        return False
    if item.get('fine_category_type') != 'equation_inline':
        return False

    raw_formula = str(item.get('content', '') or '').strip()
    if not raw_formula:
        return False

    core_text = _strip_formula_wrappers_keep_content(raw_formula)
    compact_text = re.sub(r'\s+', '', core_text)
    if compact_text.startswith('\\xrightarrow') or compact_text.startswith('\\longrightarrow'):
        return True
    if not compact_text or len(compact_text) > 24:
        return False

    strong_math_tokens = ['=', '<', '>', '≤', '≥', '≈', '≠', '^', '+', '-', '/', '\\cdot', '\\frac', '\\sqrt', '\\sum', '\\int', '\\times', '\\div', '\\approx', '\\neq', '\\leq', '\\geq', '\\xrightarrow', '\\longrightarrow']
    if any(token in core_text for token in strong_math_tokens):
        return False

    plain_text = _latex_fragment_to_plain_text(raw_formula)
    plain_compact = re.sub(r'\s+', '', plain_text)
    label_text = compact_text if compact_text else plain_compact
    if not label_text:
        return False

    if re.fullmatch(r'[A-Za-z0-9._,-]+', label_text):
        return True
    if re.fullmatch(r'[A-Za-z][A-Za-z0-9_.,-]*', label_text):
        return True
    if re.fullmatch(r'[A-Za-z0-9]+Oy', label_text):
        return True
    if re.fullmatch(r'[ΓγλμνxyzuvwMNOabcABC0-9._,-]+', label_text):
        return True
    return False


def md_tex_filter(content):
    '''
    Input: 1 page md or tex content - String
    Output: text, display, inline, table, title, code - list
    '''
    image_formula_rescue = []
    for match in image_reg.finditer(content):
        alt_text = (match.group(1) or '').strip()
        _append_formula_rescue_candidate(image_formula_rescue, alt_text, [match.start(), match.end()], 'image_formula')
    content = re.sub(img_pattern, lambda m: ' ' * len(m.group(0)), content)  # remove image but keep positions
    content = remove_markdown_fences(content)   # remove markdown fences
    content = replace_repeated_chars(content) # replace all consecutive characters
    content = content.replace('<html>', '').replace('</html>', '').replace('<body>', '').replace('</body>', '')
    content = _suppress_formula_delimiters_in_image_descriptions(content)
    
    # # 使用正则表达式对unicode进行替换
    # special_unicode = ''.join(unicode_replacements.keys())
    # content = re.sub(f'[{special_unicode}]', replace_unicode, content)

    # content = fullwidth_to_halfwidth(content)  # fullwidth to halfwidth, TODO: GT also needs this operation

    # # pylatexenc's unicode to latex
    # content = unicode_to_latex(content, unknown_char_warning=False)
    # markdown_table_content[i, j] = LatexNodes2Text().latex_to_text(content_str)
    # content_ori = copy.deepcopy(content)

    # print('--------------After pre_process: \n', content)

    pred_all = []
    # deal with inline formula
    # content_new, inline_array = inline_filter_unicode(content)
    # #print('------------inline_array----------------',inline_array)
    # for inline_item in inline_array:
    #     inline_item['content'] = inline_to_unicode(inline_item['content'])
    #     #print('------------inline_array_unicode----------------',inline_item['content'])
    #     pred_all.append({
    #         'category_type': 'text_all',
    #         'position': inline_item['position'],
    #         'content': inline_item['content'],
    #         'fine_category_type': 'equation_inline'
    #     })
    
    pred_all.extend(image_formula_rescue)

    # extract latex table 
    latex_table_array, table_positions = extract_tex_table(content)
    for latex_table, position in zip(latex_table_array, table_positions):
        position = [position[0], position[0]+len(latex_table)]   # !!!
        normalized_html = normalized_latex_table(latex_table)
        pred_all.append({
            'category_type': 'html_table' if normalized_html else 'latex_table',
            'position': position,
            'content': normalized_html if normalized_html else latex_table,
            'fine_category_type': 'latex2html_table' if normalized_html else 'latex_table'
        })
        content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace latex table with space

    # print('--------After latex table: \n', content)
    # print('-------latex_table_array: \n', latex_table_array)

    # extract html table  
    html_table_array, table_positions = extract_html_table(content)
    for html_table, position in zip(html_table_array, table_positions):
        position = [position[0], position[0]+len(html_table)]
        pred_all.append({
            'category_type': 'html_table',
            'position': position,
            'content': html_table
        })
        pred_all.extend(_build_html_table_formula_rescue_items(html_table, position))
        content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace html table with space
    # html_table_array = []
    # html_table_matches = html_table_reg.finditer(content)
    # if html_table_matches:
    #     for match in html_table_matches:
    #         matched = match.group(0)
    #         position = [match.start(), match.end()]
    #         html_table_array.append(matched.strip())
    #         # content = content.replace(matched, ' '*len(matched)) # replace html table with space
    #         content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace html table with space
    #         pred_all.append({
    #             'category_type': 'html_table',
    #             'position': position,
    #             'content': matched.strip()
    #         })

    # print('--------------After html table: \n', content)
    # # extract tables in latex and html
    # table_array = []
    # table_matches = table_reg.finditer(content)
    # tables = ""
    # for match in table_matches:
    #     matched = match.group(0)
    #     if matched:
    #         tables += matched
    #         tables += "\n\n"
    #         table_array.append(matched)
    #         content = content.replace(matched, '')

    # extract interline formula
    display_matches = display_reg.finditer(content)
    content_copy = content
    for match in display_matches:
        matched = match.group(0)
        if matched:
            # single_line = ''.join(matched.split())
            single_line = ' '.join(matched.strip().split('\n'))
            position = [match.start(), match.end()]
            # replace $$ with \[\]
            dollar_pattern = re.compile(r'\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)', re.DOTALL)
            sub_match = dollar_pattern.search(single_line)
            if sub_match is None:
                # pass
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]
                _append_display_formula_item(pred_all, single_line, position)
            elif sub_match.group(1):
                single_line = re.sub(dollar_pattern, r'\\[\1\\]', single_line)
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace equation with space
                _append_display_formula_item(pred_all, single_line, position)
            else:
                # start, end = match.span()
                # char_before = content_copy[start-1] if start > 0           else '\n'
                # char_after  = content_copy[end]   if end   < len(content_copy) else '\n'
                # if char_before == '\n' or char_after == '\n':
                #     single_line = re.sub(dollar_pattern, r'\\[\2\3\\]', single_line)
                #     pred_all.append({
                #         'category_type': 'equation_isolated',
                #         'position': position,
                #         'content': single_line,
                #         'fine_category_type': 'equation_inline'
                #     })
                single_line = re.sub(dollar_pattern, r'\\[\2\3\\]', single_line)
                _append_display_formula_item(pred_all, single_line, position, fine_category_type='equation_inline')
            # single_line = re.sub(dollar_pattern, r'\\[\1\2\3\\]', single_line)
            # print('single_line: ', single_line)
            # content = content.replace(matched, ' '*len(matched))
            # pred_all.append({
            #     'category_type': 'equation_isolated',
            #     'position': position,
            #     'content': single_line
            # })
            # print('-----Found display formula: ', matched)

    # print('-------------After display: \n', content)
    # extract md table with ||
    md_table_mathces = md_table_reg.findall(content+'\n')
    if len(md_table_mathces) >= 2:
        # print("md table found!")
        # print("content:", content)
        content = convert_markdown_to_html(content)
        # print('----------content after converting md table to html:', content)
        html_table_matches = html_table_reg.finditer(content)
        if html_table_matches:
            for match in html_table_matches:
                matched = match.group(0)
                position = [match.start(), match.end()]
                # content = content.replace(match, '')
                # print('content after removing the md table:', content)
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace md table with space
                pred_all.append({
                    'category_type': 'html_table',
                    'position': position,
                    'content': matched.strip(),
                    'fine_category_type': 'md2html_table'
                })
                pred_all.extend(_build_html_table_formula_rescue_items(matched.strip(), position))
    # print('---------After md table: \n', content)

    # extract code blocks
    code_matches = code_block_reg.finditer(content)
    if code_matches:
        for match in code_matches:
            position = [match.start(), match.end()]
            language = match.group(1)
            code = match.group(2).strip()
            # content = content.replace(match.group(0), '')
            content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace code block with space
            pred_all.append({
                'category_type': 'text_all',
                'position': position,
                'content': code,
                'language': language,
                'fine_category_type': 'code'
            })

    # print('-------After code block: \n', content)

    # # Extract titles: Do not extract titles, as some models do not wrap code blocks, causing all comments to be treated as titles
    # title_matches = title_reg.finditer(content)
    # if title_matches:
    #     for match in title_matches:
    #         position = [match.start(), match.end()]
    #         matched = match.group(0)
    #         matched = matched.replace("#", "").strip()
    #         # content = content.replace(match, '')
    #         # print('content after removing the titles:', content)
    #         if matched:
    #             # print('Add title: ', matched)
    #             content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]
    #             pred_all.append({
    #                 'category_type': 'text_all',
    #                 'position': position,
    #                 'content': matched,
    #                 'fine_category_type': 'title'
    #             })
    
    # print('----------After title: \n', content)
            
    # # Delete extracted content
    # extracted_position = [_['position'] for _ in pred_all]
    # for start, end in sorted(extracted_position, reverse=True):
    #     content = content[:start] + content[end:]

    # print('----------After delete extracted: \n', content)

    # Remove latex style
    content = re.sub(r'\\title\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\title\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\section\*?\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\section\*?\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)

    # extract texts
    res = content.split('\n\n')
    if len(res) == 1:
        res = content.split('\n')  # some models do not use double newlines, so use single newlines to split

    content_position = 0
    for text in res:
        position = [content_position, content_position+len(text)]
        content_position += len(text)
        text = text.strip()
        text = text.strip('\n')
        # print('ori_text: ', text)
        text = '\n'.join([_.strip() for _ in text.split('\n') if _.strip()])   # avoid some single newline content with many spaces
        text = _normalize_html_text_fragment(text)
        # print('after strip text: ', text)

        if text:  # Check if the stripped text is not empty
            text = text.strip()
            if text:
                pred_all.append({
                    'category_type': 'text_all',
                    'position': position,
                    'content': text,
                    'fine_category_type': 'text_block'
                })
                _append_formula_rescue_candidate(pred_all, text, position, 'text_formula')
            # if '$' in text:
            #     for formula in re.findall(r'\$(.*?)\$', text):
            #         formula_array.append(formula)

    pred_dataset = defaultdict(list)
    pred_all = sorted(pred_all, key=lambda x: x['position'][0])
    for item in pred_all:
        if _should_drop_short_inline_formula_item(item):
            continue
        pred_dataset[item['category_type']].append(item)
    return pred_dataset


# def replace_or_extract(match):
#     content = match.group(1) if match.group(1) is not None else match.group(2)
    
#     if any(char in content for char in r'\^_'):
#         inline_array.append(match.group(0))
#         return ''
#     else:
#         return content

# extract inline math equations in text
# def inline_filter(text):

#     inline_array = []
#     inline_matches = inline_reg.finditer(text)
#     for match in inline_matches:
#         content = match.group(1) if match.group(1) is not None else match.group(2)
        
#         # remove \\, \_, \&, \%, \^
#         clean_content = re.sub(r'\\([\\_&%^])', '', content)

#         if any(char in clean_content for char in r'\^_'):
#             inline_array.append(match.group(0))
#             text = text.replace(match.group(0), '')
#         else:
#             text = text.replace(match.group(0), content)

#     return text, inline_array

# def extract_tex_table(content):
#     tables = []
#     positions = []

#     walker = LatexWalker(content)
#     nodes, _, _ = walker.get_latex_nodes()
#     if nodes is None:
#         return tables, positions

#     for node in nodes:
#         if isinstance(node, LatexEnvironmentNode) and (
#             node.environmentname == 'tabular' or node.environmentname == 'table'):
#             # table_latex = extract_node_content(node)
#             table_latex = content[node.pos:node.pos_end]
#             tables.append(table_latex)
#             start_pos = node.pos
#             end_pos = get_node_end_pos(node)
#             positions.append((start_pos, end_pos))

#     return tables, positions

def extract_tex_table(content):
    tables = []
    tables_positions = []

    pattern = r'\\begin{table}(.*?)\\end{table}'
    for match in re.finditer(pattern, content, re.DOTALL):
        start_pos = match.start()
        end_pos = match.end()
        table_content = match.group(0)
        tables.append(table_content)
        tables_positions.append((start_pos, end_pos))
        content = content[:start_pos] + ' '*(end_pos-start_pos) + content[end_pos:]

    tabulars, tabular_positions = extract_tabular(content)
    all_tables = tables + tabulars
    all_positions = tables_positions + tabular_positions

    all_result = sorted([[pos, table]for pos, table in zip(all_positions, all_tables)], key=lambda x: x[0][0])
    all_tables = [x[1] for x in all_result]
    all_positions = [x[0] for x in all_result]

    return all_tables, all_positions

# def extract_html_table(content):
#     soup = BeautifulSoup(content, 'html.parser')
#     all_tables = soup.find_all('table')
#     tables = []
#     positions = []
    
#     for table in all_tables:
#         if table.find_parent('table') is None:
#             table_str = str(table)
#             start_pos = content.find(table_str)
#             end_pos = start_pos + len(table_str)
            
#             tables.append(table_str)
#             positions.append((start_pos, end_pos))
#     return tables, positions

def extract_html_table(text):
    begin_pattern = r'<table(?:[^>]*)>'
    end_pattern = r'</table>'

    tabulars = []
    positions = []
    current_pos = 0
    stack = []
    
    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])
        
        if not begin_match and not end_match:
            break
            
        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1
    
    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_html_table(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions


def extract_node_content(node):
    """ Recursively extract content from LatexEnvironmentNode and rebuild LaTeX table representation """
    if isinstance(node, LatexCharsNode):
        return node.chars  # Use chars attribute
    elif isinstance(node, LatexGroupNode):
        return "{" + "".join(extract_node_content(n) for n in node.nodelist) + "}"
    elif isinstance(node, LatexMacroNode):
        # Extract macro command and its arguments
        macro_content = "\\" + node.macroname
        if node.nodeargs:
            macro_content += "".join([extract_node_content(arg) for arg in node.nodeargs])
        return macro_content
    elif isinstance(node, LatexEnvironmentNode):
        # Extract environment, preserve environment name and arguments
        content = "\\begin{" + node.environmentname + "}"
        if node.nodeargd and node.nodeargd.argnlist:
            # content += "".join("{" + extract_node_content(arg) + "}" for arg in node.nodeargd)
            # content += "".join("{" + extract_node_content(node.nodeargd) + "}")
            content += "{" + extract_node_content(node.nodeargd.argnlist[0]) + "}"
        if node.nodelist:
            content += "".join(extract_node_content(n) for n in node.nodelist)
        content += "\\end{" + node.environmentname + "}"
        return content
    elif isinstance(node, LatexSpecialsNode):  # Changed to LatexSpecialsNode
        return node.specials_chars
    else:
        return ""
        
def get_node_end_pos(node):
    """Recursively determine the end position of a node"""
    if hasattr(node, 'nodelist') and node.nodelist:
        # If the node has child nodes, recursively find the end position of the last child node
        return get_node_end_pos(node.nodelist[-1])
    elif hasattr(node, 'pos_end'):
        # If the node has pos_end attribute, return it directly
        return node.pos_end
    else:
        # If there are no child nodes, assume the node ends at the last character of its content
        return node.pos + len(str(node))

def remove_tex_table(content):
    tables, positions = extract_tex_table(content)

    # Delete in reverse order by position to avoid affecting unprocessed start positions
    for start, end in sorted(positions, reverse=True):
        content = content[:start] + content[end:]  # Remove table content

    return content
