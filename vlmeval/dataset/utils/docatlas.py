"""Page scoring of DocAtlas-Bench, as in the official script (https://github.com/ahmedheakl/DocAtlas/tree/main/eval).

The official script uses the matching of OmniDocBench v1.5. Whatever is unchanged since the version in
``vlmeval/dataset/OmniDocBench`` is imported from there; the functions below are the ones that changed,
without the branches that DocAtlas annotations do not reach (formulas, ignored categories, truncated blocks).
"""
import re
from collections import defaultdict
from copy import deepcopy

import Levenshtein
from bs4 import BeautifulSoup
from func_timeout import FunctionTimedOut, func_timeout
from scipy.optimize import linear_sum_assignment

from ..OmniDocBench.metrics import TEDS
from ..OmniDocBench.utils import (cal_final_match, clean_string, code_block_reg,
                                  compute_edit_distance_matrix_new, convert_markdown_to_html,
                                  display_reg, extract_html_table, extract_tex_table,
                                  fuzzy_match_unmatched_items, get_pred_category_type,
                                  html_table_reg, img_pattern, initialize_indices,
                                  match_gt2pred_simple, md_table_reg, merge_matches,
                                  normalized_table, process_matches, recalculate_edit_distances,
                                  textblock2unicode)

TABLE_TYPES = ('html_table', 'latex_table', 'md2html_table')
ARRAY_RE = re.compile(r'\\begin\{array\}\{(?P<spec>[^}]*)\}(?P<body>.*?)\\end\{array\}', re.S)
DOLLAR_RE = re.compile(r'\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)', re.DOTALL)
FORMULA_FILTERS = [
    '\\mathbf', '\\mathrm', '\\mathnormal', '\\mathit', '\\mathbb', '\\mathcal', '\\mathscr', '\\mathfrak',
    '\\mathsf', '\\mathtt', '\\textbf', '\\text', '\\boldmath', '\\boldsymbol', '\\operatorname', '\\bm',
    '\\symbfit', '\\mathbfcal', '\\symbf', '\\scriptscriptstyle', '\\notag', '\\setlength', '\\coloneqq',
    '\\space', '\\thickspace', '\\thinspace', '\\medspace', '\\nobreakspace', '\\negmedspace', '\\quad',
    '\\qquad', '\\enspace', '\\substackw', ' ', '$$', '\\left', '\\right', '\\displaystyle', '\\text']


def md_tex_filter(content):
    """Split a Markdown prediction into typed segments."""
    content = re.sub(img_pattern, '', content)
    for fence in (r'^```markdown\n?', r'^```html\n?', r'^```latex\n?', r'```\n?$'):
        content = re.sub(fence, '', content, flags=re.MULTILINE)
    content = re.sub(r' {4,}', '    ', re.sub(r'_{4,}', '____', content))
    content = content.replace('<html>', '').replace('</html>', '').replace('<body>', '').replace('</body>', '')
    items = []

    def add(category, position, text, **extra):
        items.append(dict(category_type=category, position=position, content=text, **extra))

    def blank(position):
        return content[:position[0]] + ' ' * (position[1] - position[0]) + content[position[1]:]

    for extract, category in ((extract_tex_table, 'latex_table'), (extract_html_table, 'html_table')):
        tables, positions = extract(content)
        for table, position in zip(tables, positions):
            position = [position[0], position[0] + len(table)]
            add(category, position, table)
            content = blank(position)

    for match in display_reg.finditer(content):
        if not match.group(0):
            continue
        single_line = ' '.join(match.group(0).strip().split('\n'))
        position = [match.start(), match.end()]
        sub_match = DOLLAR_RE.search(single_line)
        if sub_match is None:
            content = blank(position)
            add('equation_isolated', position, single_line)
        elif sub_match.group(1):
            content = blank(position)
            add('equation_isolated', position, re.sub(DOLLAR_RE, r'\\[\1\\]', single_line))
        else:
            add('equation_isolated', position, re.sub(DOLLAR_RE, r'\\[\2\3\\]', single_line),
                fine_category_type='equation_inline')

    if len(md_table_reg.findall(content + '\n')) >= 2:
        content = convert_markdown_to_html(content)
        for match in html_table_reg.finditer(content):
            position = [match.start(), match.end()]
            content = blank(position)
            add('html_table', position, match.group(0).strip(), fine_category_type='md2html_table')

    for match in code_block_reg.finditer(content):
        position = [match.start(), match.end()]
        content = blank(position)
        add('text_all', position, match.group(2).strip(), language=match.group(1), fine_category_type='code')

    content = re.sub(r'\\title\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\title\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\section\*?\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\section\*?\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)

    blocks = content.split('\n\n')
    if len(blocks) == 1:
        blocks = content.split('\n')
    start = 0
    for text in blocks:
        position = [start, start + len(text)]
        start += len(text)
        text = '\n'.join(line.strip() for line in text.strip().split('\n') if line.strip())
        if text.startswith('<table') and text.endswith('</table>'):
            add('html_table', position, text)
        elif text.startswith('$') and text.endswith('$'):
            if text.replace('$', '').strip():
                add('equation_isolated', position, text)
        elif text:
            add('text_all', position, text, fine_category_type='text_block')

    pred = defaultdict(list)
    for item in sorted(items, key=lambda item: item['position'][0]):
        pred[item['category_type']].append(item)
    return pred


def normalized_formula(text):
    text = text.strip().strip('$').strip('\n')
    match = re.search(r'\\\[(.+?)(?<!\\)\\\]', text)
    if match:
        text = match.group(1).strip()
    for pattern in (r'\\tag\{.*?\}', r'\\hspace\{.*?\}', r'\\begin\{.*?\}', r'\\end\{.*?\}', r'\\arraycolsep.*?\}'):
        text = re.sub(pattern, '', text)
    text = text.strip('.')
    for filter_text in FORMULA_FILTERS:
        text = text.replace(filter_text, '')
    return text.lower()


def normalized_text(item):
    """Text of a ground-truth block or a predicted segment, as it is compared."""
    if item['category_type'] == 'equation_isolated':
        return normalized_formula(str(item['content']))
    return clean_string(textblock2unicode(str(item.get('content', item.get('text', '')))))


def split_equation_arrays(items):
    """Split predicted single-column formula arrays into one segment per line."""
    out = []
    for item in items:
        match = ARRAY_RE.search(item['content']) if item['category_type'] == 'equation_isolated' else None
        spec = re.sub(r'\s+|\|', '', match.group('spec')) if match else ''
        if re.sub(r'!{[^}]*}', '', re.sub(r'@{[^}]*}', '', spec)) not in ('l', 'c', 'r'):
            out.append(item)
            continue
        body, cursor = match.group('body'), 0
        for line in [line.strip() for line in re.split(r'\\\\', body) if line.strip()]:
            index = body.find(line, cursor)
            index = cursor if index == -1 else index
            cursor = index + len(line)
            start = item['position'][0] + match.start('body') + index
            out.append({**deepcopy(item), 'content': f'\\[{line}\\]', 'position': [start, start + len(line) - 1]})
    return out


def convert_final_matches(final_matches, gt_lines, pred_lines):
    results = []
    for pred_key, info in final_matches.items():
        for gt_idx in sorted(set(info['gt_indices'])):
            results.append({'gt_idx': int(gt_idx), 'pred_idx': list(pred_key)})

    matched_gt = set().union(*[set(info['gt_indices']) for info in final_matches.values()])
    unmatched_gt = set(range(len(gt_lines))) - matched_gt
    matched_pred = set(idx for pred_key in final_matches.keys() for idx in pred_key if isinstance(idx, int))
    unmatched_pred = set(range(len(pred_lines))) - matched_pred
    if unmatched_pred and unmatched_gt:
        distance = [[Levenshtein.distance(gt_lines[i], pred_lines[j]) / max(len(gt_lines[i]), len(pred_lines[j]))
                     for j in unmatched_pred] for i in unmatched_gt]
        for i, j in zip(*linear_sum_assignment(distance)):
            results.append({'gt_idx': int(list(unmatched_gt)[i]), 'pred_idx': [list(unmatched_pred)[j]]})
    elif unmatched_pred:
        results.append({'gt_idx': '', 'pred_idx': list(unmatched_pred)})
    else:
        results.extend({'gt_idx': int(gt_idx), 'pred_idx': ''} for gt_idx in unmatched_gt)
    return results


def merge_duplicates_add_unmatched(results, gt_lines):
    merged, processed_pred, processed_gt = [], set(), set()
    for entry in results:
        pred_idx = tuple(entry['pred_idx']) if isinstance(entry['pred_idx'], list) else (entry['pred_idx'],)
        if pred_idx in processed_pred or pred_idx == ('',):
            continue
        gt_idx = [entry['gt_idx']]
        for other in results:
            other_pred_idx = tuple(other['pred_idx']) if isinstance(other['pred_idx'], list) else (other['pred_idx'],)
            if other_pred_idx == pred_idx and other is not entry:
                gt_idx.append(other['gt_idx'])
                processed_gt.add(other['gt_idx'])
        merged.append({'gt_idx': gt_idx, 'pred_idx': entry['pred_idx']})
        processed_pred.add(pred_idx)
        processed_gt.add(entry['gt_idx'])
    unmatched_gt = [gt_idx for gt_idx in range(len(gt_lines)) if gt_idx not in processed_gt]
    merged.extend({'gt_idx': [gt_idx], 'pred_idx': ['']} for gt_idx in unmatched_gt)
    return merged


def match_quick(gt_items, pred_items):
    """Match ground-truth text blocks to predicted segments; one entry per matched or missed block."""
    pred_items = sorted(pred_items, key=lambda item: (item.get('fine_category_type') == 'equation_inline',
                                                      item['position'][0]))
    pred_items = split_equation_arrays(pred_items)
    gt_lines = [line for line in map(normalized_text, gt_items) if line]
    pred_lines = [line for line in map(normalized_text, pred_items) if line]

    def entry(gt_idx, pred_idx):
        return {
            'norm_gt': ''.join(gt_lines[i] for i in gt_idx),
            'norm_pred': ''.join(pred_lines[i] for i in pred_idx),
            'gt_position': [gt_items[i].get('order') or '' for i in gt_idx],
            'pred_position': pred_items[pred_idx[0]]['position'][0] if pred_idx else '',
        }

    if not gt_lines:
        return []
    if not pred_lines:
        return [entry([i], []) for i in range(len(gt_lines))]
    if len(gt_lines) == 1 and len(pred_lines) == 1:
        return [entry([0], [0])]

    cost_matrix = compute_edit_distance_matrix_new(gt_lines, pred_lines)
    matched_col_idx, row_ind, cost_list = cal_final_match(cost_matrix, gt_lines, pred_lines)
    gt_lens_dict, _ = initialize_indices(gt_lines, pred_lines)
    matches, unmatched_gt, _ = process_matches(matched_col_idx, row_ind, cost_list, gt_lines, pred_lines, pred_lines)
    final_matches = merge_matches(matches, fuzzy_match_unmatched_items(unmatched_gt, gt_lines, pred_lines))
    recalculate_edit_distances(final_matches, gt_lens_dict, gt_lines, pred_lines)
    merged = merge_duplicates_add_unmatched(convert_final_matches(final_matches, gt_lines, pred_lines), gt_lines)

    entries, skip = [], False
    for match in merged:
        if skip:  # upstream removes the entry of unmatched inline formulas while iterating, which skips the next
            skip = False
        elif match['gt_idx'] == ['']:
            skip = get_pred_category_type(match['pred_idx'][0], pred_items) == 'equation_inline'
        else:
            pred_idx = match['pred_idx'] if isinstance(match['pred_idx'], list) else [match['pred_idx']]
            entries.append(entry(match['gt_idx'], [] if pred_idx == [''] else pred_idx))
    return entries


def match_simple(gt_items, pred_items):
    """The fallback of the official script when ``match_quick`` takes more than 30 s."""
    gt_items = [{**item, 'content': normalized_text(item), 'html': ''} for item in gt_items]
    pred_items = [{**item, 'content': normalized_text(item)} for item in pred_items]
    return [match for match in match_gt2pred_simple(gt_items, pred_items, 'html_table', '') if match['gt_idx'] != ['']]


def reading_order_edit(matches):
    matches = [match for match in matches if match['gt_position'] != ['']]
    gt = sorted(position for match in matches for position in match['gt_position'] if position)
    in_pred_order = sorted((match for match in matches if match['pred_position'] != ''),
                           key=lambda match: match['pred_position'])
    pred = [position for match in in_pred_order for position in match['gt_position']]
    if not gt and not pred:
        return None
    return Levenshtein.distance(gt, pred) / max(len(gt), len(pred))


def score_page(page, prediction):
    """Return the text edit distances, table TEDS scores and reading-order edit distance of one page."""
    gt = defaultdict(list)
    for item in sorted(page['layout_dets'], key=lambda item: item['order']):
        gt[item['category_type']].append(item)
    pred = md_tex_filter(prediction)
    pred_mix = [item for category, items in pred.items() if category not in TABLE_TYPES for item in items]

    tables = []
    if gt['table']:
        # LaTeX tables are matched when they outnumber the HTML tables, but they score 0.
        latex = len(pred['latex_table']) > len(pred['html_table'])
        pred_tables = pred['latex_table' if latex else 'html_table']
        for match in match_gt2pred_simple(gt['table'], pred_tables, 'html_table', ''):
            if match['gt_idx'] == ['']:  # the cells of unmatched tables take part in the text matching
                for table in (pred_tables[index] for index in match['pred_idx']):
                    for cell in BeautifulSoup(table['content'], 'html.parser').find_all('td'):
                        if cell.string:
                            text = re.sub(r'\$\\cdot\$', '', cell.string).strip()
                            pred_mix.append({**deepcopy(table), 'content': text, 'category_type': 'text_all'})
                continue
            gt_html, pred_html = match['gt'], '' if latex else match['pred']
            try:
                tables.append(TEDS().evaluate(normalized_table(pred_html) or pred_html,
                                              normalized_table(gt_html) or gt_html))
            except Exception:
                tables.append(0.0)

    try:
        matches = func_timeout(30, match_quick, args=(gt['text_block'], pred_mix))
    except FunctionTimedOut:
        matches = match_simple(gt['text_block'], pred_mix)
    text = [Levenshtein.distance(match['norm_pred'], match['norm_gt'])
            / max(len(match['norm_pred']), len(match['norm_gt'])) for match in matches]

    language = page['page_info']['page_attribute']['language']
    return dict(language=language, text=text, tables=tables, reading_order=reading_order_edit(matches))
