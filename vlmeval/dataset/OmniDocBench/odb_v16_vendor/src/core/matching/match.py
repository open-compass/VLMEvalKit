from scipy.optimize import linear_sum_assignment
import Levenshtein
import numpy as np
import re
import sys
from src.core.preprocess.data_preprocess import (
    normalized_formula,
    normalized_html_table,
    normalized_text,
    textblock_with_norm_formula,
)
from src.core.preprocess.table_postprocess import table_to_text_lines
from copy import deepcopy
from loguru import logger

class MatchResult(list):
    def __init__(self, iterable=(), fallback_reason=None):
        super().__init__(iterable)
        self.fallback_reason = fallback_reason


def get_pred_category_type(pred_idx, pred_items):
    if pred_items[pred_idx].get('fine_category_type'):
        pred_pred_category_type = pred_items[pred_idx]['fine_category_type']
    else:
        pred_pred_category_type = pred_items[pred_idx]['category_type']
    return pred_pred_category_type


def compute_edit_distance_matrix_new(gt_lines, matched_lines):
    try:
        distance_matrix = np.zeros((len(gt_lines), len(matched_lines)))
        for i, gt_line in enumerate(gt_lines):
            for j, matched_line in enumerate(matched_lines):
                if len(gt_line) == 0 and len(matched_line) == 0:
                    distance_matrix[i][j] = 0  
                else:
                    distance_matrix[i][j] = Levenshtein.distance(gt_line, matched_line) / max(len(matched_line), len(gt_line))
        return distance_matrix
    except ZeroDivisionError:
        raise  


def normalize_table_match_content(text):
    text = str(text or '')
    if not text:
        return ''
    if '<table' in text.lower():
        return normalized_html_table(text)
    return text


def split_pred_table_to_text_items(pred_items):
    text_items = []
    for original_item in pred_items or []:
        table_lines = table_to_text_lines(original_item.get('content', ''))
        if not table_lines:
            continue

        base_position = original_item.get('position', [0, 0])
        base_start = base_position[0] if isinstance(base_position, list) and base_position else 0

        for offset, text_line in enumerate(table_lines):
            new_item = deepcopy(original_item)
            new_item['content'] = text_line
            new_item['category_type'] = 'text_all'
            new_item['fine_category_type'] = 'table_to_text'
            new_item['position'] = [base_start + offset, base_start + offset + max(1, len(text_line))]
            text_items.append(new_item)
    return text_items


TEXT_GT_CATEGORIES = {
    'text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
    'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm',
    'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number'
}


def _position_sort_key(item):
    order = item.get('order')
    if isinstance(order, (int, float)):
        return (0, order)

    position = item.get('position', [None])
    if isinstance(position, list) and position:
        first = position[0]
        if isinstance(first, (int, float)):
            return (1, first)
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (int, float)):
            return (1, first[0])

    return (2, 0)


def _normalized_edit_distance(gt_text, pred_text):
    if not gt_text and not pred_text:
        return 0
    if not gt_text or not pred_text:
        return 1
    return Levenshtein.distance(gt_text, pred_text) / max(len(gt_text), len(pred_text))


def _timeout_match_threshold(norm_text):
    length = len(norm_text or '')
    if length <= 6:
        return 0.40
    if length <= 20:
        return 0.55
    if length <= 80:
        return 0.68
    return 0.78


def _build_timeout_fallback_gt_records(gt_items):
    records = []
    for source_idx, item in enumerate(gt_items or []):
        gt_category = item.get('fine_category_type') or item.get('category_type') or ''
        if item.get('content'):
            raw_text = str(item['content'])
            norm_text = normalized_text(raw_text) if item.get('category_type') == 'text_all' else raw_text
        elif item.get('category_type') in TEXT_GT_CATEGORIES:
            raw_text = str(item.get('text', ''))
            norm_text = normalized_text(raw_text)
        elif item.get('category_type') == 'equation_isolated':
            raw_text = str(item.get('latex', ''))
            norm_text = normalized_formula(raw_text)
        else:
            continue

        if not raw_text or not norm_text:
            continue

        records.append({
            'item': item,
            'source_idx': source_idx,
            'raw': raw_text,
            'norm': norm_text,
            'category': gt_category,
            'position': [item.get('order') if item.get('order') else item.get('position', [""])[0]],
            'attribute': [item.get('attribute', {})],
        })
    return records


def _build_timeout_fallback_pred_records(pred_items):
    records = []
    for source_idx, item in enumerate(pred_items or []):
        pred_category = item.get('fine_category_type') or item.get('category_type') or ''
        raw_text = str(item.get('content', ''))
        if not raw_text:
            continue

        if item.get('category_type') == 'text_all':
            norm_text = normalized_text(raw_text)
        elif item.get('category_type') == 'equation_isolated':
            norm_text = normalized_formula(raw_text)
        else:
            norm_text = raw_text

        if not norm_text:
            continue

        records.append({
            'item': item,
            'source_idx': source_idx,
            'raw': raw_text,
            'norm': norm_text,
            'category': pred_category,
            'position': item.get('position', [""])[0] if item.get('position') else '',
        })
    return sorted(records, key=lambda record: _position_sort_key(record['item']))


def _safe_positive_int(value, default, minimum=1, maximum=None):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _safe_non_negative_float(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(0.0, value)


def _record_is_formula(record):
    item = record.get('item', {})
    if item.get('category_type') == 'equation_isolated':
        return True
    return record.get('category') in {'equation_isolated', 'equation_inline', 'formula_rescue'}


def _record_is_text_like(record):
    return not _record_is_formula(record)


def _is_open_ended_text(raw_text):
    text = str(raw_text or '').rstrip()
    if not text:
        return False
    return text[-1] not in '.!?。！？;；:：)]}」』”’"'


def _starts_with_continuation(raw_text):
    text = str(raw_text or '').lstrip()
    if not text:
        return False
    return text[0].islower() or text[0].isdigit() or text[0] in ',.;:!?)]}%'


def _resolve_chunked_hungarian_params(
    gt_records,
    short_line_max_chars=None,
    target_chunk_chars=None,
    max_chunk_chars=None,
    max_chunk_span=8,
    order_window=None,
    order_penalty=0.08,
):
    gt_text_lengths = [len(record['norm']) for record in gt_records if record.get('norm') and _record_is_text_like(record)]
    if gt_text_lengths:
        gt_median_len = int(np.median(gt_text_lengths))
    else:
        gt_median_len = 120

    short_line_max_chars = _safe_positive_int(
        short_line_max_chars,
        default=max(24, min(72, int(gt_median_len * 0.45))),
        minimum=12,
        maximum=160,
    )
    target_chunk_chars = _safe_positive_int(
        target_chunk_chars,
        default=max(short_line_max_chars * 2, min(220, int(gt_median_len * 0.95))),
        minimum=short_line_max_chars,
        maximum=420,
    )
    max_chunk_chars = _safe_positive_int(
        max_chunk_chars,
        default=max(target_chunk_chars + 48, min(320, int(gt_median_len * 1.35))),
        minimum=target_chunk_chars,
        maximum=640,
    )
    max_chunk_span = _safe_positive_int(max_chunk_span, default=8, minimum=1, maximum=32)
    order_penalty = _safe_non_negative_float(order_penalty, default=0.08)

    if order_window is None:
        order_window = None
    else:
        order_window = _safe_positive_int(order_window, default=12, minimum=2, maximum=128)

    return {
        'short_line_max_chars': short_line_max_chars,
        'target_chunk_chars': target_chunk_chars,
        'max_chunk_chars': max_chunk_chars,
        'max_chunk_span': max_chunk_span,
        'order_window': order_window,
        'order_penalty': order_penalty,
    }


def _make_pred_chunk(span_records):
    raw_parts = [record['raw'] for record in span_records if record.get('raw')]
    norm_parts = [record['norm'] for record in span_records if record.get('norm')]
    if not raw_parts or not norm_parts:
        return None

    if all(record.get('category') == span_records[0].get('category') for record in span_records):
        category = span_records[0].get('category', '')
    elif all(_record_is_text_like(record) for record in span_records):
        category = 'text_block'
    else:
        category = span_records[0].get('category', '')

    return {
        'pred_indices': [record['source_idx'] for record in span_records],
        'raw': ' '.join(raw_parts),
        'norm': ''.join(norm_parts),
        'category': category,
        'position': span_records[0].get('position', ''),
        'records': span_records,
    }


def _build_timeout_fallback_pred_chunks(pred_records, chunk_params):
    if not pred_records:
        return []

    short_line_max_chars = chunk_params['short_line_max_chars']
    target_chunk_chars = chunk_params['target_chunk_chars']
    max_chunk_chars = chunk_params['max_chunk_chars']
    max_chunk_span = chunk_params['max_chunk_span']

    pred_chunks = []
    idx = 0
    while idx < len(pred_records):
        current_record = pred_records[idx]
        span_records = [current_record]
        idx += 1

        if _record_is_text_like(current_record):
            current_norm_len = len(current_record.get('norm', ''))
            while idx < len(pred_records):
                next_record = pred_records[idx]
                if not _record_is_text_like(next_record):
                    break
                if len(span_records) >= max_chunk_span:
                    break

                next_norm_len = len(next_record.get('norm', ''))
                merged_norm_len = current_norm_len + next_norm_len
                if merged_norm_len > max_chunk_chars:
                    break

                should_merge = (
                    current_norm_len < target_chunk_chars
                    or next_norm_len <= short_line_max_chars
                    or _is_open_ended_text(span_records[-1].get('raw', ''))
                    or _starts_with_continuation(next_record.get('raw', ''))
                )
                if not should_merge:
                    break

                span_records.append(next_record)
                idx += 1
                current_norm_len = merged_norm_len

                if current_norm_len >= target_chunk_chars and not _is_open_ended_text(span_records[-1].get('raw', '')):
                    break

        chunk = _make_pred_chunk(span_records)
        if chunk is not None:
            pred_chunks.append(chunk)

    return pred_chunks


def _build_chunked_cost_matrices(gt_records, pred_chunks, order_window=None, order_penalty=0.08):
    gt_norms = [record['norm'] for record in gt_records]
    pred_norms = [chunk['norm'] for chunk in pred_chunks]
    edit_matrix = compute_edit_distance_matrix_new(gt_norms, pred_norms)
    cost_matrix = np.array(edit_matrix, copy=True)

    gt_count = len(gt_records)
    pred_count = len(pred_chunks)
    if gt_count <= 1 or pred_count <= 1:
        normalized_gt_den = 1
        normalized_pred_den = 1
    else:
        normalized_gt_den = gt_count - 1
        normalized_pred_den = pred_count - 1

    for gt_idx, gt_record in enumerate(gt_records):
        gt_is_formula = _record_is_formula(gt_record)
        expected_pred_idx = 0
        if pred_count > 1 and gt_count > 1:
            expected_pred_idx = round(gt_idx * normalized_pred_den / normalized_gt_den)
        for pred_idx, pred_chunk in enumerate(pred_chunks):
            penalty = 0.0
            pred_is_formula = any(_record_is_formula(record) for record in pred_chunk['records'])
            if gt_is_formula and not pred_is_formula:
                penalty += 0.18
            elif not gt_is_formula and pred_is_formula:
                penalty += 0.10

            if order_window is not None:
                overflow = abs(pred_idx - expected_pred_idx) - order_window
                if overflow > 0:
                    penalty += 0.30 + 0.02 * overflow
            else:
                normalized_gap = abs((gt_idx / normalized_gt_den) - (pred_idx / normalized_pred_den))
                penalty += order_penalty * normalized_gap

            cost_matrix[gt_idx][pred_idx] += penalty

    return edit_matrix, cost_matrix


def match_gt2pred_timeout_safe(
    gt_items,
    pred_items,
    img_name,
    short_line_max_chars=None,
    target_chunk_chars=None,
    max_chunk_chars=None,
    max_chunk_span=8,
    order_window=None,
    order_penalty=0.08,
    fallback_reason=None,
):
    gt_records = _build_timeout_fallback_gt_records(gt_items)
    pred_records = _build_timeout_fallback_pred_records(pred_items)
    match_list = []

    chunk_params = _resolve_chunked_hungarian_params(
        gt_records,
        short_line_max_chars=short_line_max_chars,
        target_chunk_chars=target_chunk_chars,
        max_chunk_chars=max_chunk_chars,
        max_chunk_span=max_chunk_span,
        order_window=order_window,
        order_penalty=order_penalty,
    )
    pred_chunks = _build_timeout_fallback_pred_chunks(pred_records, chunk_params)

    if chunk_params['order_window'] is None and pred_chunks:
        chunk_params['order_window'] = max(6, min(48, len(pred_chunks) // 6 + 6))

    print(
        f"[timeout-fallback] {img_name}: gt={len(gt_records)} pred={len(pred_records)} chunk={len(pred_chunks)} short<={chunk_params['short_line_max_chars']} target={chunk_params['target_chunk_chars']} max={chunk_params['max_chunk_chars']} span={chunk_params['max_chunk_span']} window={chunk_params['order_window']}",
        flush=True,
    )

    def append_unmatched_gt(gt_record):
        match_list.append({
            'gt_idx': [gt_record['source_idx']],
            'gt': gt_record['raw'],
            'norm_gt': gt_record['norm'],
            'gt_category_type': gt_record['category'],
            'gt_position': gt_record['position'],
            'gt_attribute': gt_record['attribute'],
            'pred_idx': [""],
            'pred': '',
            'norm_pred': '',
            'pred_category_type': '',
            'pred_position': '',
            'edit': 1,
            'img_id': img_name,
        })

    def append_unmatched_pred(pred_chunk):
        match_list.append({
            'gt_idx': [""],
            'gt': '',
            'norm_gt': '',
            'gt_category_type': '',
            'gt_position': [""],
            'gt_attribute': [{}],
            'pred_idx': pred_chunk['pred_indices'],
            'pred': pred_chunk['raw'],
            'norm_pred': pred_chunk['norm'],
            'pred_category_type': pred_chunk['category'],
            'pred_position': pred_chunk['position'],
            'edit': 1,
            'img_id': img_name,
        })

    if not gt_records:
        for pred_chunk in pred_chunks:
            append_unmatched_pred(pred_chunk)
        return MatchResult(match_list, fallback_reason=fallback_reason)

    if not pred_chunks:
        for gt_record in gt_records:
            append_unmatched_gt(gt_record)
        return MatchResult(match_list, fallback_reason=fallback_reason)

    edit_matrix, cost_matrix = _build_chunked_cost_matrices(
        gt_records,
        pred_chunks,
        order_window=chunk_params['order_window'],
        order_penalty=chunk_params['order_penalty'],
    )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    row_to_col = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    matched_chunk_indices = set()

    for gt_idx, gt_record in enumerate(gt_records):
        pred_chunk_idx = row_to_col.get(gt_idx)
        if pred_chunk_idx is None:
            append_unmatched_gt(gt_record)
            continue

        pred_chunk = pred_chunks[pred_chunk_idx]
        pure_edit = float(edit_matrix[gt_idx][pred_chunk_idx])
        threshold = _timeout_match_threshold(gt_record['norm'])
        gt_is_formula = _record_is_formula(gt_record)
        pred_is_formula = any(_record_is_formula(record) for record in pred_chunk['records'])
        if gt_is_formula != pred_is_formula:
            threshold -= 0.08
        threshold = max(0.25, threshold)

        if pure_edit <= threshold:
            matched_chunk_indices.add(pred_chunk_idx)
            match_list.append({
                'gt_idx': [gt_record['source_idx']],
                'gt': gt_record['raw'],
                'norm_gt': gt_record['norm'],
                'gt_category_type': gt_record['category'],
                'gt_position': gt_record['position'],
                'gt_attribute': gt_record['attribute'],
                'pred_idx': pred_chunk['pred_indices'],
                'pred': pred_chunk['raw'],
                'norm_pred': pred_chunk['norm'],
                'pred_category_type': pred_chunk['category'],
                'pred_position': pred_chunk['position'],
                'edit': pure_edit,
                'img_id': img_name,
            })
        else:
            append_unmatched_gt(gt_record)

    for pred_chunk_idx, pred_chunk in enumerate(pred_chunks):
        if pred_chunk_idx not in matched_chunk_indices:
            append_unmatched_pred(pred_chunk)

    return MatchResult(match_list, fallback_reason=fallback_reason)


## 混合匹配here  0403
def get_gt_pred_lines(gt_mix,pred_dataset_mix,line_type):

    norm_html_lines,gt_lines,pred_lines,norm_gt_lines,norm_pred_lines,gt_cat_list = [],[],[],[],[],[]
    if line_type in ['html_table','latex_table']:
        for item in gt_mix:
            if item.get('fine_category_type'):
                gt_cat_list.append(item['fine_category_type'])
            else:
                gt_cat_list.append(item['category_type'])
            if item.get('content'):
                table_content = str(item['content'])
            elif line_type == 'html_table':
                table_content = str(item.get('html', ''))
            elif line_type == 'latex_table':
                table_content = str(item.get('html', '') or item.get('latex', ''))
            else:
                table_content = ''
            gt_lines.append(table_content)
            norm_html_lines.append(normalize_table_match_content(table_content))
        
        pred_lines = [str(item['content']) for item in pred_dataset_mix]
        if line_type == 'formula':
            norm_gt_lines = [normalized_formula(_) for _ in gt_lines]
            norm_pred_lines = [normalized_formula(_) for _ in pred_lines]
        elif line_type == 'text':
            norm_gt_lines = [normalized_text(_) for _ in gt_lines]
            norm_pred_lines = [normalized_text(_) for _ in pred_lines]
        else:
            norm_gt_lines = norm_html_lines
            norm_pred_lines = [normalize_table_match_content(_) for _ in pred_lines]

    else:
        for item in pred_dataset_mix:
            # text
            if item['category_type'] == 'text_all':
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(normalized_text(str(item['content'])))
            # formula
            elif  item['category_type']=='equation_isolated':
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(normalized_formula(str(item['content'])))
            # table
            else:
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(str(item['content']))
        
        for item in gt_mix:
            if item.get('content'):
                gt_lines.append(str(item['content']))
                if item['category_type'] == 'text_all':              
                    norm_gt_lines.append(normalized_text(str(item['content'])))
                else:
                   norm_gt_lines.append(item['content'])
                
                norm_html_lines.append(str(item['content']))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])
            # text      
            elif item['category_type'] in ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption','figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption','header', 'footer', 'page_footnote', 'page_number']:
                gt_lines.append(str(item['text']))
                norm_gt_lines.append(normalized_text(str(item['text'])))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])

            # formula
            elif item['category_type'] == 'equation_isolated':
                gt_lines.append(str(item['latex']))
                norm_gt_lines.append(normalized_formula(str(item['latex'])))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])
            # table
            # elif item['category_type'] == 'table':
            #     gt_lines.append(str(item['html']))
            #     norm_gt_lines.append(str(item['html']))

            #     if item.get('fine_category_type'):
            #         gt_cat_list.append(item['fine_category_type'])
            #     else:
            #         gt_cat_list.append(item['category_type'])


    filtered_lists = [(a, b, c) for a, b, c in zip(gt_lines, norm_gt_lines, gt_cat_list) if a and b]

    # decompress to three lists
    if filtered_lists:
        gt_lines_c, norm_gt_lines_c, gt_cat_list_c = zip(*filtered_lists)

        # convert to lists
        gt_lines_c = list(gt_lines_c)
        norm_gt_lines_c = list(norm_gt_lines_c)
        gt_cat_list_c = list(gt_cat_list_c)
    else:
        gt_lines_c = []
        norm_gt_lines_c = []
        gt_cat_list_c = []

    # pred's empty values
    filtered_lists = [(a, b) for a, b in zip(pred_lines, norm_pred_lines) if a and b]

    # decompress to two lists
    if filtered_lists:
        pred_lines_c, norm_pred_lines_c = zip(*filtered_lists)

        # convert to lists
        pred_lines_c = list(pred_lines_c)
        norm_pred_lines_c = list(norm_pred_lines_c)
    else:
        pred_lines_c = []
        norm_pred_lines_c = []

    return gt_lines_c, norm_gt_lines_c, gt_cat_list_c, pred_lines_c, norm_pred_lines_c, gt_mix, pred_dataset_mix


def match_gt2pred_simple(gt_items, pred_items, line_type, img_name):

    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines, gt_items, pred_items = get_gt_pred_lines(gt_items, pred_items,line_type)
    match_list = []

    if not norm_gt_lines: # not matched pred should be concatenate
        pred_idx_list = range(len(norm_pred_lines))
        match_list.append({
            'gt_idx': [""],
            'gt': "",
            'pred_idx': pred_idx_list,
            'pred': ''.join(pred_lines[_] for _ in pred_idx_list), 
            'gt_position': [""],
            'pred_position': pred_items[pred_idx_list[0]]['position'][0],  # get the first pred's position
            'norm_gt': "",
            'norm_pred': ''.join(norm_pred_lines[_] for _ in pred_idx_list),
            'gt_category_type': "",
            'pred_category_type': get_pred_category_type(pred_idx_list[0], pred_items), # get the first pred's category
            'gt_attribute': [{}],
            'edit': 1,
            'img_id': img_name
        })
        return match_list,None
    elif not norm_pred_lines: # not matched gt should be separated
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
        return match_list,None
    
    cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, norm_pred_lines)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    
    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx in row_ind:
            row_i = list(row_ind).index(gt_idx)
            pred_idx = int(col_ind[row_i])
            pred_line = pred_lines[pred_idx]
            norm_pred_line = norm_pred_lines[pred_idx]
            edit = cost_matrix[gt_idx][pred_idx]
        else:
            pred_idx = ""
            pred_line = ""
            norm_pred_line = ""
            edit = 1
        

        match_list.append({
            'gt_idx': [gt_idx],
            'gt': gt_lines[gt_idx],
            'norm_gt': norm_gt_lines[gt_idx],
            'gt_category_type': gt_cat_list[gt_idx],
            'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
            'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
            'pred_idx': [pred_idx],
            'pred': pred_line,
            'norm_pred': norm_pred_line,
            'pred_category_type': get_pred_category_type(pred_idx, pred_items) if pred_idx else "",
            'pred_position': pred_items[pred_idx]['position'][0] if pred_idx else "",
            'edit': edit,
            'img_id': img_name
        })
    
    pred_idx_list = [pred_idx for pred_idx in range(len(norm_pred_lines)) if pred_idx not in col_ind] # get not matched preds
    if pred_idx_list:
        if line_type in ['html_table', 'latex_table']:
            unmatch_table_pred = split_pred_table_to_text_items([pred_items[i] for i in pred_idx_list])
            return match_list, unmatch_table_pred  
        
        else:
            match_list.append({
                'gt_idx': [""],
                'gt': "",
                'pred_idx': pred_idx_list,
                'pred': ''.join(pred_lines[_] for _ in pred_idx_list), 
                'gt_position': [""],
                'pred_position': pred_items[pred_idx_list[0]]['position'][0],  # get the first pred's position
                'norm_gt': "",
                'norm_pred': ''.join(norm_pred_lines[_] for _ in pred_idx_list),
                'gt_category_type': "",
                'pred_category_type': get_pred_category_type(pred_idx_list[0], pred_items), # get the first pred's category
                'gt_attribute': [{}],
                'edit': 1,
                'img_id': img_name
            })
    return match_list,None


def match_gt2pred_no_split(gt_items, pred_items, line_type, img_name):
    # directly concatenate gt and pred by position
    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines = get_gt_pred_lines(gt_items, pred_items)
    gt_line_with_position = []
    for gt_line, norm_gt_line, gt_item in zip(gt_lines, norm_gt_lines, gt_items):
        gt_position = gt_item['order'] if gt_item.get('order') else gt_item.get('position', [""])[0]
        if gt_position:
            gt_line_with_position.append((gt_position, gt_line, norm_gt_line))
    sorted_gt_lines = sorted(gt_line_with_position, key=lambda x: x[0])
    gt = '\n\n'.join([_[1] for _ in sorted_gt_lines])
    norm_gt = '\n\n'.join([_[2] for _ in sorted_gt_lines])
    pred_line_with_position = [(pred_item['position'], pred_line, pred_norm_line) for pred_line, pred_norm_line, pred_item in zip(pred_lines, norm_pred_lines, pred_items)]
    sorted_pred_lines = sorted(pred_line_with_position, key=lambda x: x[0])
    pred = '\n\n'.join([_[1] for _ in sorted_pred_lines])
    norm_pred = '\n\n'.join([_[2] for _ in sorted_pred_lines])
    # edit = Levenshtein.distance(norm_gt, norm_pred)/max(len(norm_gt), len(norm_pred))
    if norm_gt or norm_pred:
        return [{
                'gt_idx': [0],
                'gt': gt,
                'norm_gt': norm_gt,
                'gt_category_type': "text_merge",
                'gt_position': [""],
                'gt_attribute': [{}],
                'pred_idx': [0],
                'pred': pred,
                'norm_pred': norm_pred,
                'pred_category_type': "text_merge",
                'pred_position': "",
                # 'edit': edit,
                'img_id': img_name
            }]
    else:
        return []
    
