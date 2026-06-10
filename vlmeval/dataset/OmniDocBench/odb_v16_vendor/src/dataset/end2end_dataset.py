import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.core.matching import (
    get_gt_pred_lines,
    match_gt2pred_no_split,
    match_gt2pred_quick,
    match_gt2pred_simple,
    match_gt2pred_timeout_safe,
    normalize_logic_brace_formula,
    sort_by_position_skip_inline,
    split_equation_arrays,
    split_gt_equation_arrays,
)
from src.core.preprocess import (
    _sanitize_formula_candidate,
    build_matrix_cdm_variants,
    formula_to_text,
    latex_timeout_context,
    md_tex_filter,
    normalized_table,
    normalized_text,
    read_md_file,
    strip_formula_delimiters,
    textblock2unicode,
    textblock_with_norm_formula,
)
from src.core.registry import DATASET_REGISTRY
from src.dataset.table_dataset import RecognitionTableDataset
from src.runtime.concurrency import resolve_match_workers
import Levenshtein
from tqdm import tqdm
from func_timeout import FunctionTimedOut, func_timeout
from loguru import logger
import sys
import traceback

@DATASET_REGISTRY.register("end2end_dataset")
class End2EndDataset():
    def __init__(self, cfg_task):
        gt_path = cfg_task['dataset']['ground_truth']['data_path']
        pred_folder = cfg_task['dataset']['prediction']['data_path']
        self.match_method = cfg_task['dataset'].get('match_method', 'quick_match')
        self.enable_timeout_match_fallback = cfg_task['dataset'].get('enable_timeout_match_fallback', True)
        self.match_timeout_sec = cfg_task['dataset'].get('match_timeout_sec', 420)
        self.quick_match_truncated_timeout_sec = cfg_task['dataset'].get('quick_match_truncated_timeout_sec', 300)
        self.timeout_fallback_short_line_max_chars = cfg_task['dataset'].get('timeout_fallback_short_line_max_chars')
        self.timeout_fallback_target_chunk_chars = cfg_task['dataset'].get('timeout_fallback_target_chunk_chars')
        self.timeout_fallback_max_chunk_chars = cfg_task['dataset'].get('timeout_fallback_max_chunk_chars')
        self.timeout_fallback_max_chunk_span = cfg_task['dataset'].get('timeout_fallback_max_chunk_span', cfg_task['dataset'].get('linear_fallback_max_merge_span', 10))
        self.timeout_fallback_order_window = cfg_task['dataset'].get('timeout_fallback_order_window')
        self.timeout_fallback_order_penalty = cfg_task['dataset'].get('timeout_fallback_order_penalty', 0.10)
        self.match_workers = resolve_match_workers(cfg_task.get('dataset'))
        self.slow_stage_log_sec = self._resolve_float_value(
            cfg_task['dataset'].get('slow_stage_log_sec', os.getenv('OMNIDOCBENCH_SLOW_STAGE_LOG_SEC', 60)),
            60.0,
        )
        self._match_debug_lock = Lock()
        self.match_debug_info = {
            'workers': self.match_workers,
            'page_count': 0,
            'quick_match_truncated_timeout_sec': self.quick_match_truncated_timeout_sec,
            'match_timeout_sec': self.match_timeout_sec,
            'text_match_fallback_pages': {
                'quick_match_timeout': set(),
                'page_timeout': set(),
            },
        }
        filtered_types = cfg_task['dataset'].get('filter')

        with open(gt_path, 'r') as f:
            gt_samples = json.load(f)

        filtered_gt_samples = []
        if filtered_types:
            for gt_sample in gt_samples:
                select_flag = True
                for k, v in filtered_types.items():
                    if gt_sample["page_info"]["page_attribute"][k] != v:
                        select_flag = False
                if select_flag:
                    filtered_gt_samples.append(gt_sample)
        else:
            filtered_gt_samples = gt_samples

        self.gt_pages_by_element = self._collect_gt_pages_by_element(filtered_gt_samples)
        self.match_debug_info['page_count'] = len(filtered_gt_samples)
        self.samples = self.get_matched_elements(filtered_gt_samples, pred_folder)
        self._finalize_match_debug_info()
     
        
    def __getitem__(self, cat_name, idx):
        return self.samples[cat_name][idx]
    
    def _resolve_float_value(self, value, default):
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            resolved = float(default)
        return max(0.0, resolved)

    def _log_slow_stage(self, img_name, stage_name, start_time):
        if stage_name != 'page_total':
            return
        if self.slow_stage_log_sec <= 0:
            return
        elapsed = time.monotonic() - start_time
        if elapsed >= self.slow_stage_log_sec:
            logger.warning(f"[match-stage-slow] {img_name}: stage={stage_name} elapsed={elapsed:.2f}s")
    
    def _record_match_fallback(self, img_name, reason):
        if not img_name or not reason:
            return
        fallback_pages = self.match_debug_info.get('text_match_fallback_pages', {})
        if reason not in fallback_pages:
            fallback_pages[reason] = set()
        with self._match_debug_lock:
            fallback_pages[reason].add(img_name)

    def _finalize_match_debug_info(self):
        fallback_pages = self.match_debug_info.get('text_match_fallback_pages', {})
        fallback_counts = {}
        for reason, pages in list(fallback_pages.items()):
            if isinstance(pages, set):
                pages = sorted(pages)
            else:
                pages = sorted(set(pages))
            fallback_pages[reason] = pages
            fallback_counts[reason] = len(pages)
        self.match_debug_info['text_match_fallback_counts'] = fallback_counts

    def _collect_gt_pages_by_element(self, gt_samples):
        gt_pages = {
            'display_formula': set(),
            'table': set(),
        }
        for sample in gt_samples or []:
            img_name = os.path.basename(sample.get("page_info", {}).get("image_path", ""))
            if not img_name:
                continue

            for item in sample.get('layout_dets', []):
                if item.get('ignore', False):
                    continue
                category_type = item.get('category_type')
                if category_type == 'equation_isolated':
                    gt_pages['display_formula'].add(img_name)
                elif category_type == 'table':
                    gt_pages['table'].add(img_name)

        return gt_pages


    # 匹配元素 处理文本截断问题，将截断的文本块合并，并将元素按类别存储在字典中
    def get_page_elements(self, selected_annos):
        
        saved_element_dict = defaultdict(list) #存储元素
        related_truncated = [] #存储需要合并的截断文本块列表
        truncated_all = {} #存储截断文本块信息
        for relation in selected_annos["extra"]["relation"]:   # Handle truncated text issues
            if relation["relation_type"] == 'truncated':
                source_anno_id = relation["source_anno_id"]
                target_anno_id = relation["target_anno_id"]
                truncated_all[source_anno_id] = ""
                truncated_all[target_anno_id] = ""

                merged_group = {source_anno_id, target_anno_id}
                remaining_groups = []
                for merge_group in related_truncated:
                    if merge_group & merged_group:
                        merged_group |= merge_group
                    else:
                        remaining_groups.append(merge_group)
                remaining_groups.append(merged_group)
                related_truncated = remaining_groups
        for item in selected_annos['layout_dets']:
            if item['anno_id'] not in truncated_all.keys():
                if item.get('ignore', False):
                    continue
                saved_element_dict[item["category_type"]].append(item)
            else:
                truncated_all[item['anno_id']] = item

        for merge_list in related_truncated:
            text_block_list = [truncated_all[key] for key in merge_list if truncated_all.get(key)]
            if not text_block_list or any(block.get('ignore', False) for block in text_block_list):
                continue
            sorted_block = sorted(text_block_list, key=lambda x: x['order'])
            text = ""
            for block in sorted_block:
                text += block['text']
            merged_block = {
                "category_type": sorted_block[0]["category_type"], # Directly use information from the first block
                "order": sorted_block[0]["order"],
                "anno_id": sorted_block[0]["anno_id"],   
                "text": text,
                "merge_list": sorted_block
            }
            saved_element_dict[sorted_block[0]["category_type"]].append(merged_block)
            # print('Merged truncated')

        return saved_element_dict
    
    # 根据类别列表 category_list 从 gt_page_elements 中提取元素，并将它们合并到一个列表中
    def get_page_elements_list(self, gt_page_elements, category_list):
        element_list = []
        for category_type in category_list:
            if gt_page_elements.get(category_type):
                element_list.extend(gt_page_elements[category_type])
        return element_list

    # 根据元素的 order 字段对元素列表进行排序，并返回排序后的元素列表。
    def get_sorted_text_list(self, selected_annos):
        # txt_type: text, latex, html
        text_list = []
        for item in selected_annos:
            if item.get('order'):
                order = item['order']
            else:
                order = 0
            text_list.append((order, item))
        sorted_text_list = sorted(text_list, key=lambda x: x[0])
        return [_[1] for _ in sorted_text_list]
    
    # 从元素列表 items 中过滤掉 gt_category_type 在 ignore_category_list 中的元素。
    def filtered_out_ignore(self, items, ignore_category_list):
        filted_items = []
        for item in items:
            if item['gt_category_type'] not in ignore_category_list:
                filted_items.append(item)
        return filted_items

    # 计算预测结果和地面真值的阅读顺序之间的编辑距离，并返回包含相关信息的字典。
    def get_order_paired(self, order_match_s, img_name):
        matched = [(item['gt_position'], item['pred_position']) for item in order_match_s if (item['gt_position'] != [""] and item['pred_position'] != "")]
        gt_idx_all = [item['gt_position'] for item in order_match_s if (item['gt_position'] != [""])]
        read_order_pred = [i[0] for i in sorted(matched, key=lambda x: x[1])]  # Sort by pred idx to get Pred ordered GT_idx
        read_order_gt = sum(gt_idx_all, []) # Convert to one-dimensional list
        read_order_gt = [x for x in read_order_gt if x]  # For truncated merges, some discarded classes may be merged in, remove them when calculating edit distance
        gt = sorted(read_order_gt) # Sort by all GT idx to get GT ordered GT_idx
        pred = sum(read_order_pred, [])
        pred = [x for x in pred if x]
        if len(pred) > 0 or len(gt) > 0:
            edit = Levenshtein.distance(gt, pred)/ max(len(pred), len(gt))
            return {
                'gt': gt,  
                'pred': pred,
                'img_id': img_name,
                'edit': edit
            }
        else:
            return {}  # If both GT and pred are empty for the page, return empty

    # 为公式匹配结果添加 img_id 信息。
    def formula_format(self, formula_matches, img_name):
        # formated_list = []
        for i, item in enumerate(formula_matches):
            item["img_id"] = img_name + '_' + str(i)
        return formula_matches

    def _is_formula_category(self, category_type):
        return category_type in ['equation_isolated', 'equation_inline']


    def _contains_cjk(self, text):
        return bool(re.search(r'[一-鿿]', str(text or '')))

    def _should_attach_formula_cdm_surrogate(self, match_item):
        pred_category_type = match_item.get('pred_category_type', '')
        if pred_category_type in ['equation_inline', 'equation_isolated', '']:
            return False

        gt_text = str(match_item.get('gt', '') or '')
        pred_text = ' '.join(str(match_item.get('pred', '') or '').split())
        if not pred_text or len(pred_text) > 240:
            return False

        if not (self._contains_cjk(gt_text) or self._contains_cjk(pred_text)):
            return False

        edit = match_item.get('edit', 1)
        try:
            edit = float(edit)
        except Exception:
            return False
        if edit > 0.35:
            return False

        formula_hints = ['=', '≈', '≠', '≤', '≥', '÷', '×', '+', '-', '/', '%', '^', '_', r'\frac', r'\div', r'\times', r'\sum', r'\int', r'\(', r'\)', r'\[', r'\]', '$']
        if any(token in pred_text for token in formula_hints):
            return True

        if edit <= 0.28 and len(pred_text) <= 80:
            return True
        return False

    def _attach_formula_cdm_surrogates(self, formula_matches):
        updated_matches = []
        for item in formula_matches:
            updated_item = dict(item)
            gt_cdm, pred_cdm_alt = build_matrix_cdm_variants(
                updated_item.get('gt', ''),
                updated_item.get('pred', ''),
            )
            if gt_cdm:
                updated_item['gt_cdm'] = gt_cdm
            if pred_cdm_alt:
                updated_item['pred_cdm_alt'] = pred_cdm_alt

            if self._should_attach_formula_cdm_surrogate(updated_item):
                raw_pred_cdm = str(updated_item.get('pred', '') or '').strip()
                alt_pred_cdm = _sanitize_formula_candidate(updated_item.get('pred', ''))
                if raw_pred_cdm:
                    updated_item['pred_cdm'] = raw_pred_cdm
                    if alt_pred_cdm and alt_pred_cdm != raw_pred_cdm and not updated_item.get('pred_cdm_alt'):
                        updated_item['pred_cdm_alt'] = alt_pred_cdm
                    updated_item['gt_cdm'] = updated_item.get('gt_cdm', updated_item.get('gt', ''))
            updated_matches.append(updated_item)
        return updated_matches

    def _normalized_edit_distance(self, gt_text, pred_text):
        if not gt_text and not pred_text:
            return 0
        if not gt_text or not pred_text:
            return 1
        return Levenshtein.distance(gt_text, pred_text) / max(len(gt_text), len(pred_text))

    def _is_text_merge_fallback_candidate(self, match_item):
        if match_item.get('gt_category_type') != 'text_block':
            return False
        if match_item.get('gt_idx') == [""]:
            return False

        norm_gt = str(match_item.get('norm_gt', '') or '')
        if len(norm_gt) < 400:
            return False

        try:
            edit = float(match_item.get('edit', 1))
        except Exception:
            return False
        return edit > 0.35

    def _build_text_pred_pool(self, pred_dataset_mix):
        text_pred_pool = []
        sorted_pred_pairs = sort_by_position_skip_inline(pred_dataset_mix)
        for sorted_idx, (_, pred_item) in enumerate(sorted_pred_pairs):
            if pred_item.get('category_type') != 'text_all':
                continue

            raw_text = str(pred_item.get('content', '') or '')
            norm_text = normalized_text(raw_text)
            if not norm_text:
                continue

            text_pred_pool.append({
                'sorted_idx': sorted_idx,
                'raw_text': raw_text,
                'norm_text': norm_text,
                'pred_item': pred_item,
                'pred_category_type': pred_item.get('fine_category_type') or pred_item.get('category_type') or 'text_block',
            })
        return text_pred_pool

    def _search_best_text_merge_span(self, match_item, text_pred_pool):
        norm_gt = str(match_item.get('norm_gt', '') or '')
        current_edit = float(match_item.get('edit', 1))
        max_span = 160

        best_candidate = None
        for start in range(len(text_pred_pool)):
            merged_raw_parts = []
            merged_norm_parts = []
            prev_sorted_idx = None

            for end in range(start, min(len(text_pred_pool), start + max_span)):
                candidate = text_pred_pool[end]
                if prev_sorted_idx is not None and candidate['sorted_idx'] != prev_sorted_idx + 1:
                    break

                merged_raw_parts.append(candidate['raw_text'])
                merged_norm_parts.append(candidate['norm_text'])
                prev_sorted_idx = candidate['sorted_idx']

                merged_norm = ''.join(merged_norm_parts)
                edit = self._normalized_edit_distance(norm_gt, merged_norm)
                if best_candidate is None or edit < best_candidate['edit']:
                    pred_indices = [text_pred_pool[idx]['sorted_idx'] for idx in range(start, end + 1)]
                    best_candidate = {
                        'edit': edit,
                        'pred_idx': pred_indices,
                        'pred': ''.join(merged_raw_parts),
                        'norm_pred': merged_norm,
                        'pred_position': text_pred_pool[start]['pred_item']['position'][0],
                        'pred_category_type': text_pred_pool[start]['pred_category_type'],
                    }

                if len(merged_norm) > len(norm_gt) * 1.35 and edit > current_edit:
                    break

        if not best_candidate:
            return None
        if best_candidate['edit'] >= current_edit - 0.15:
            return None
        if best_candidate['edit'] > 0.35:
            return None
        return best_candidate

    def _text_contains_formula_hint(self, text):
        text = str(text or '')
        if not text:
            return False
        if re.search(r'\$[^$]+\$|\\\([^)]*\\\)', text):
            return True
        if any(token in text for token in [r'\frac', r'\sum', r'\int', r'\sqrt', r'\left', r'\right']):
            return True
        if ('^' in text or '_' in text) and ('$' in text or '\\' in text):
            return True
        return False

    def _normalize_text_match_target(self, gt_category, gt_text, has_formula_hint=False):
        if has_formula_hint and not self._is_formula_category(gt_category):
            return normalized_text(textblock_with_norm_formula(str(gt_text or '')))
        return normalized_text(self._cross_category_text(gt_category, gt_text))

    def _build_local_text_search_pool(self, pred_dataset_mix):
        pred_pool = []
        for sorted_idx, (_, pred_item) in enumerate(sort_by_position_skip_inline(pred_dataset_mix)):
            category_type = pred_item.get('category_type')
            if category_type not in ['text_all', 'equation_isolated']:
                continue
            raw_text = str(pred_item.get('content', '') or '')
            rendered_text = self._cross_category_text(category_type, raw_text)
            pred_pool.append({
                'sorted_idx': sorted_idx,
                'category_type': category_type,
                'raw_text': raw_text,
                'rendered_text': rendered_text,
                'pred_item': pred_item,
            })
        return pred_pool

    def _match_item_pred_pool_indices(self, match_item, pred_pool):
        pred_indices = []
        for pred_idx in match_item.get('pred_idx', []) or []:
            if isinstance(pred_idx, np.integer):
                pred_idx = int(pred_idx)
            if isinstance(pred_idx, int) and 0 <= pred_idx < len(pred_pool):
                pred_indices.append(pred_idx)

        if pred_indices:
            return sorted(set(pred_indices))

        pred_position = match_item.get('pred_position')
        if pred_position in ([""], "", None):
            return []

        matched_indices = []
        for idx, pool_item in enumerate(pred_pool):
            position = pool_item['pred_item'].get('position', ["", ""])
            if not isinstance(position, list) or len(position) < 2:
                continue
            try:
                start = float(position[0])
                end = float(position[1])
                pred_pos = float(pred_position)
            except (TypeError, ValueError):
                continue
            if start <= pred_pos <= end:
                matched_indices.append(idx)
        return matched_indices

    def _should_try_local_text_span_fallback(self, match_item):
        if match_item.get('gt_idx') == [""]:
            return False

        gt_category = match_item.get('gt_category_type')
        if gt_category not in ['title', 'text_block']:
            return False

        try:
            edit = float(match_item.get('edit', 1))
        except Exception:
            return False
        if edit <= 0.45:
            return False

        if gt_category == 'title':
            return True

        gt_text = str(match_item.get('gt', '') or '')
        norm_gt = str(match_item.get('norm_gt', '') or '')
        return self._text_contains_formula_hint(gt_text) or len(norm_gt) >= 80

    def _search_best_local_text_span(self, match_item, pred_pool, blocked_indices):
        if not pred_pool:
            return None

        gt_text = str(match_item.get('gt', '') or '')
        gt_category = match_item.get('gt_category_type', '')
        has_formula_hint = self._text_contains_formula_hint(gt_text)
        gt_norm = self._normalize_text_match_target(gt_category, gt_text, has_formula_hint=has_formula_hint)
        if not gt_norm:
            return None

        try:
            current_edit = float(match_item.get('edit', 1))
        except Exception:
            current_edit = 1.0

        owned_indices = self._match_item_pred_pool_indices(match_item, pred_pool)
        if owned_indices:
            lower_bound = max(0, min(owned_indices) - 8)
            upper_bound = min(len(pred_pool) - 1, max(owned_indices) + 8)
        else:
            lower_bound = 0
            upper_bound = len(pred_pool) - 1

        allowed_categories = {'text_all'}
        if has_formula_hint:
            allowed_categories.add('equation_isolated')

        max_span = 8 if has_formula_hint else 4
        max_allowed_edit = 0.45 if has_formula_hint else 0.35
        best_candidate = None

        for start_idx in range(lower_bound, upper_bound + 1):
            start_item = pred_pool[start_idx]
            if start_item['category_type'] not in allowed_categories or start_idx in blocked_indices:
                continue

            raw_parts = []
            rendered_parts = []
            for end_idx in range(start_idx, min(upper_bound + 1, start_idx + max_span)):
                current_item = pred_pool[end_idx]
                if current_item['category_type'] not in allowed_categories:
                    break
                if end_idx in blocked_indices:
                    break

                raw_parts.append(current_item['raw_text'])
                rendered_parts.append(current_item['rendered_text'])
                norm_pred = normalized_text(''.join(rendered_parts))
                if not norm_pred:
                    continue

                edit = self._normalized_edit_distance(gt_norm, norm_pred)
                candidate = {
                    'pred_idx': list(range(start_idx, end_idx + 1)),
                    'pred': ''.join(raw_parts),
                    'pred_position': pred_pool[start_idx]['pred_item']['position'][0],
                    'pred_category_type': 'text_block',
                    'norm_pred': norm_pred,
                    'edit': edit,
                    'span_len': end_idx - start_idx + 1,
                    'distance_to_current': (
                        abs(start_idx - owned_indices[0]) if owned_indices else 0
                    ),
                }

                if best_candidate is None:
                    best_candidate = candidate
                    continue

                if edit + 1e-12 < best_candidate['edit']:
                    best_candidate = candidate
                    continue
                if edit > best_candidate['edit'] + 1e-12:
                    continue
                if candidate['span_len'] < best_candidate['span_len']:
                    best_candidate = candidate
                    continue
                if (
                    candidate['span_len'] == best_candidate['span_len']
                    and candidate['distance_to_current'] < best_candidate['distance_to_current']
                ):
                    best_candidate = candidate

        if not best_candidate:
            return None
        if best_candidate['edit'] >= current_edit - 0.2:
            return None
        if best_candidate['edit'] > max_allowed_edit:
            return None
        return best_candidate

    def _maybe_apply_local_text_span_fallback(self, match, pred_dataset_mix):
        if not match:
            return match

        pred_pool = self._build_local_text_search_pool(pred_dataset_mix)
        if not pred_pool:
            return match

        updated_matches = [dict(item) for item in match]
        replacement_candidates = []
        for match_idx, match_item in enumerate(updated_matches):
            if not self._should_try_local_text_span_fallback(match_item):
                continue

            blocked_indices = set()
            for other_idx, other_item in enumerate(updated_matches):
                if other_idx == match_idx or other_item.get('gt_idx') == [""]:
                    continue
                blocked_indices.update(self._match_item_pred_pool_indices(other_item, pred_pool))

            candidate = self._search_best_local_text_span(match_item, pred_pool, blocked_indices)
            if not candidate:
                continue

            replacement_candidates.append({
                'match_idx': match_idx,
                'gain': float(match_item.get('edit', 1)) - candidate['edit'],
                'candidate': candidate,
            })

        used_pred_indices = set()
        for replacement in sorted(replacement_candidates, key=lambda item: item['gain'], reverse=True):
            candidate = replacement['candidate']
            candidate_idx_set = set(candidate['pred_idx'])
            if candidate_idx_set & used_pred_indices:
                continue

            updated_matches[replacement['match_idx']].update({
                'pred_idx': candidate['pred_idx'],
                'pred': candidate['pred'],
                'pred_position': candidate['pred_position'],
                'pred_category_type': candidate['pred_category_type'],
                'norm_pred': candidate['norm_pred'],
                'edit': candidate['edit'],
            })
            used_pred_indices.update(candidate_idx_set)
            logger.info(
                f"[text-fallback] {updated_matches[replacement['match_idx']].get('img_id', '')}: "
                f"gt_idx={updated_matches[replacement['match_idx']].get('gt_idx')} "
                f"edit={float(replacement['gain']) + float(candidate['edit']):.4f}->{candidate['edit']:.4f} "
                f"pred_idx={candidate['pred_idx']}"
            )

        return updated_matches

    def _maybe_apply_text_merge_fallback(self, plain_text_match_s, pred_dataset_mix):
        if not plain_text_match_s:
            return plain_text_match_s

        text_pred_pool = self._build_text_pred_pool(pred_dataset_mix)
        if not text_pred_pool:
            return plain_text_match_s

        updated_matches = [dict(item) for item in plain_text_match_s]
        replacement_candidates = []

        for match_idx, match_item in enumerate(updated_matches):
            if not self._is_text_merge_fallback_candidate(match_item):
                continue

            best_candidate = self._search_best_text_merge_span(match_item, text_pred_pool)
            if not best_candidate:
                continue

            replacement_candidates.append({
                'match_idx': match_idx,
                'gain': float(match_item.get('edit', 1)) - best_candidate['edit'],
                'candidate': best_candidate,
            })

        used_pred_indices = set()
        for replacement in sorted(replacement_candidates, key=lambda x: x['gain'], reverse=True):
            candidate = replacement['candidate']
            candidate_idx_set = set(candidate['pred_idx'])
            if candidate_idx_set & used_pred_indices:
                continue

            used_pred_indices.update(candidate_idx_set)
            updated_matches[replacement['match_idx']].update(candidate)

        return updated_matches

    def _formula_match_page_stats(self, formula_matches, pred_count=None, extra_unmatched_pred_hits=0):
        rows = [item for item in formula_matches if item.get('gt_idx') != [""]]
        if not rows:
            unmatched_pred_hits = (pred_count or 0) + extra_unmatched_pred_hits
            total_slots = unmatched_pred_hits
            return {
                'matched': 0,
                'matched_pairs': 0,
                'avg_edit': None,
                'weighted_edit': None,
                'inline_hits': 0,
                'merged_hits': 0,
                'unmatched_hits': 0,
                'unmatched_pred_hits': unmatched_pred_hits,
                'page_cost': float(unmatched_pred_hits),
                'page_cost_norm': (float(unmatched_pred_hits) / total_slots) if total_slots else 0,
            }

        total_edit_num = 0
        total_upper_len = 0
        used_pred_indices = set()
        matched_pairs = 0
        total_row_cost = 0.0
        for item in rows:
            total_row_cost += float(item.get('edit', 1))
            pred_indices = item.get('pred_idx', [])
            if pred_indices in ([""], ""):
                continue
            matched_pairs += 1
            for pred_idx in pred_indices:
                if pred_idx == "":
                    continue
                used_pred_indices.add(pred_idx)
            norm_gt = str(item.get('norm_gt', '') or '')
            norm_pred = str(item.get('norm_pred', '') or '')
            upper_len = max(len(norm_gt), len(norm_pred))
            total_upper_len += upper_len
            if upper_len > 0:
                total_edit_num += Levenshtein.distance(norm_gt, norm_pred)

        unmatched_gt_hits = sum(1 for item in rows if item.get('pred_idx') in ([""], ""))
        unmatched_pred_hits = max(0, (pred_count or 0) - len(used_pred_indices)) + extra_unmatched_pred_hits
        total_slots = len(rows) + (pred_count or 0) + extra_unmatched_pred_hits
        page_cost = total_row_cost + unmatched_pred_hits

        return {
            'matched': len(rows),
            'matched_pairs': matched_pairs,
            'avg_edit': sum(float(item.get('edit', 1)) for item in rows) / len(rows),
            'weighted_edit': (total_edit_num / total_upper_len) if total_upper_len else 0,
            'inline_hits': sum(1 for item in rows if item.get('pred_category_type') == 'equation_inline'),
            'merged_hits': sum(1 for item in rows if item.get('pred_idx') not in ([""], "") and len(item.get('pred_idx', [])) >= 2),
            'unmatched_hits': unmatched_gt_hits,
            'unmatched_pred_hits': unmatched_pred_hits,
            'page_cost': page_cost,
            'page_cost_norm': (page_cost / total_slots) if total_slots else 0,
        }

    def _get_formula_position(self, item):
        if item.get('order') is not None:
            return item.get('order')
        return item.get('position', [""])[0]

    def _get_formula_range(self, item):
        position = item.get('position', [""])
        if isinstance(position, list) and len(position) >= 2:
            return position[0], position[1]
        start = position[0] if position else ""
        return start, start

    def _get_formula_pred_category_type(self, pred_item):
        pred_category_type = pred_item.get('fine_category_type') or pred_item.get('category_type') or 'equation_isolated'
        if pred_category_type in ['text_formula', 'image_formula', 'formula_rescue', 'table_formula_rescue']:
            return 'equation_isolated'
        return pred_category_type

    def _normalize_formula_match_categories(self, formula_matches):
        normalized_matches = []
        if not formula_matches:
            return normalized_matches
        for item in formula_matches:
            normalized_item = dict(item)
            if normalized_item.get('pred_category_type') in ['text_formula', 'image_formula', 'formula_rescue', 'table_formula_rescue']:
                normalized_item['pred_category_type'] = 'equation_isolated'
            normalized_matches.append(normalized_item)
        return normalized_matches

    def _build_formula_candidate_pool(self, pred_dataset):
        candidate_pool = []
        seen = set()

        def append_candidate(item, force_formula=False):
            candidate = dict(item)
            if force_formula:
                candidate['category_type'] = 'equation_isolated'
                candidate['fine_category_type'] = candidate.get('fine_category_type') or 'formula_rescue'
            pos_start, pos_end = self._get_formula_range(candidate)
            key = (pos_start, pos_end, candidate.get('content', ''), candidate.get('fine_category_type', ''), candidate.get('category_type', ''))
            if key in seen:
                return
            seen.add(key)
            candidate_pool.append(candidate)

        for item in pred_dataset.get('equation_isolated', []):
            append_candidate(item)
        for item in pred_dataset.get('formula_rescue', []):
            append_candidate(item, force_formula=True)
        return candidate_pool

    def _build_merged_formula_content(self, pred_parts):
        pieces = []
        for part in pred_parts:
            stripped = strip_formula_delimiters(str(part or '').strip())
            if stripped:
                pieces.append('{' + stripped + '}')
        if not pieces:
            return ''
        if len(pieces) == 1:
            return pieces[0][1:-1]
        return '\\begin{array}{l} ' + ' \\\\ '.join(pieces) + ' \\end{array}'

    def _build_formula_source_key(self, pred_item):
        pred_start, pred_end = self._get_formula_range(pred_item)
        return (
            pred_start,
            pred_end,
            pred_item.get('content', ''),
            pred_item.get('fine_category_type', ''),
            pred_item.get('category_type', ''),
        )

    def _annotate_formula_source_items(self, pred_formula):
        annotated_items = []
        for pred_item in pred_formula or []:
            annotated_item = dict(pred_item)
            annotated_item.setdefault('source_formula_key', self._build_formula_source_key(pred_item))
            annotated_items.append(annotated_item)
        return annotated_items

    def _normalize_formula_logic_brace_pred_items(self, pred_formula):
        normalized_items = []
        normalized_count = 0
        for pred_item in self._annotate_formula_source_items(pred_formula):
            normalized_item = dict(pred_item)
            if normalized_item.get('category_type') == 'equation_isolated':
                raw_content = str(normalized_item.get('content', ''))
                normalized_content = normalize_logic_brace_formula(raw_content)
                if normalized_content != raw_content:
                    normalized_item['content'] = normalized_content
                    normalized_count += 1
            normalized_items.append(normalized_item)
        if normalized_count <= 0:
            return None, 0
        return normalized_items, normalized_count

    def _prepare_formula_candidate_inputs(self, gt_formula, pred_formula, split_pred_formula=True):
        prepared_gt_formula = split_gt_equation_arrays(gt_formula)
        pred_formula = self._annotate_formula_source_items(pred_formula)
        ordered_pred_formula_raw = sorted(
            pred_formula,
            key=lambda item: (
                self._get_formula_range(item)[0],
                self._get_formula_pred_category_type(item) == 'equation_inline',
                self._get_formula_range(item)[1],
            ),
        )
        ordered_pred_formula = (
            split_equation_arrays(ordered_pred_formula_raw)
            if split_pred_formula else ordered_pred_formula_raw
        )
        gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines, gt_items, pred_items = get_gt_pred_lines(
            prepared_gt_formula,
            ordered_pred_formula,
            None,
        )
        return {
            'pred_formula': pred_formula,
            'ordered_pred_formula_raw': ordered_pred_formula_raw,
            'ordered_pred_formula': ordered_pred_formula,
            'gt_lines': gt_lines,
            'norm_gt_lines': norm_gt_lines,
            'gt_cat_list': gt_cat_list,
            'pred_lines': pred_lines,
            'norm_pred_lines': norm_pred_lines,
            'gt_items': gt_items,
            'pred_items': pred_items,
            'gt_count': len(gt_lines),
            'pred_count': len(pred_lines),
            'pred_raw_count': len(ordered_pred_formula_raw),
            'pred_was_split': len(ordered_pred_formula) > len(ordered_pred_formula_raw),
            'rescue_pred_count': sum(
                1
                for item in pred_formula
                if self._get_formula_pred_category_type(item) in ['text_formula', 'image_formula', 'formula_rescue', 'table_formula_rescue']
            ),
        }

    def _resolve_formula_candidate_max_merge(self, gt_count, pred_count):
        if pred_count <= 0:
            return 0
        if gt_count <= 1:
            return min(pred_count, 10)
        base_max_merge = 12 if abs(gt_count - pred_count) >= 2 else 10
        return min(pred_count, base_max_merge)

    def _enumerate_formula_merge_sequences(self, pred_count, start_idx, max_items, max_span, max_skips):
        if start_idx >= pred_count:
            return []

        max_end = min(pred_count, start_idx + max_span)
        sequences = []

        def dfs(sequence, skips_left):
            sequences.append(tuple(sequence))
            if len(sequence) >= max_items:
                return
            last_idx = sequence[-1]
            for next_idx in range(last_idx + 1, max_end):
                skipped_count = next_idx - last_idx - 1
                if skipped_count > skips_left:
                    continue
                dfs(sequence + [next_idx], skips_left - skipped_count)

        dfs([start_idx], max_skips)
        return sequences

    def _build_formula_match_entry(self, prepared, gt_idx, img_name, pred_seq=None):
        def _json_safe_index(value):
            if isinstance(value, np.integer):
                return int(value)
            return value

        gt_item = prepared['gt_items'][gt_idx]
        gt_position = [_json_safe_index(self._get_formula_position(gt_item))]
        if pred_seq:
            pred_seq = [_json_safe_index(idx) for idx in list(pred_seq)]
            pred_position = prepared['pred_items'][pred_seq[0]]['position'][0]
            pred_parts = [prepared['pred_lines'][idx] for idx in pred_seq]
            pred = self._build_merged_formula_content(pred_parts) if len(pred_seq) >= 2 else pred_parts[0]
            norm_pred = ' '.join(prepared['norm_pred_lines'][idx] for idx in pred_seq).strip()
            pred_category_type = 'equation_isolated' if len(pred_seq) >= 2 else self._get_formula_pred_category_type(prepared['pred_items'][pred_seq[0]])
            edit = self._normalized_edit_distance(prepared['norm_gt_lines'][gt_idx], norm_pred)
            pred_idx = pred_seq
        else:
            pred_position = ''
            pred = ''
            norm_pred = ''
            pred_category_type = ''
            edit = 1
            pred_idx = ['']

        return {
            'gt_idx': [_json_safe_index(gt_idx)],
            'gt': prepared['gt_lines'][gt_idx],
            'pred_idx': pred_idx,
            'pred': pred,
            'gt_position': gt_position,
            'pred_position': _json_safe_index(pred_position),
            'norm_gt': prepared['norm_gt_lines'][gt_idx],
            'norm_pred': norm_pred,
            'gt_category_type': prepared['gt_cat_list'][gt_idx],
            'pred_category_type': pred_category_type,
            'gt_attribute': [gt_item.get('attribute', {})],
            'edit': edit,
            'img_id': img_name,
        }

    def _build_display_formula_merge_candidate(self, prepared, img_name, allow_gaps=False):
        if prepared['gt_count'] != 1 or prepared['pred_count'] < 2:
            return None

        gt_norm = prepared['norm_gt_lines'][0]
        best_seq = None
        best_edit = 1.0
        max_merge = self._resolve_formula_candidate_max_merge(prepared['gt_count'], prepared['pred_count'])
        max_span = min(prepared['pred_count'], max_merge + 2) if allow_gaps else max_merge
        max_skips = 1 if (allow_gaps and prepared['pred_count'] >= 4) else 0
        for start in range(prepared['pred_count']):
            for seq in self._enumerate_formula_merge_sequences(prepared['pred_count'], start, max_merge, max_span, max_skips):
                if len(seq) < 2:
                    continue
                merged_norm = ' '.join(prepared['norm_pred_lines'][idx] for idx in seq).strip()
                if not merged_norm:
                    continue
                edit = self._normalized_edit_distance(gt_norm, merged_norm)
                if edit < best_edit:
                    best_edit = edit
                    best_seq = seq

        if best_seq is None:
            return None
        return [self._build_formula_match_entry(prepared, 0, img_name, best_seq)]

    def _can_merge_formula_sequence(self, prepared, seq):
        if len(seq) < 2:
            return True

        source_keys = {
            prepared['pred_items'][idx].get('source_formula_key')
            for idx in seq
        }
        if len(source_keys) == 1:
            return True

        if seq[-1] - seq[0] + 1 != len(seq):
            return False

        pred_types = [
            self._get_formula_pred_category_type(prepared['pred_items'][idx])
            for idx in seq
        ]
        return all(pred_type == 'equation_isolated' for pred_type in pred_types)

    def _build_display_formula_dp_candidate(self, prepared, img_name, allow_gaps=False):
        gt_count = prepared['gt_count']
        pred_count = prepared['pred_count']
        if gt_count <= 0 or pred_count <= 0:
            return None

        max_merge = self._resolve_formula_candidate_max_merge(gt_count, pred_count)
        max_span = min(pred_count, max_merge + 2) if allow_gaps else max_merge
        max_skips = 0
        if allow_gaps:
            if abs(gt_count - pred_count) >= 1 or prepared.get('rescue_pred_count', 0) > 0 or pred_count >= 4:
                max_skips = 1
            if abs(gt_count - pred_count) >= 3:
                max_skips = 2

        sequences_by_start = {}
        for start in range(pred_count):
            sequence_entries = []
            for seq in self._enumerate_formula_merge_sequences(pred_count, start, max_merge, max_span, max_skips):
                merged_pred_parts = [prepared['pred_lines'][idx] for idx in seq]
                sequence_entries.append((
                    seq,
                    {
                        'pred': self._build_merged_formula_content(merged_pred_parts) if len(seq) >= 2 else merged_pred_parts[0],
                        'norm_pred': ' '.join(prepared['norm_pred_lines'][idx] for idx in seq).strip(),
                        'pred_category_type': 'equation_isolated' if len(seq) >= 2 else self._get_formula_pred_category_type(prepared['pred_items'][seq[0]]),
                    }
                ))
            sequences_by_start[start] = sequence_entries

        inf = (10 ** 9, 10 ** 9, 10 ** 9, 10 ** 9)
        dp = [[inf for _ in range(pred_count + 1)] for _ in range(gt_count + 1)]
        parent = [[None for _ in range(pred_count + 1)] for _ in range(gt_count + 1)]
        dp[0][0] = (0.0, 0, 0, 0)

        def is_better_state(new_state, old_state):
            if old_state == inf:
                return True
            if new_state[0] + 1e-12 < old_state[0]:
                return True
            if new_state[0] > old_state[0] + 1e-12:
                return False

            new_upper = new_state[2]
            old_upper = old_state[2]
            if new_upper > 0 and old_upper > 0:
                left = new_state[1] * old_upper
                right = old_state[1] * new_upper
                if left < right:
                    return True
                if left > right:
                    return False
            elif new_upper > 0:
                return True
            elif old_upper > 0:
                return False

            if new_state[3] < old_state[3]:
                return True
            return False

        for gt_idx in range(gt_count + 1):
            for pred_idx in range(pred_count + 1):
                current_state = dp[gt_idx][pred_idx]
                if current_state == inf:
                    continue

                if pred_idx < pred_count:
                    next_state = (current_state[0] + 1.0, current_state[1], current_state[2], current_state[3])
                    if is_better_state(next_state, dp[gt_idx][pred_idx + 1]):
                        dp[gt_idx][pred_idx + 1] = next_state
                        parent[gt_idx][pred_idx + 1] = (gt_idx, pred_idx, 'skip_pred', None)

                if gt_idx >= gt_count:
                    continue

                next_state = (current_state[0] + 1.0, current_state[1], current_state[2], current_state[3])
                if is_better_state(next_state, dp[gt_idx + 1][pred_idx]):
                    dp[gt_idx + 1][pred_idx] = next_state
                    parent[gt_idx + 1][pred_idx] = (gt_idx, pred_idx, 'skip_gt', None)

                if pred_idx >= pred_count:
                    continue
                for seq, segment in sequences_by_start[pred_idx]:
                    if not self._can_merge_formula_sequence(prepared, seq):
                        continue
                    if not segment['norm_pred']:
                        continue
                    end_idx = seq[-1]
                    norm_gt = prepared['norm_gt_lines'][gt_idx]
                    norm_pred = segment['norm_pred']
                    edit = self._normalized_edit_distance(norm_gt, norm_pred)
                    upper_len = max(len(norm_gt), len(norm_pred))
                    char_edit_num = Levenshtein.distance(norm_gt, norm_pred) if upper_len > 0 else 0
                    inline_hits = current_state[3] + (1 if segment['pred_category_type'] == 'equation_inline' else 0)
                    next_state = (
                        current_state[0] + edit,
                        current_state[1] + char_edit_num,
                        current_state[2] + upper_len,
                        inline_hits,
                    )
                    if is_better_state(next_state, dp[gt_idx + 1][end_idx + 1]):
                        dp[gt_idx + 1][end_idx + 1] = next_state
                        parent[gt_idx + 1][end_idx + 1] = (
                            gt_idx,
                            pred_idx,
                            'match',
                            {
                                'seq': list(seq),
                            },
                        )

        if parent[gt_count][pred_count] is None:
            return None

        result = []
        gt_idx = gt_count
        pred_idx = pred_count
        while gt_idx > 0 or pred_idx > 0:
            previous = parent[gt_idx][pred_idx]
            if previous is None:
                return None
            prev_gt_idx, prev_pred_idx, action, payload = previous
            if action == 'skip_gt':
                result.append(self._build_formula_match_entry(prepared, prev_gt_idx, img_name, None))
            elif action == 'match':
                result.append(self._build_formula_match_entry(prepared, prev_gt_idx, img_name, payload['seq']))
            gt_idx, pred_idx = prev_gt_idx, prev_pred_idx

        result.reverse()
        if len(result) != gt_count:
            return None
        return result

    def _formula_candidate_total_key(self, candidate):
        stats = candidate.get('stats', {})

        def _safe_float(value, default=float('inf')):
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        total_edit = sum(
            float(item.get('edit', 1))
            for item in candidate.get('rows', [])
            if item.get('gt_idx') != [""]
        )

        used_source_keys = set()
        pred_items = candidate.get('pred_items') or []
        for row in candidate.get('rows', []):
            pred_idx = row.get('pred_idx', [])
            if pred_idx in ([""], ""):
                continue
            for pred_item_idx in pred_idx:
                if pred_item_idx in ("", None):
                    continue
                try:
                    pred_item = pred_items[int(pred_item_idx)]
                except (TypeError, ValueError, IndexError):
                    continue
                source_key = pred_item.get('source_formula_key')
                if source_key is not None:
                    used_source_keys.add(source_key)

        raw_pred_count = int(candidate.get('raw_pred_count', len(pred_items)))
        raw_unmatched_pred = max(0, raw_pred_count - len(used_source_keys))
        return (
            total_edit,
            _safe_float(stats.get('weighted_edit')),
            raw_unmatched_pred,
            _safe_float(stats.get('avg_edit')),
            -int(candidate.get('logic_brace_normalized_count', 0)),
            int(stats.get('unmatched_hits', 0)),
        )

    def _is_formula_total_cost_better(self, candidate, best_candidate):
        candidate_key = self._formula_candidate_total_key(candidate)
        best_key = self._formula_candidate_total_key(best_candidate)

        for candidate_value, best_value in zip(candidate_key, best_key):
            if isinstance(candidate_value, float) or isinstance(best_value, float):
                if candidate_value + 1e-12 < best_value:
                    return True
                if candidate_value > best_value + 1e-12:
                    return False
            else:
                if candidate_value < best_value:
                    return True
                if candidate_value > best_value:
                    return False
        return False

    def _build_formula_quick_candidate(self, gt_formula, pred_formula, img_name):
        formula_matches = match_gt2pred_quick(
            gt_formula,
            pred_formula,
            'formula',
            img_name,
            truncated_timeout_sec=self.quick_match_truncated_timeout_sec,
            fallback_short_line_max_chars=self.timeout_fallback_short_line_max_chars,
            fallback_target_chunk_chars=self.timeout_fallback_target_chunk_chars,
            fallback_max_chunk_chars=self.timeout_fallback_max_chunk_chars,
            fallback_max_chunk_span=self.timeout_fallback_max_chunk_span,
            fallback_order_window=self.timeout_fallback_order_window,
            fallback_order_penalty=self.timeout_fallback_order_penalty,
            split_pred_formula=False,
        )
        formula_matches = [dict(item) for item in formula_matches if item.get('gt_idx') != [""]]
        return self._normalize_formula_match_categories(formula_matches)

    def _wrap_formula_candidate_content(self, content):
        stripped = str(content or '').strip()
        if not stripped:
            return ''
        if (
            (stripped.startswith('$$') and stripped.endswith('$$'))
            or (stripped.startswith('\\[') and stripped.endswith('\\]'))
            or (stripped.startswith('\\(') and stripped.endswith('\\)'))
        ):
            return stripped
        return f'\\[{stripped}\\]'

    def _enumerate_formula_line_partitions(self, split_items):
        item_count = len(split_items or [])
        if item_count <= 0:
            return []
        if item_count == 1:
            return [[split_items]]

        partitions = []
        for cut_mask in range(1 << (item_count - 1)):
            start_idx = 0
            groups = []
            for boundary_idx in range(item_count - 1):
                if cut_mask & (1 << boundary_idx):
                    groups.append(split_items[start_idx:boundary_idx + 1])
                    start_idx = boundary_idx + 1
            groups.append(split_items[start_idx:item_count])
            partitions.append(groups)
        return partitions

    def _build_formula_partition_segment_item(self, segment_items):
        if len(segment_items) == 1:
            return dict(segment_items[0])

        merged_item = dict(segment_items[0])
        merged_item['content'] = self._wrap_formula_candidate_content(
            self._build_merged_formula_content([item.get('content', '') for item in segment_items])
        )
        merged_item['position'] = [
            segment_items[0].get('position', ["", ""])[0],
            segment_items[-1].get('position', ["", ""])[1],
        ]
        merged_item['fine_category_type'] = 'equation_isolated'
        return merged_item

    def _build_formula_partitioned_pred_candidates(self, pred_formula, max_candidates=128, max_partition_lines=8):
        ordered_pred_formula_raw = sorted(
            self._annotate_formula_source_items(pred_formula),
            key=lambda item: (
                self._get_formula_range(item)[0],
                self._get_formula_pred_category_type(item) == 'equation_inline',
                self._get_formula_range(item)[1],
            ),
        )
        option_sets = []
        total_candidates = 1
        has_splittable_formula = False

        for pred_item in ordered_pred_formula_raw:
            split_items = split_equation_arrays([pred_item])
            if len(split_items) <= 1 or len(split_items) > max_partition_lines:
                option_sets.append([[dict(pred_item)]])
                continue

            has_splittable_formula = True
            partitions = self._enumerate_formula_line_partitions(split_items)
            partition_options = [
                [self._build_formula_partition_segment_item(segment_items) for segment_items in partition]
                for partition in partitions
            ]
            total_candidates *= max(1, len(partition_options))
            if total_candidates > max_candidates:
                return []
            option_sets.append(partition_options)

        if not has_splittable_formula:
            return []

        candidates = []

        def backtrack(option_idx, current_items):
            if option_idx >= len(option_sets):
                candidates.append([dict(item) for item in current_items])
                return

            for option in option_sets[option_idx]:
                current_items.extend(option)
                backtrack(option_idx + 1, current_items)
                del current_items[-len(option):]

        backtrack(0, [])
        return candidates

    def _build_display_formula_hungarian_candidate(self, prepared, img_name):
        gt_count = prepared['gt_count']
        pred_count = prepared['pred_count']
        if gt_count <= 0:
            return []

        if pred_count <= 0:
            return [
                self._build_formula_match_entry(prepared, gt_idx, img_name, None)
                for gt_idx in range(gt_count)
            ]

        total_size = gt_count + pred_count
        cost_matrix = np.ones((total_size, total_size), dtype=float)
        cost_matrix[gt_count:, pred_count:] = 0.0

        for gt_idx in range(gt_count):
            norm_gt = prepared['norm_gt_lines'][gt_idx]
            for pred_idx in range(pred_count):
                norm_pred = prepared['norm_pred_lines'][pred_idx]
                cost_matrix[gt_idx, pred_idx] = self._normalized_edit_distance(norm_gt, norm_pred)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment = {row_idx: col_idx for row_idx, col_idx in zip(row_ind, col_ind)}

        result = []
        for gt_idx in range(gt_count):
            pred_idx = assignment.get(gt_idx)
            if pred_idx is not None and pred_idx < pred_count:
                result.append(self._build_formula_match_entry(prepared, gt_idx, img_name, [pred_idx]))
            else:
                result.append(self._build_formula_match_entry(prepared, gt_idx, img_name, None))
        return result

    def _build_formula_split_hungarian_candidates(self, gt_formula, pred_formula, img_name):
        pred_candidates = self._build_formula_partitioned_pred_candidates(pred_formula)
        if not pred_candidates:
            return []

        candidates = []
        seen_keys = set()
        for candidate_pred_formula in pred_candidates:
            prepared = self._prepare_formula_candidate_inputs(
                gt_formula,
                candidate_pred_formula,
                split_pred_formula=False,
            )
            candidate_rows = self._build_display_formula_hungarian_candidate(prepared, img_name)
            if not candidate_rows:
                continue

            candidate_key = tuple(
                (
                    tuple(row.get('gt_position') or []),
                    tuple(row.get('pred_idx') or []),
                    round(float(row.get('edit', 1)), 6),
                )
                for row in candidate_rows
            )
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)

            candidates.append({
                'name': 'split_hungarian',
                'rows': self._normalize_formula_match_categories(candidate_rows),
                'pred_count': len(prepared['ordered_pred_formula_raw']),
                'pred_items': prepared['ordered_pred_formula_raw'],
                'raw_pred_count': len(self._annotate_formula_source_items(pred_formula)),
            })
        return candidates

    def _build_formula_split_dp_candidates(self, gt_formula, pred_formula, img_name):
        prepared = self._prepare_formula_candidate_inputs(
            gt_formula,
            pred_formula,
            split_pred_formula=True,
        )
        if not prepared.get('pred_was_split'):
            return []

        candidate_specs = [
            ('split_dp', False),
            ('split_dp_allow_gaps', True),
        ]
        candidates = []
        for candidate_name, allow_gaps in candidate_specs:
            candidate_rows = self._build_display_formula_dp_candidate(
                prepared,
                img_name,
                allow_gaps=allow_gaps,
            )
            if not candidate_rows:
                continue
            candidates.append({
                'name': candidate_name,
                'rows': self._normalize_formula_match_categories(candidate_rows),
                'pred_count': len(prepared['ordered_pred_formula']),
                'pred_items': prepared['ordered_pred_formula'],
                'raw_pred_count': len(prepared['ordered_pred_formula_raw']),
            })
        return candidates

    def _build_logic_brace_normalized_formula_candidates(self, gt_formula, pred_formula, img_name):
        normalized_pred_formula, normalized_count = self._normalize_formula_logic_brace_pred_items(pred_formula)
        if not normalized_pred_formula:
            return []

        prepared_unsplit = self._prepare_formula_candidate_inputs(
            gt_formula,
            normalized_pred_formula,
            split_pred_formula=False,
        )
        candidates = []

        quick_rows = self._build_formula_quick_candidate(gt_formula, normalized_pred_formula, img_name)
        if self._is_complete_formula_candidate_rows(quick_rows, prepared_unsplit['gt_count']):
            candidates.append({
                'name': 'logic_norm_quick_unsplit',
                'rows': quick_rows,
                'pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'pred_items': prepared_unsplit['ordered_pred_formula_raw'],
                'raw_pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'logic_brace_normalized_count': normalized_count,
            })

        hungarian_rows = self._build_display_formula_hungarian_candidate(prepared_unsplit, img_name)
        if self._is_complete_formula_candidate_rows(hungarian_rows, prepared_unsplit['gt_count']):
            candidates.append({
                'name': 'logic_norm_hungarian_unsplit',
                'rows': self._normalize_formula_match_categories(hungarian_rows),
                'pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'pred_items': prepared_unsplit['ordered_pred_formula_raw'],
                'raw_pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'logic_brace_normalized_count': normalized_count,
            })

        for candidate_name, allow_gaps in [('logic_norm_dp_unsplit', False), ('logic_norm_dp_unsplit_allow_gaps', True)]:
            dp_rows = self._build_display_formula_dp_candidate(
                prepared_unsplit,
                img_name,
                allow_gaps=allow_gaps,
            )
            if not self._is_complete_formula_candidate_rows(dp_rows, prepared_unsplit['gt_count']):
                continue
            candidates.append({
                'name': candidate_name,
                'rows': self._normalize_formula_match_categories(dp_rows),
                'pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'pred_items': prepared_unsplit['ordered_pred_formula_raw'],
                'raw_pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'logic_brace_normalized_count': normalized_count,
            })

        for split_candidate in self._build_formula_split_hungarian_candidates(gt_formula, normalized_pred_formula, img_name):
            candidate_copy = dict(split_candidate)
            candidate_copy['name'] = 'logic_norm_' + candidate_copy['name']
            candidate_copy['logic_brace_normalized_count'] = normalized_count
            candidates.append(candidate_copy)

        for split_candidate in self._build_formula_split_dp_candidates(gt_formula, normalized_pred_formula, img_name):
            candidate_copy = dict(split_candidate)
            candidate_copy['name'] = 'logic_norm_' + candidate_copy['name']
            candidate_copy['logic_brace_normalized_count'] = normalized_count
            candidates.append(candidate_copy)

        return candidates

    def _is_complete_formula_candidate_rows(self, candidate_rows, expected_gt_count):
        if expected_gt_count <= 0:
            return True
        if not candidate_rows or len(candidate_rows) != expected_gt_count:
            return False

        allowed_pred_types = {'', 'equation_inline', 'equation_isolated'}
        seen_gt_idx = set()
        for row in candidate_rows:
            gt_idx = row.get('gt_idx', [])
            if not isinstance(gt_idx, list) or len(gt_idx) != 1:
                return False
            gt_idx_value = gt_idx[0]
            if gt_idx_value in ("", None):
                return False
            if gt_idx_value in seen_gt_idx:
                return False
            seen_gt_idx.add(gt_idx_value)
            if row.get('pred_category_type', '') not in allowed_pred_types:
                return False
        return len(seen_gt_idx) == expected_gt_count

    def _match_display_formula_candidates(self, gt_formula, pred_formula, current_formula_matches, img_name):
        prepared_unsplit = self._prepare_formula_candidate_inputs(
            gt_formula,
            pred_formula,
            split_pred_formula=False,
        )
        expected_gt_count = prepared_unsplit['gt_count']
        candidates = []

        current_rows = self._normalize_formula_match_categories(
            [dict(item) for item in current_formula_matches if item.get('gt_idx') != [""]]
        )
        if self._is_complete_formula_candidate_rows(current_rows, expected_gt_count):
            candidates.append({
                'name': 'current_match',
                'rows': current_rows,
                'pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'pred_items': prepared_unsplit['ordered_pred_formula_raw'],
                'raw_pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
            })

        quick_rows = self._build_formula_quick_candidate(gt_formula, pred_formula, img_name)
        if self._is_complete_formula_candidate_rows(quick_rows, expected_gt_count):
            candidates.append({
                'name': 'quick_unsplit',
                'rows': quick_rows,
                'pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'pred_items': prepared_unsplit['ordered_pred_formula_raw'],
                'raw_pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
            })

        hungarian_rows = self._build_display_formula_hungarian_candidate(prepared_unsplit, img_name)
        if self._is_complete_formula_candidate_rows(hungarian_rows, expected_gt_count):
            candidates.append({
                'name': 'hungarian_unsplit',
                'rows': self._normalize_formula_match_categories(hungarian_rows),
                'pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'pred_items': prepared_unsplit['ordered_pred_formula_raw'],
                'raw_pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
            })

        for candidate_name, allow_gaps in [('dp_unsplit', False), ('dp_unsplit_allow_gaps', True)]:
            dp_rows = self._build_display_formula_dp_candidate(
                prepared_unsplit,
                img_name,
                allow_gaps=allow_gaps,
            )
            if not self._is_complete_formula_candidate_rows(dp_rows, expected_gt_count):
                continue
            candidates.append({
                'name': candidate_name,
                'rows': self._normalize_formula_match_categories(dp_rows),
                'pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
                'pred_items': prepared_unsplit['ordered_pred_formula_raw'],
                'raw_pred_count': len(prepared_unsplit['ordered_pred_formula_raw']),
            })

        candidates.extend(self._build_formula_split_hungarian_candidates(gt_formula, pred_formula, img_name))
        candidates.extend(self._build_formula_split_dp_candidates(gt_formula, pred_formula, img_name))
        candidates.extend(self._build_logic_brace_normalized_formula_candidates(gt_formula, pred_formula, img_name))

        best_candidate = None
        for candidate in candidates:
            candidate_stats = self._formula_match_page_stats(
                candidate['rows'],
                pred_count=candidate['pred_count'],
            )
            candidate['stats'] = candidate_stats
            if best_candidate is None or self._is_formula_total_cost_better(candidate, best_candidate):
                best_candidate = candidate

        if best_candidate and best_candidate['name'] != 'current_match':
            logger.info(
                f"[formula-match-candidate] {img_name}: choose={best_candidate['name']} "
                f"page_cost={best_candidate['stats']['page_cost']:.4f}"
            )

        return list(best_candidate['rows']) if best_candidate else []

    def _should_try_formula_fallback(self, prepared, current_stats):
        if current_stats['matched'] <= 0 or current_stats['avg_edit'] is None:
            return False
        gt_count = prepared['gt_count']
        pred_count = prepared['pred_count']
        if gt_count <= 0 or pred_count <= 0:
            return False
        count_gap = abs(gt_count - pred_count)
        if gt_count == 1 and pred_count >= 2:
            return True
        if prepared.get('rescue_pred_count', 0) > 0 and current_stats['avg_edit'] >= 0.2:
            return True
        if current_stats['inline_hits'] > 0:
            return True
        if count_gap >= 2:
            return True
        if count_gap >= 1 and current_stats['avg_edit'] >= 0.45 and max(gt_count, pred_count) >= 3:
            return True
        if current_stats['avg_edit'] >= 0.6 and gt_count >= 2:
            return True
        return False

    def _formula_match_position_penalty(self, formula_matches):
        matched_rows = [
            item for item in (formula_matches or [])
            if item.get('gt_idx') != [""] and item.get('pred_idx') not in ([""], "")
        ]
        if len(matched_rows) <= 1:
            return 0.0

        def _safe_order(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float('inf')

        gt_sorted = sorted(
            range(len(matched_rows)),
            key=lambda idx: _safe_order((matched_rows[idx].get('gt_position') or [float('inf')])[0]),
        )
        pred_sorted = sorted(
            range(len(matched_rows)),
            key=lambda idx: _safe_order(matched_rows[idx].get('pred_position')),
        )
        gt_rank = {row_idx: rank for rank, row_idx in enumerate(gt_sorted)}
        pred_rank = {row_idx: rank for rank, row_idx in enumerate(pred_sorted)}
        denom = max(1, len(matched_rows) - 1)
        return sum(abs(gt_rank[idx] - pred_rank[idx]) / denom for idx in range(len(matched_rows))) / len(matched_rows)

    def _formula_candidate_visual_key(self, stats, position_penalty):
        def _as_score(value, default=1.0):
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        matched = int(stats.get('matched', 0))
        unmatched_pred = int(stats.get('unmatched_pred_hits', 0))
        unmatched_gt = int(stats.get('unmatched_hits', 0))
        total_slots = max(1, matched + unmatched_pred)
        unmatched_pred_ratio = unmatched_pred / total_slots
        weighted = _as_score(stats.get('weighted_edit'))
        avg = _as_score(stats.get('avg_edit'))
        visual_weighted = weighted + 0.03 * unmatched_pred_ratio + 0.05 * position_penalty
        visual_avg = avg + 0.03 * unmatched_pred_ratio + 0.05 * position_penalty
        return (
            visual_weighted,
            visual_avg,
            unmatched_gt,
            unmatched_pred,
            position_penalty,
        )

    def _is_better_formula_candidate(self, candidate_stats, best_stats, candidate_position_penalty, best_position_penalty):
        candidate_unmatched_gt = int(candidate_stats.get('unmatched_hits', 0))
        best_unmatched_gt = int(best_stats.get('unmatched_hits', 0))
        if candidate_unmatched_gt > best_unmatched_gt:
            return False

        def _safe_float(value):
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        candidate_weighted = _safe_float(candidate_stats.get('weighted_edit'))
        best_weighted = _safe_float(best_stats.get('weighted_edit'))
        candidate_avg = _safe_float(candidate_stats.get('avg_edit'))
        best_avg = _safe_float(best_stats.get('avg_edit'))
        if (
            candidate_weighted is not None and best_weighted is not None
            and candidate_avg is not None and best_avg is not None
            and candidate_weighted > best_weighted + 1e-12
            and candidate_avg > best_avg + 1e-12
        ):
            return False

        candidate_key = self._formula_candidate_visual_key(candidate_stats, candidate_position_penalty)
        best_key = self._formula_candidate_visual_key(best_stats, best_position_penalty)
        return candidate_key < best_key

    def _formula_order_key(self, value):
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return value

    def _formula_sort_key(self, value):
        normalized_value = self._formula_order_key(value)
        if isinstance(normalized_value, (int, float)):
            return (0, float(normalized_value))
        return (1, str(normalized_value))

    def _formula_match_uses_pred_item(self, match_item, pred_item):
        pred_position = match_item.get('pred_position')
        if pred_position in ([""], "", None):
            return False

        pred_start, pred_end = self._get_formula_range(pred_item)
        try:
            pred_position = float(pred_position)
            pred_start = float(pred_start)
            pred_end = float(pred_end)
        except (TypeError, ValueError):
            return pred_position == pred_start
        return pred_start <= pred_position <= pred_end

    def _formula_pred_item_signature(self, pred_item):
        pred_start, pred_end = self._get_formula_range(pred_item)
        return (
            pred_start,
            pred_end,
            pred_item.get('content', ''),
            pred_item.get('fine_category_type', ''),
            pred_item.get('category_type', ''),
        )

    def _formula_row_pred_item_indices(self, match_item, pred_items):
        if match_item.get('pred_position') in ([""], "", None):
            return []
        return [
            idx for idx, pred_item in enumerate(pred_items)
            if self._formula_match_uses_pred_item(match_item, pred_item)
        ]

    def _expand_formula_replacement_block(self, current_rows, linked_indices, ordered_pred_formula_raw, target_pred_idx, split_line_count):
        block_start = min(linked_indices)
        block_end = max(linked_indices)
        raw_pred_indices = {target_pred_idx}
        for row_idx in range(block_start, block_end + 1):
            raw_pred_indices.update(self._formula_row_pred_item_indices(current_rows[row_idx], ordered_pred_formula_raw))

        max_block_rows = max(block_end - block_start + 1, split_line_count + 2)
        while block_end - block_start + 1 < max_block_rows:
            expanded = False
            min_raw_pred_idx = min(raw_pred_indices) if raw_pred_indices else target_pred_idx
            max_raw_pred_idx = max(raw_pred_indices) if raw_pred_indices else target_pred_idx

            if block_start > 0:
                left_indices = self._formula_row_pred_item_indices(current_rows[block_start - 1], ordered_pred_formula_raw)
                if not left_indices or max(left_indices) >= min_raw_pred_idx - 1:
                    block_start -= 1
                    raw_pred_indices.update(left_indices)
                    expanded = True

            if block_end - block_start + 1 >= max_block_rows:
                break

            if block_end + 1 < len(current_rows):
                right_indices = self._formula_row_pred_item_indices(current_rows[block_end + 1], ordered_pred_formula_raw)
                if not right_indices or min(right_indices) <= max_raw_pred_idx + 1:
                    block_end += 1
                    raw_pred_indices.update(right_indices)
                    expanded = True

            if not expanded:
                break

        return block_start, block_end, raw_pred_indices

    def _build_local_split_formula_candidate(self, local_gt_items, local_pred_items, pred_item, img_name):
        pred_signature = self._formula_pred_item_signature(pred_item)
        mixed_pred_items = []
        inserted_split = False
        split_line_count = 0

        for local_pred_item in local_pred_items:
            if not inserted_split and (
                local_pred_item is pred_item
                or self._formula_pred_item_signature(local_pred_item) == pred_signature
            ):
                split_pred_items = split_equation_arrays([local_pred_item])
                if len(split_pred_items) <= 1:
                    return None, None
                mixed_pred_items.extend(split_pred_items)
                split_line_count = len(split_pred_items)
                inserted_split = True
            else:
                mixed_pred_items.append(local_pred_item)

        if not inserted_split:
            return None, None

        prepared = self._prepare_formula_candidate_inputs(
            local_gt_items,
            mixed_pred_items,
            split_pred_formula=False,
        )
        candidate_rows = self._build_display_formula_dp_candidate(prepared, img_name, allow_gaps=False)
        if not candidate_rows:
            return None, prepared
        return self._normalize_formula_match_categories(candidate_rows), {
            'prepared': prepared,
            'split_line_count': split_line_count,
        }

    def _build_multiline_formula_replacement_candidates(self, gt_formula, pred_formula, current_formula_matches, img_name):
        if not gt_formula or not pred_formula or not current_formula_matches:
            return list(current_formula_matches or []), []

        current_rows = sorted(
            [dict(item) for item in current_formula_matches],
            key=lambda item: self._formula_sort_key((item.get('gt_position') or [""])[0]),
        )

        split_gt_formula = split_gt_equation_arrays(gt_formula)
        gt_item_by_order = {
            self._formula_order_key(self._get_formula_position(item)): item
            for item in split_gt_formula
        }
        ordered_pred_formula_raw = sorted(
            pred_formula,
            key=lambda item: (
                self._get_formula_range(item)[0],
                self._get_formula_pred_category_type(item) == 'equation_inline',
                self._get_formula_range(item)[1],
            ),
        )

        replacement_candidates = []
        for pred_raw_idx, pred_item in enumerate(ordered_pred_formula_raw):
            split_pred_items = split_equation_arrays([pred_item])
            if len(split_pred_items) <= 1:
                continue

            linked_indices = [
                idx for idx, row in enumerate(current_rows)
                if self._formula_match_uses_pred_item(row, pred_item)
            ]
            if not linked_indices:
                continue

            block_start, block_end, raw_pred_indices = self._expand_formula_replacement_block(
                current_rows,
                linked_indices,
                ordered_pred_formula_raw,
                pred_raw_idx,
                len(split_pred_items),
            )
            block_rows = current_rows[block_start:block_end + 1]
            local_gt_items = []
            valid_block = True
            for row in block_rows:
                gt_order_key = self._formula_order_key((row.get('gt_position') or [""])[0])
                gt_item = gt_item_by_order.get(gt_order_key)
                if gt_item is None:
                    valid_block = False
                    break
                local_gt_items.append(gt_item)
            if not valid_block or len(local_gt_items) <= 1:
                continue

            if raw_pred_indices:
                raw_pred_window_start = min(raw_pred_indices)
                raw_pred_window_end = max(raw_pred_indices)
            else:
                raw_pred_window_start = pred_raw_idx
                raw_pred_window_end = pred_raw_idx
            local_pred_items = ordered_pred_formula_raw[raw_pred_window_start:raw_pred_window_end + 1]

            candidate_rows, candidate_meta = self._build_local_split_formula_candidate(
                local_gt_items,
                local_pred_items,
                pred_item,
                img_name,
            )
            if not candidate_rows:
                continue

            current_cost = sum(float(row.get('edit', 1)) for row in block_rows)
            candidate_cost = sum(float(row.get('edit', 1)) for row in candidate_rows)
            if candidate_cost + 1e-12 >= current_cost:
                continue

            replaced_rows = []
            for local_idx, candidate_row in enumerate(candidate_rows):
                base_row = block_rows[local_idx]
                updated_row = dict(candidate_row)
                updated_row['gt_idx'] = base_row.get('gt_idx', candidate_row.get('gt_idx'))
                updated_row['gt_position'] = base_row.get('gt_position', candidate_row.get('gt_position'))
                updated_row['gt_attribute'] = base_row.get('gt_attribute', candidate_row.get('gt_attribute'))
                replaced_rows.append(updated_row)

            replacement_candidates.append({
                'block_start': block_start,
                'block_end': block_end,
                'rows': replaced_rows,
                'gain': current_cost - candidate_cost,
                'current_cost': current_cost,
                'candidate_cost': candidate_cost,
                'split_line_count': candidate_meta['split_line_count'],
            })

        return current_rows, replacement_candidates

    def _maybe_apply_formula_only_fallback(self, gt_formula, pred_formula, current_formula_matches, match_gt2pred, img_name):
        current_rows, replacement_candidates = self._build_multiline_formula_replacement_candidates(
            gt_formula,
            pred_formula,
            current_formula_matches,
            img_name,
        )
        if not replacement_candidates:
            return current_formula_matches

        updated_matches = list(current_rows)
        used_row_indices = set()
        for candidate in sorted(replacement_candidates, key=lambda item: item['gain'], reverse=True):
            block_indices = set(range(candidate['block_start'], candidate['block_end'] + 1))
            if block_indices & used_row_indices:
                continue
            updated_matches[candidate['block_start']:candidate['block_end'] + 1] = candidate['rows']
            used_row_indices.update(block_indices)
            logger.info(
                f"[formula-fallback] {img_name}: local_split gain={candidate['gain']:.4f}, "
                f"edit {candidate['current_cost']:.4f}->{candidate['candidate_cost']:.4f}, "
                f"rows={candidate['block_start']}-{candidate['block_end']}, "
                f"split_lines={candidate['split_line_count']}"
            )
        return updated_matches

    def _should_skip_formula_to_text_cross_norm(self, text):
        text = str(text or '')
        if not text:
            return False

        stripped = text.strip()
        if len(stripped) < 80:
            return False

        latex_commands = re.findall(r'\[A-Za-z]+', stripped)
        ascii_words = re.findall(r'[A-Za-z]{3,}', stripped)
        math_symbol_count = sum(ch in '=+-*/_^{}<>≈≠≤≥×÷' for ch in stripped)
        malformed_delimiter = (
            stripped.count('\[') != stripped.count('\]')
            or stripped.count('\(') != stripped.count('\)')
            or stripped.count('$') % 2 == 1
        )

        if len(ascii_words) >= 8 and len(latex_commands) <= 2 and math_symbol_count <= 6:
            return True
        if malformed_delimiter and len(ascii_words) >= 6:
            return True
        return False

    def _should_use_aggressive_formula_cross_norm(self):
        flag = str(
            os.getenv(
                'OMNIDOCBENCH_AGGRESSIVE_FORMULA_CROSS_NORM',
                os.getenv('FORMULA_CROSS_NORM', '0'),
            )
        ).strip().lower()
        return flag in {'1', 'true', 'yes', 'on'}

    def _cross_category_text(self, category_type, text):
        text = str(text or '')
        if not self._is_formula_category(category_type):
            return textblock2unicode(text)
        stripped_text = strip_formula_delimiters(text)
        if not self._should_use_aggressive_formula_cross_norm():
            return textblock2unicode(stripped_text)
        if self._should_skip_formula_to_text_cross_norm(text):
            return textblock2unicode(stripped_text)
        return formula_to_text(text)

    def _adapt_cross_category_norm(self, item):
        gt_category = item.get('gt_category_type', '')
        pred_category = item.get('pred_category_type', '')
        if not gt_category or not pred_category:
            return item
        if self._is_formula_category(gt_category) == self._is_formula_category(pred_category):
            return item

        gt_text = self._cross_category_text(gt_category, item.get('gt', ''))
        pred_text = self._cross_category_text(pred_category, item.get('pred', ''))
        item['gt'] = gt_text
        item['pred'] = pred_text
        item['norm_gt'] = normalized_text(gt_text)
        item['norm_pred'] = normalized_text(pred_text)
        item['edit'] = self._normalized_edit_distance(item['norm_gt'], item['norm_pred'])
        return item

    def _median_length(self, lengths):
        if not lengths:
            return 0
        sorted_lengths = sorted(lengths)
        middle = len(sorted_lengths) // 2
        if len(sorted_lengths) % 2 == 1:
            return sorted_lengths[middle]
        return (sorted_lengths[middle - 1] + sorted_lengths[middle]) / 2

    def _text_item_length(self, item):
        category_type = item.get('category_type')
        if category_type == 'equation_isolated':
            return 0
        raw_text = item.get('content')
        if raw_text is None:
            raw_text = item.get('text', '')
        norm_text = normalized_text(str(raw_text or ''))
        return len(norm_text)

    def _resolve_prediction_path(self, pred_folder, img_name):
        pred_path = os.path.join(pred_folder, img_name[:-4] + '.md')
        if os.path.exists(pred_path):
            return pred_path

        pred_path = os.path.join(pred_folder, img_name[:-4].replace('.pdf', '') + '.mmd')
        if os.path.exists(pred_path):
            return pred_path

        pred_path = os.path.join(pred_folder, img_name[:-4].replace('.pdf', '') + '.md')
        if os.path.exists(pred_path):
            return pred_path

        pred_path = os.path.join(pred_folder, img_name + '.md')
        if os.path.exists(pred_path):
            return pred_path

        print(f'!!!WARNING: No prediction for {img_name}, evaluate as empty page')
        return None

    def _match_single_page(self, index, sample, pred_folder):
        page_start = time.monotonic()
        img_name = os.path.basename(sample["page_info"]["image_path"])
        pred_path = self._resolve_prediction_path(pred_folder, img_name)

        with latex_timeout_context(img_name=img_name, pred_path=pred_path):
            read_start = time.monotonic()
            pred_content = read_md_file(pred_path) if pred_path else ""
            self._log_slow_stage(img_name, 'read_prediction', read_start)

            process_start = time.monotonic()
            result = self.process_get_matched_elements(sample, pred_content, img_name)
            self._log_slow_stage(img_name, 'process_page', process_start)
            self._log_slow_stage(img_name, 'page_total', page_start)

        return {
            'index': index,
            'img_name': img_name,
            'pred_path': pred_path,
            'result': result,
        }

    def _collect_page_matches(self, gt_samples, pred_folder):
        if self.match_workers <= 1:
            process_bar = tqdm(gt_samples, ascii=True, ncols=140)
            page_results = []
            for index, sample in enumerate(process_bar):
                page_result = self._match_single_page(index, sample, pred_folder)
                process_bar.set_description(
                    f"Processing {os.path.basename(page_result['pred_path']) if page_result['pred_path'] else page_result['img_name']}"
                )
                page_results.append(page_result)
            return page_results

        logger.info(f"[quick-match] using page workers={self.match_workers}")
        page_results = []
        progress_bar = tqdm(total=len(gt_samples), ascii=True, ncols=140, desc='Matching pages')
        with ThreadPoolExecutor(max_workers=self.match_workers) as executor:
            future_to_index = {
                executor.submit(self._match_single_page, index, sample, pred_folder): index
                for index, sample in enumerate(gt_samples)
            }
            for future in as_completed(future_to_index):
                try:
                    page_results.append(future.result())
                except Exception:
                    print(traceback.format_exc())
                    raise
                finally:
                    progress_bar.update(1)
        progress_bar.close()
        page_results.sort(key=lambda item: item['index'])
        return page_results

    # 对gt和预测结果进行匹配，调用 process_get_matched_elements 函数进行匹配处理，最终将匹配结果整理成一个字典返回
    def get_matched_elements(self, gt_samples, pred_folder):
        plain_text_match = []
        display_formula_match = []
        html_table_match = []
        latex_table_match = []
        order_match = []
        for page_result in self._collect_page_matches(gt_samples, pred_folder):
            plain_text_match_clean, formated_display_formula, latex_table_match_s, html_table_match_s, order_match_single = page_result['result']

            if order_match_single:
                order_match.append(order_match_single)
            if plain_text_match_clean:
                plain_text_match.extend(plain_text_match_clean)
            if formated_display_formula:
                display_formula_match.extend(formated_display_formula)
            if latex_table_match_s:
                latex_table_match.extend(latex_table_match_s)
            if html_table_match_s:
                html_table_match.extend(html_table_match_s)

        display_formula_match_clean, display_formula_match_others = [], []
        for item in display_formula_match:
            pred_category_type = item.get("pred_category_type", None)
            if pred_category_type not in ['equation_inline', 'equation_isolated', ''] and not item.get('pred_cdm'):
                display_formula_match_others.append(item)
            else:
                display_formula_match_clean.append(item)
        display_formula_match = display_formula_match_clean
        if display_formula_match_others:
            plain_text_match.extend(display_formula_match_others)

        table_match = html_table_match
        table_format = 'html'
            
        # with open('./qwen_latex_table_match.json','w',encoding='utf-8') as f:
        #     json.dump(latex_table_match,f,indent=4,ensure_ascii=False)
        # with open('./qwen_html_table_match.json','w',encoding='utf-8') as f:
        #     json.dump(html_table_match,f,indent=4,ensure_ascii=False)


        matched_samples_all = {
            'text_block': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(plain_text_match),
            'display_formula':  DATASET_REGISTRY.get('recogition_end2end_base_dataset')(display_formula_match), 
            'table': DATASET_REGISTRY.get('recogition_end2end_table_dataset')(table_match, table_format),
            'reading_order': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(order_match)
        }
      

        return matched_samples_all
    
    #0403 提取gt的table跟pred的table进行匹配 -> 未匹配上的pred_table 去掉html格式然后丢进去混合匹配
    def process_get_matched_elements(self, sample, pred_content, img_name):
        if self.match_method == 'simple_match':   # add match choice
            match_gt2pred = match_gt2pred_simple
        elif self.match_method == 'quick_match':
            def match_gt2pred(gt_items, pred_items, line_type, img_name):
                return match_gt2pred_quick(
                    gt_items,
                    pred_items,
                    line_type,
                    img_name,
                    truncated_timeout_sec=self.quick_match_truncated_timeout_sec,
                    fallback_short_line_max_chars=self.timeout_fallback_short_line_max_chars,
                    fallback_target_chunk_chars=self.timeout_fallback_target_chunk_chars,
                    fallback_max_chunk_chars=self.timeout_fallback_max_chunk_chars,
                    fallback_max_chunk_span=self.timeout_fallback_max_chunk_span,
                    fallback_order_window=self.timeout_fallback_order_window,
                    fallback_order_penalty=self.timeout_fallback_order_penalty,
                )
        elif self.match_method == 'no_split':
            match_gt2pred = match_gt2pred_no_split
        else:
            print('Invalid match method name. The quick_match will be used.')
            match_gt2pred = match_gt2pred_quick

        stage_start = time.monotonic()
        pred_dataset = md_tex_filter(pred_content)
        self._log_slow_stage(img_name, 'md_tex_filter', stage_start)
        stage_start = time.monotonic()
        gt_page_elements = self.get_page_elements(sample)
        self._log_slow_stage(img_name, 'get_page_elements', stage_start)

        gt_mix,pred_dataset_mix = [],[]
        for category in pred_dataset:
            if category not in ['html_table','latex_table','md2html_table','formula_rescue']:
                pred_dataset_mix.extend(pred_dataset[category])
        # for category in gt_page_elements:
        #     if category not in ['table']:
        #         gt_mix.extend(gt_page_elements[category])
        gt_mix = self.get_page_elements_list(gt_page_elements, ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
                                                'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption',
                                                'header', 'footer', 'page_footnote', 'page_number', 'equation_isolated'])
        if gt_mix:
            gt_mix = self.get_sorted_text_list(gt_mix)

        display_formula_match_s = []
        plain_text_match_clean = []
        latex_table_match_s = []
        html_table_match_s = []
        order_match_single = []

        pred_table_candidates = list(pred_dataset['html_table']) if pred_dataset['html_table'] else []
        if pred_dataset['latex_table']:
            for table_item in pred_dataset['latex_table']:
                converted_item = dict(table_item)
                converted_item['content'] = normalized_table(table_item['content'], 'latex')
                converted_item['category_type'] = 'html_table'
                converted_item['fine_category_type'] = converted_item.get('fine_category_type', 'latex2html_table')
                pred_table_candidates.append(converted_item)

        if gt_page_elements.get('table'):
            stage_start = time.monotonic()
            gt_table = self.get_sorted_text_list(gt_page_elements['table'])
            html_table_match_s, unmatch_table_pred = match_gt2pred_simple(gt_table, pred_table_candidates, 'html_table', img_name)
            html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds
            self._log_slow_stage(img_name, 'table_match', stage_start)

        stage_start = time.monotonic()
        try:
            match = func_timeout(self.match_timeout_sec, match_gt2pred, args=(gt_mix, pred_dataset_mix, 'text_all', img_name))
        except FunctionTimedOut as e1:
            if self.enable_timeout_match_fallback:
                print(f"[match-timeout] {img_name}: quick_match exceeded {self.match_timeout_sec}s, fallback to chunked Hungarian", flush=True)
                match = match_gt2pred_timeout_safe(
                    gt_mix,
                    pred_dataset_mix,
                    img_name,
                    short_line_max_chars=self.timeout_fallback_short_line_max_chars,
                    target_chunk_chars=self.timeout_fallback_target_chunk_chars,
                    max_chunk_chars=self.timeout_fallback_max_chunk_chars,
                    max_chunk_span=self.timeout_fallback_max_chunk_span,
                    order_window=self.timeout_fallback_order_window,
                    order_penalty=self.timeout_fallback_order_penalty,
                    fallback_reason='page_timeout',
                )
            else:
                match = []
        except Exception as e:
            print(traceback.format_exc())
            sys.exit()  
        
        fallback_reason = getattr(match, 'fallback_reason', None)
        if fallback_reason:
            self._record_match_fallback(img_name, fallback_reason)

        self._log_slow_stage(img_name, 'text_match', stage_start)

        stage_start = time.monotonic()
        match = [self._adapt_cross_category_norm(item) for item in match]
        self._log_slow_stage(img_name, 'cross_category_norm', stage_start)
        stage_start = time.monotonic()
        match = self._maybe_apply_local_text_span_fallback(match, pred_dataset_mix)
        self._log_slow_stage(img_name, 'text_local_fallback', stage_start)

        plain_text_match_s = []
        for item in match:
            gt_category = item.get('gt_category_type',None)
            if gt_category in ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
                                                    'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption',
                                                    'header', 'footer', 'page_footnote', 'page_number']:
                plain_text_match_s.append(item)
            elif gt_category == 'equation_isolated':
                display_formula_match_s.append(item)

        display_formula_match_s = [x for x in display_formula_match_s if x['gt_idx'] != [""]]

        gt_formula = self.get_sorted_text_list(gt_page_elements['equation_isolated']) if gt_page_elements.get('equation_isolated') else []
        pred_formula = self._build_formula_candidate_pool(pred_dataset)
        if gt_formula or pred_formula:
            stage_start = time.monotonic()
            display_formula_match_s = self._match_display_formula_candidates(
                gt_formula,
                pred_formula,
                display_formula_match_s,
                img_name,
            )
            self._log_slow_stage(img_name, 'formula_match_candidate', stage_start)
            display_formula_match_s = self._attach_formula_cdm_surrogates(display_formula_match_s)

        if not plain_text_match_s:
            # print(f'Time out for text match of {img_name}. The plain text match will be empty.')
            # print(f'No text match of {img_name}. The plain text match will be empty.')
            pass
        else:
            # Categories that need to be ignored for text
            plain_text_match_clean = self.filtered_out_ignore(plain_text_match_s, ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption'])
            # plain_text_match_clean = self.filtered_out_ignore(plain_text_match_s, ['figure_footnote', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number'])
        order_match_s = []
        for matches in [plain_text_match_clean, display_formula_match_s, html_table_match_s, latex_table_match_s]:
            if matches:
                order_match_s.extend(matches)
        if order_match_s:
            stage_start = time.monotonic()
            order_match_single = self.get_order_paired(order_match_s, img_name)
            self._log_slow_stage(img_name, 'reading_order_pair', stage_start)

        return [plain_text_match_clean, display_formula_match_s, latex_table_match_s, html_table_match_s, order_match_single]        

    


@DATASET_REGISTRY.register("recogition_end2end_base_dataset")
class RecognitionEnd2EndBaseDataset():
    def __init__(self, samples):
        img_id = 0
        for sample in samples:
            if not sample.get('img_id'):
                sample['img_id'] = img_id
            img_id += 1
        self.samples = samples
    def __getitem__(self, idx):
        return self.samples[idx]

@DATASET_REGISTRY.register("recogition_end2end_table_dataset")
class RecognitionEnd2EndTableDataset(RecognitionTableDataset):
    def __init__(self, samples, table_format):
        self.pred_table_format = table_format
        self.samples = self.normalize_data(samples)

    def normalize_data(self, samples):
        img_id = 0

        for sample in samples:
            p = sample['pred']
            r = sample['gt']
            p = normalized_table(p, self.pred_table_format)
            r = normalized_table(r)
            sample['norm_gt'] = r
            sample['norm_pred'] = p
            sample['img_id'] = sample['img_id'] if sample.get('img_id') else img_id
            img_id += 1

        return samples
