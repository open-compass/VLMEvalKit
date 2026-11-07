import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import vlmeval.dataset.MDPBench.metrics.cal_metric  # noqa: F401
from vlmeval.smp import dump, get_intermediate_file_path, load
from .dataset.end2end_dataset import End2EndDataset
from .metrics.show_result import get_full_labels_results, get_page_split, show_result
from .registry.registry import DATASET_REGISTRY, METRIC_REGISTRY


class NativeEnd2EndEvaluator:
    def __init__(
        self,
        eval_file: str,
        tsv_path: str,
        match_method: str = 'quick_match',
        filter_types: Optional[Dict] = None,
    ):
        self.eval_file = eval_file
        self.match_method = match_method
        self.filter_types = filter_types

        predictions = load(eval_file)
        references = load(tsv_path)

        if 'index' not in predictions.columns or 'index' not in references.columns:
            raise ValueError('Both prediction file and dataset TSV must contain an index column.')

        df = pd.merge(predictions, references, on='index', how='inner')
        pred_col = 'prediction_x' if 'prediction_x' in df.columns else 'prediction'
        ans_col = 'answer_y' if 'answer_y' in df.columns else 'answer'

        if ans_col not in df.columns:
            raise ValueError('Dataset TSV must contain answer column for MDPBench evaluation.')
        if pred_col not in df.columns:
            df['prediction'] = ''
            pred_col = 'prediction'

        self.references = []
        self.predictions = []
        for _, row in df.iterrows():
            ans = row.get(ans_col)
            pred = row.get(pred_col, '')
            try:
                if isinstance(ans, str):
                    ans = json.loads(ans)
                if isinstance(ans, dict):
                    ans = dict(ans)
                    ans['index'] = str(row.get('index', ''))
                    self.references.append(ans)
                    self.predictions.append(str(pred) if pred is not None else '')
            except Exception:
                continue

        if self.filter_types:
            filtered_refs = []
            filtered_preds = []
            for gt_sample, pred in zip(self.references, self.predictions):
                select_flag = True
                page_attr = gt_sample.get('page_info', {}).get('page_attribute', {})
                if isinstance(page_attr, list) and page_attr:
                    page_attr = page_attr[0]
                for k, v in self.filter_types.items():
                    if not isinstance(page_attr, dict) or page_attr.get(k) != v:
                        select_flag = False
                        break
                if select_flag:
                    filtered_refs.append(gt_sample)
                    filtered_preds.append(pred)
            self.references = filtered_refs
            self.predictions = filtered_preds

        self.dafault_metircs_dict = {
            'text_block': {'metric': ['Edit_dist']},
            'display_formula': {'metric': ['Edit_dist', 'CDM']},
            'table': {'metric': ['TEDS', 'Edit_dist']},
            'reading_order': {'metric': ['Edit_dist']},
        }

        self.dataset_runner = End2EndDataset.__new__(End2EndDataset)
        self.dataset_runner.match_method = self.match_method
        self.dataset_runner.slim_mode = False

    def _postprocess_matches(self, plain_text_match, display_formula_match,
                             html_table_match, latex_table_match):
        from pylatexenc.latex2text import LatexNodes2Text

        from .utils.data_preprocess import clean_string

        display_formula_match_clean, display_formula_match_others = [], []
        for item in display_formula_match:
            pred_category_type = item.get('pred_category_type', None)
            if pred_category_type not in ['equation_inline', 'equation_isolated', '']:
                gt = item.get('gt', None)
                try:
                    item['gt'] = LatexNodes2Text().latex_to_text(gt)
                except Exception:
                    pass
                item['norm_gt'] = clean_string(item['gt'])
                display_formula_match_others.append(item)
            else:
                display_formula_match_clean.append(item)

        display_formula_match = display_formula_match_clean
        if display_formula_match_others and plain_text_match:
            plain_text_match.extend(display_formula_match_others)

        if latex_table_match:
            latex_to_html = []
            for latex_table in latex_table_match:
                for k in list(latex_table.keys()):
                    if 'pred' in k:
                        latex_table[k] = ''
                latex_table['edit'] = 1
                latex_to_html.append(latex_table)
            html_table_match.extend(latex_to_html)

        if len(latex_table_match) > len(html_table_match):
            table_match = latex_table_match
            table_format = 'latex'
        else:
            table_match = html_table_match
            table_format = 'html'

        return plain_text_match, display_formula_match, table_match, table_format

    def get_matched_elements(self):
        plain_text_match = []
        display_formula_match = []
        html_table_match = []
        latex_table_match = []
        order_match = []
        save_time = time.time()

        for i, sample in tqdm(list(enumerate(self.references)),
                              desc='Matching elements', ascii=True):
            pred_content = self.predictions[i] if i < len(self.predictions) else ''
            img_name = os.path.basename(sample.get('page_info', {}).get('image_path', f'page_{i}'))
            img_id = os.path.splitext(img_name)[0] if img_name else f'page_{i}'
            index_name = os.path.splitext(os.path.basename(str(sample.get('index', ''))))[0]
            # Keep the matching id rule aligned with original MDPBench: use current instance id.
            current_img_id = index_name if index_name else img_id
            is_digit = len([p for p in current_img_id.split('_') if p]) == 3

            result = self.dataset_runner.process_get_matched_elements(
                sample,
                pred_content,
                current_img_id,
                save_time,
                bool(is_digit),
            )

            (
                plain_text_match_clean,
                formated_display_formula,
                latex_table_match_s,
                html_table_match_s,
                order_match_single,
            ) = result

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

        plain_text_match, display_formula_match, table_match, table_format = self._postprocess_matches(
            plain_text_match,
            display_formula_match,
            html_table_match,
            latex_table_match,
        )

        matched_samples_all = {
            'text_block': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(plain_text_match),
            'display_formula': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(display_formula_match),
            'table': DATASET_REGISTRY.get('recogition_end2end_table_dataset')(table_match, table_format),
            'reading_order': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(order_match),
        }
        return matched_samples_all

    @staticmethod
    def _page_id_from_indexed_key(key: str) -> str:
        m = re.match(r'^(.*)_\[(\d+)\]$', key)
        if m:
            return m.group(1)
        parts = key.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return key

    @staticmethod
    def _lang_from_img_id(img_id: str) -> str:
        if not isinstance(img_id, str) or '_' not in img_id:
            return 'UNKNOWN'
        lang = img_id.split('_', 1)[0].upper()
        return 'ZH-T' if lang == 'ZH-CHT' else lang

    @staticmethod
    def _compute_present_tasks_overall(pages: pd.DataFrame) -> pd.DataFrame:
        df = pages.copy()
        for k in ['text', 'formula', 'table']:
            if f'n_{k}' not in df:
                df[f'n_{k}'] = 0.0
            else:
                df[f'n_{k}'] = df[f'n_{k}'].fillna(0).astype(float)
            if f's_{k}' not in df:
                df[f's_{k}'] = np.nan
            else:
                df[f's_{k}'] = df[f's_{k}'].astype(float)

        use_text = (df['n_text'] > 0) & df['s_text'].notna()
        use_formula = (df['n_formula'] > 0) & df['s_formula'].notna()
        use_table = (df['n_table'] > 0) & df['s_table'].notna()

        denom = use_text.astype(int) + use_formula.astype(int) + use_table.astype(int)
        numer = (
            df['s_text'].fillna(0) * use_text.astype(int)
            + df['s_formula'].fillna(0) * use_formula.astype(int)
            + df['s_table'].fillna(0) * use_table.astype(int)
        )

        df['overall_i_scheme2'] = (numer / denom.replace({0: np.nan})).clip(0, 1)
        return df

    def _split_filtered_samples(self, samples, split_tag: str):
        if split_tag == 'All':
            return list(samples)
        want_is_digit = split_tag == 'digit'
        out = []
        for s in samples:
            is_digit = bool(s.get('is_digit', True))
            if is_digit == want_is_digit:
                out.append(s)
        return out

    def _summary_like_notebook_cell3(self, sample_cache: Dict[str, list]) -> pd.DataFrame:
        split_tags = ['All', 'digit', 'photo']
        split_rename_map = {'digit': 'Digit.', 'photo': 'Photo.', 'All': 'All'}
        latin_langs = ['DE', 'EN', 'ES', 'FR', 'ID', 'IT', 'NL', 'PT', 'VI']
        non_latin_langs = ['AR', 'HI', 'JP', 'KO', 'RU', 'TH', 'ZH', 'ZH-T']

        model_data = {}

        for split in split_tags:
            text_samples = self._split_filtered_samples(sample_cache.get('text_block', []), split)
            formula_samples = self._split_filtered_samples(
                sample_cache.get('display_formula', []), split)
            table_samples = self._split_filtered_samples(sample_cache.get('table', []), split)

            # text: per-page s_text = 1 - weighted Edit_dist
            text_edit_num = defaultdict(float)
            text_upper = defaultdict(float)
            text_cnt = defaultdict(int)
            for s in text_samples:
                pid = s.get('image_name') or s.get('page_id') or s.get('img_id')
                if not pid:
                    continue
                if s.get('upper_len', 0) > 0 and s.get('Edit_num') is not None:
                    text_edit_num[pid] += float(s.get('Edit_num', 0.0))
                    text_upper[pid] += float(s.get('upper_len', 0.0))
                text_cnt[pid] += 1
            s_text = {k: max(
                0.0, min(1.0, 1.0 - (text_edit_num[k] / text_upper[k]))) for k in text_upper if text_upper[k] > 0}

            # formula: per-page mean CDM
            formula_sum = defaultdict(float)
            formula_cnt = defaultdict(int)
            for s in formula_samples:
                pid = s.get('image_name') or s.get('page_id') or s.get('img_id')
                if not pid:
                    continue
                cdm = (s.get('metric') or {}).get('CDM')
                if cdm is None:
                    continue
                formula_sum[pid] += float(cdm)
                formula_cnt[pid] += 1
            s_formula = {k: formula_sum[k] / formula_cnt[k]
                         for k in formula_cnt if formula_cnt[k] > 0}

            # table: per-page mean TEDS
            table_sum = defaultdict(float)
            table_cnt = defaultdict(int)
            for s in table_samples:
                pid = s.get('image_name') or s.get('page_id') or s.get('img_id')
                if not pid:
                    continue
                teds = (s.get('metric') or {}).get('TEDS')
                if teds is None:
                    continue
                table_sum[pid] += float(teds)
                table_cnt[pid] += 1
            s_table = {k: table_sum[k] / table_cnt[k] for k in table_cnt if table_cnt[k] > 0}

            all_page_ids = set(text_cnt) | set(formula_cnt) | set(table_cnt)
            if not all_page_ids:
                continue

            pages = pd.DataFrame(index=sorted(all_page_ids))
            pages.index.name = 'img_id'
            pages['s_text'] = pd.Series(s_text)
            pages['n_text'] = pd.Series(text_cnt)
            pages['s_formula'] = pd.Series(s_formula)
            pages['n_formula'] = pd.Series(formula_cnt)
            pages['s_table'] = pd.Series(s_table)
            pages['n_table'] = pd.Series(table_cnt)
            pages['lang'] = [self._lang_from_img_id(i) for i in pages.index]

            pages_scheme2 = self._compute_present_tasks_overall(pages)

            if split == 'All':
                lang_group = pages_scheme2.groupby('lang')['overall_i_scheme2'].mean()
                for lang, score in lang_group.items():
                    model_data[(lang, '')] = round(float(score) * 100, 1)

            overall_score = pages_scheme2['overall_i_scheme2'].mean(skipna=True)
            model_data[('Overall', split_rename_map[split])] = round(float(overall_score) * 100, 1)

        latin_scores = [model_data[(lang, '')] for lang in latin_langs if (lang, '') in model_data]
        if latin_scores:
            model_data[('Latin_Avg', '')] = round(sum(latin_scores) / len(latin_scores), 1)

        non_latin_scores = [model_data[(lang, '')] for lang in non_latin_langs if (lang, '') in model_data]
        if non_latin_scores:
            model_data[('Non-Latin_Avg', '')] = round(sum(non_latin_scores)
                                                      / len(non_latin_scores), 1)

        row_entry = {}
        for lang in latin_langs:
            row_entry[f'Latin_{lang}'] = model_data.get((lang, ''), np.nan)
        row_entry['Latin_Avg'] = model_data.get(('Latin_Avg', ''), np.nan)

        for lang in non_latin_langs:
            row_entry[f'NonLatin_{lang}'] = model_data.get((lang, ''), np.nan)
        row_entry['NonLatin_Avg'] = model_data.get(('Non-Latin_Avg', ''), np.nan)

        row_entry['Overall_All'] = model_data.get(('Overall', 'All'), np.nan)
        row_entry['Overall_Digit'] = model_data.get(('Overall', 'Digit.'), np.nan)
        row_entry['Overall_Photo'] = model_data.get(('Overall', 'Photo.'), np.nan)

        return pd.DataFrame([row_entry], index=['end2end']).round(3)

    def process_generated_metric_results(self, samples, save_name: str = 'mdpbench_end2end'):
        result_all = {}
        page_info = {}
        metircs_dict = self.dafault_metircs_dict
        pages = self.references
        sample_cache = {}

        for page in pages:
            img_path = os.path.basename(page.get('page_info', {}).get('image_path', ''))
            page_info[img_path] = page.get('page_info', {}).get('page_attribute', {})

        for element in metircs_dict.keys():
            result = {}
            group_info = metircs_dict[element].get('group', [])
            cur_samples = samples[element]

            for metric in metircs_dict[element]['metric']:
                metric_val = METRIC_REGISTRY.get(metric)
                cur_samples, result_s = metric_val(cur_samples).evaluate(
                    group_info, f'{save_name}_{element}')
                if result_s:
                    result.update(result_s)

            if result:
                print(element)
                show_result(result)

            group_result = get_full_labels_results(cur_samples)
            page_result = get_page_split(cur_samples, page_info)

            result_all[element] = {'all': result, 'group': group_result, 'page': page_result}

            saved_samples = cur_samples if isinstance(cur_samples, list) else cur_samples.samples
            sample_cache[element] = saved_samples
            result_file = get_intermediate_file_path(
                self.eval_file, f'_{save_name}_{element}_result', 'json')
            dump(saved_samples, result_file)

        metric_result_file = get_intermediate_file_path(
            self.eval_file, f'_{save_name}_metric_result', 'json')
        dump(result_all, metric_result_file)

        df = self._summary_like_notebook_cell3(sample_cache)

        e2e_eval_file = get_intermediate_file_path(self.eval_file, '_End2End_Evaluation', 'json')
        dump(result_all, e2e_eval_file)
        overall_file = get_intermediate_file_path(self.eval_file, '_overall')
        dump(df, overall_file)

        print(f'The save path of End2End_Evaluation is: {e2e_eval_file}')
        print(f'The save path of overall metrics is: {overall_file}')
        return df

    def score(self):
        samples = self.get_matched_elements()
        return self.process_generated_metric_results(samples)
