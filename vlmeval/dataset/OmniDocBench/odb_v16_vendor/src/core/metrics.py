"""Structured metric entrypoints for evaluation and reporting."""

from collections import defaultdict

import pandas as pd
from tabulate import tabulate

from src.metrics.cal_metric import (
    call_BLEU,
    call_CDM,
    call_Edit_dist,
    call_METEOR,
    call_TEDS,
)


def show_result(results):
    for metric_name in results.keys():
        print(f'{metric_name}:')
        score_table = [[k, v] for k, v in results[metric_name].items()]
        print(tabulate(score_table))
        print('=' * 100)


def sort_nested_dict(d):
    if isinstance(d, dict):
        return {k: sort_nested_dict(v) for k, v in sorted(d.items())}
    return d


def _iter_attribute_dicts(attribute_value):
    if isinstance(attribute_value, dict):
        yield attribute_value
        return
    if isinstance(attribute_value, list):
        for item in attribute_value:
            yield from _iter_attribute_dicts(item)


def _build_page_attribute_labels(page_info_s):
    labels = ['ALL']
    for k, v in (page_info_s or {}).items():
        if isinstance(v, list):
            for special_issue in v:
                if 'table' not in special_issue:
                    labels.append(special_issue)
        else:
            labels.append(k + ": " + str(v))
    return labels


def _append_missing_page_rows(metric_rows, metric_name, page_info, gt_page_names):
    existing_pairs = {(row['image_name'], row['attribute']) for row in metric_rows}
    for img_name in sorted(gt_page_names or []):
        page_info_s = page_info.get(img_name, {})
        for attribute in _build_page_attribute_labels(page_info_s):
            key = (img_name, attribute)
            if key in existing_pairs:
                continue
            metric_rows.append({
                'image_name': img_name,
                'metric': metric_name,
                'attribute': attribute,
                'score': 0.0,
                'upper_len': 0,
            })
            existing_pairs.add(key)


def get_full_labels_results(samples):
    if not samples:
        return {}
    label_group_dict = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        label_list = []
        if not sample.get("gt_attribute") or not sample.get('metric'):
            continue
        for anno in sample["gt_attribute"]:
            for anno_dict in _iter_attribute_dicts(anno):
                for k, v in anno_dict.items():
                    label_list.append(k + ": " + str(v))
        for label_name in list(set(label_list)):
            for metric, score in sample['metric'].items():
                label_group_dict[label_name][metric].append(score)

    print('----Anno Attribute---------------')
    result = {'sample_count': {}}
    for attribute in label_group_dict.keys():
        for metric, scores in label_group_dict[attribute].items():
            mean_score = sum(scores) / len(scores)
            if not result.get(metric):
                result[metric] = {}
            result[metric][attribute] = mean_score
            result['sample_count'][attribute] = len(scores)
    result = sort_nested_dict(result)
    show_result(result)
    return result


def get_page_split(samples, page_info, gt_page_names=None, expected_metrics=None):
    if not page_info:
        return {}
    result_list = defaultdict(list)
    for sample in samples:
        img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith('.png') else '_'.join(sample['img_id'].split('_')[:-1])
        page_info_s = page_info[img_name]
        if not sample.get('metric'):
            continue
        for metric, score in sample['metric'].items():
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            result_list[metric].append({
                'image_name': img_name,
                'metric': metric,
                'attribute': 'ALL',
                'score': score,
                'upper_len': max(len(gt), len(pred))
            })
            for attribute in _build_page_attribute_labels(page_info_s):
                if attribute == 'ALL':
                    continue
                result_list[metric].append({
                    'image_name': img_name,
                    'metric': metric,
                    'attribute': attribute,
                    'score': score,
                    'upper_len': max(len(gt), len(pred))
                })

    expected_metric_names = list(expected_metrics or [])
    if 'TEDS' in expected_metric_names and 'TEDS_structure_only' not in expected_metric_names:
        expected_metric_names.append('TEDS_structure_only')
    for metric_name in expected_metric_names:
        result_list.setdefault(metric_name, [])

    result = {}
    if result_list.get('Edit_dist'):
        df = pd.DataFrame(result_list['Edit_dist'])
        up_total_avg = df.groupby(["image_name", "attribute"]).apply(
            lambda x: (x["score"] * x['upper_len']).sum() / x['upper_len'].sum()
        ).groupby('attribute').mean()
        result['Edit_dist'] = up_total_avg.to_dict()
    for metric in result_list.keys():
        if metric == 'Edit_dist':
            continue
        metric_rows = list(result_list[metric])
        if gt_page_names:
            _append_missing_page_rows(metric_rows, metric, page_info, gt_page_names)
        if not metric_rows:
            continue
        df = pd.DataFrame(metric_rows)
        page_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: x["score"].mean()).groupby('attribute').mean()
        result[metric] = page_avg.to_dict()

    result = sort_nested_dict(result)
    show_result(result)
    return result

__all__ = [
    "call_BLEU",
    "call_CDM",
    "call_Edit_dist",
    "call_METEOR",
    "call_TEDS",
    "get_full_labels_results",
    "get_page_split",
    "show_result",
    "sort_nested_dict",
]
