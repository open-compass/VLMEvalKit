import os
from collections import defaultdict

import pandas as pd
from tabulate import tabulate


def show_result(results):
    for metric_name in results.keys():
        print(f'{metric_name}:')
        score_table = [[k, v] for k, v in results[metric_name].items()]
        print(tabulate(score_table))
        print('=' * 100)


def sort_nested_dict(d):
    # If it's a dictionary, recursively sort it
    if isinstance(d, dict):
        # Sort the current dictionary
        sorted_dict = {k: sort_nested_dict(v) for k, v in sorted(d.items())}
        return sorted_dict
    # If not a dictionary, return directly
    return d


def get_full_labels_results(samples):
    if not samples:
        return {}
    label_group_dict = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        label_list = []
        if not sample.get("gt_attribute"):
            continue
        for anno in sample["gt_attribute"]:
            for k, v in anno.items():
                label_list.append(k + ": " + str(v))
        # Currently if there are merged cases, calculate based on the set of all
        # labels involved after merging
        for label_name in list(set(label_list)):
            for metric, score in sample['metric'].items():
                label_group_dict[label_name][metric].append(score)

    print('----Anno Attribute---------------')
    result = {}
    result['sample_count'] = {}
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

# def get_page_split(samples, page_info):    # Sample level metric
#     if not page_info:
#         return {}
#     page_split_dict = defaultdict(lambda: defaultdict(list))
#     for sample in samples:
#         img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') else '_'.join(sample['img_id'].split('_')[:-1])
#         page_info_s = page_info[img_name]
#         if not sample.get('metric'):
#             continue
#         for metric, score in sample['metric'].items():
#             for k,v in page_info_s.items():
#                 if isinstance(v, list): # special issue
#                     for special_issue in v:
#                         if 'table' not in special_issue:  # Table-related special fields have duplicates
#                             page_split_dict[metric][special_issue].append(score)
#                 else:
#                     page_split_dict[metric][k+": "+str(v)].append(score)

#     print('----Page Attribute---------------')
#     result = {}
#     result['sample_count'] = {}
#     for metric in page_split_dict.keys():
#         for attribute, scores in page_split_dict[metric].items():
#             mean_score = sum(scores) / len(scores)
#             if not result.get(metric):
#                 result[metric] = {}
#             result[metric][attribute] = mean_score
#             result['sample_count'][attribute] = len(scores)
#     result = sort_nested_dict(result)
#     show_result(result)
#     return result


def get_page_split(samples, page_info):   # Page level metric
    if not page_info:
        return {}
    result_list = defaultdict(list)
    for sample in samples:
        # Try to find the base image name from the sample ID (which might be a variant name)
        # The logic here needs to be robust to handle names like
        # 'en_book_1_indoor.md' mapping back to 'en_book_1.png'

        current_id = sample['img_id']

        # Find the matching key in page_info
        # page_info keys are like 'en_book_1.png'
        # current_id could be 'en_book_1_indoor_...'

        matched_key = None
        # First try direct match
        if current_id in page_info:
            matched_key = current_id
        else:
            # Try to match by prefix
            # We look for a key in page_info such that current_id starts with key_without_extension
            for key in page_info.keys():
                key_no_ext = os.path.splitext(key)[0]
                # Check if current_id starts with key_no_ext AND followed by _ or end of string
                if current_id == key_no_ext or current_id.startswith(key_no_ext + '_'):
                    matched_key = key
                    break

        if matched_key:
            img_name = matched_key
            page_info_s = page_info[img_name]
        else:
            # Fallback to original logic if not found (though likely to fail if not found above)
            img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith(
                '.png') else '_'.join(sample['img_id'].split('_')[:-1])
            if img_name not in page_info:
                print(f"Warning: Could not find page info for {sample['img_id']}")
                continue
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
            for k, v in page_info_s.items():
                if isinstance(v, list):  # special issue
                    for special_issue in v:
                        if 'table' not in special_issue:  # Table-related special fields have duplicates
                            result_list[metric].append({
                                'image_name': img_name,
                                'metric': metric,
                                'attribute': special_issue,
                                'score': score,
                                'upper_len': max(len(gt), len(pred))
                            })
                else:
                    result_list[metric].append({
                        'image_name': img_name,
                        'metric': metric,
                        'attribute': k + ": " + str(v),
                        'score': score,
                        'upper_len': max(len(gt), len(pred))
                    })

    # Page level logic, accumulation is only done within pages, and mean
    # operation is performed between pages
    result = {}
    if result_list.get('Edit_dist'):   # 只有Edit_dist需要进行page level的计算
        df = pd.DataFrame(result_list['Edit_dist'])
        up_total_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: (x["score"] * x['upper_len']).sum() / x['upper_len'].sum(
        )).groupby('attribute').mean()  # At page level, accumulate edits, denominator is sum of max(gt, pred) from each sample
        # up_total_avg = df.groupby(["attribute"]).apply(lambda x:
        # (x["score"]*x['upper_len']).sum() / x['upper_len'].sum()) # whole_level
        result['Edit_dist'] = up_total_avg.to_dict()
    for metric in result_list.keys():
        if metric == 'Edit_dist':
            continue
        df = pd.DataFrame(result_list[metric])
        page_avg = df.groupby(["image_name", "attribute"]).apply(
            lambda x: x["score"].mean()).groupby('attribute').mean()  # 页面内部平均以后，再页面间的平均
        result[metric] = page_avg.to_dict()

    result = sort_nested_dict(result)
    # print('----Page Attribute---------------')
    show_result(result)
    return result
