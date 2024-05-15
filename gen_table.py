import copy as cp
import json
from collections import defaultdict
from urllib.request import urlopen

import gradio as gr
import numpy as np
import pandas as pd

from meta_data import DEFAULT_BENCH, META_FIELDS, URL


def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False


def load_results():
    data = json.loads(urlopen(URL).read())
    return data


def nth_large(val, vals):
    return sum([1 for v in vals if v > val]) + 1


def format_timestamp(timestamp):
    date = timestamp[:2] + '.' + timestamp[2:4] + '.' + timestamp[4:6]
    time = timestamp[6:8] + ':' + timestamp[8:10] + ':' + timestamp[10:12]
    return date + ' ' + time


def model_size_flag(sz, FIELDS):
    if pd.isna(sz) and 'Unknown' in FIELDS:
        return True
    if pd.isna(sz):
        return False
    if '<10B' in FIELDS and sz < 10:
        return True
    if '10B-20B' in FIELDS and sz >= 10 and sz < 20:
        return True
    if '20B-40B' in FIELDS and sz >= 20 and sz < 40:
        return True
    if '>40B' in FIELDS and sz >= 40:
        return True
    return False


def model_type_flag(line, FIELDS):
    if 'OpenSource' in FIELDS and line['OpenSource'] == 'Yes':
        return True
    if 'API' in FIELDS and line['OpenSource'] == 'No' and line['Verified'] == 'Yes':
        return True
    if 'Proprietary' in FIELDS and line['OpenSource'] == 'No' and line['Verified'] == 'No':
        return True
    return False


def BUILD_L1_DF(results, fields):
    check_box = {}
    check_box['essential'] = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model']
    # revise there to set default dataset
    check_box['required'] = ['Avg Score', 'Avg Rank'] + DEFAULT_BENCH
    check_box['avg'] = ['Avg Score', 'Avg Rank']
    check_box['all'] = check_box['avg'] + fields
    type_map = defaultdict(lambda: 'number')
    type_map['Method'] = 'html'
    type_map['Language Model'] = type_map['Vision Model'] = type_map['OpenSource'] = type_map['Verified'] = 'str'
    check_box['type_map'] = type_map

    res = generate_table(results, fields)
    df = pd.DataFrame(res)
    df = df.sort_values('Avg Score')
    df = df.iloc[::-1]
    return df, check_box


def BUILD_L2_DF(results, dataset):
    res = defaultdict(list)
    fields = list(list(results.values())[0][dataset].keys())
    non_overall_fields = [x for x in fields if 'Overall' not in x]
    overall_fields = [x for x in fields if 'Overall' in x]
    if dataset == 'MME':
        non_overall_fields = [x for x in non_overall_fields if not listinstr(['Perception', 'Cognition'], x)]
        overall_fields = overall_fields + ['Perception', 'Cognition']
    if dataset == 'OCRBench':
        non_overall_fields = [x for x in non_overall_fields if not listinstr(['Final Score'], x)]
        overall_fields = ['Final Score']

    for m in results:
        item = results[m]
        if dataset not in item:
            continue
        meta = item['META']
        for k in META_FIELDS:
            if k == 'Parameters (B)':
                param = meta['Parameters']
                res[k].append(float(param.replace('B', '')) if param != '' else None)
            elif k == 'Method':
                name, url = meta['Method']
                res[k].append(f'<a href="{url}">{name}</a>')
            else:
                res[k].append(meta[k])
        fields = [x for x in fields]

        for d in non_overall_fields:
            res[d].append(item[dataset][d])
        for d in overall_fields:
            res[d].append(item[dataset][d])

    df = pd.DataFrame(res)
    all_fields = overall_fields + non_overall_fields
    # Use the first 5 non-overall fields as required fields
    required_fields = overall_fields if len(overall_fields) else non_overall_fields[:5]

    if 'Overall' in overall_fields:
        df = df.sort_values('Overall')
        df = df.iloc[::-1]

    check_box = {}
    check_box['essential'] = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model']
    check_box['required'] = required_fields
    check_box['all'] = all_fields
    type_map = defaultdict(lambda: 'number')
    type_map['Method'] = 'html'
    type_map['Language Model'] = type_map['Vision Model'] = type_map['OpenSource'] = type_map['Verified'] = 'str'
    check_box['type_map'] = type_map
    return df, check_box


def generate_table(results, fields, df=None):
    res = defaultdict(list)
    for i, m in enumerate(results):
        item = results[m]
        meta = item['META']
        for k in META_FIELDS:
            if k == 'Parameters (B)':
                param = meta['Parameters']
                res[k].append(float(param.replace('B', '')) if param != '' else None)
            elif k == 'Method':
                name, url = meta['Method']
                res[k].append(f'<a href="{url}">{name}</a>')
                res['name'].append(name)
            else:
                res[k].append(meta[k])
        scores, ranks = [], []
        for d in fields:
            key_name = 'Overall' if d != 'OCRBench' else 'Final Score'
            res[d].append(item[d][key_name])
            if d == 'MME':
                scores.append(item[d][key_name] / 28)
            elif d == 'OCRBench':
                scores.append(item[d][key_name] / 10)
            else:
                scores.append(item[d][key_name])
            ranks.append(nth_large(item[d][key_name], [x[d][key_name] for x in results.values()]))
        res['Avg Score'].append(round(np.mean(scores), 1))
        res['Avg Rank'].append(round(np.mean(ranks), 2))
    if df is None:
        return res
    else:
        res = pd.DataFrame(res)
        df.set_index('name', inplace=True)
        res.set_index('name', inplace=True)
        df.update(res)
        df = df.sort_values('Avg Score')
        df = df.iloc[::-1]
    return df
