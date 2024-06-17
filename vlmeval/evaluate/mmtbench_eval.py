import os.path as osp

import pandas as pd
from tqdm import tqdm
import numpy as np

from vlmeval.evaluate.misc import build_judge
from vlmeval.utils import can_infer, track_progress_rich, TSVDataset
from vlmeval.smp import *


abbrs = {
    "visual_recognition": "VR",
    "localization": "Loc",
    "ocr": "OCR",
    "counting": "Count",
    "hallucination": "HLN",
    "image_retrieval": "IR",
    "threed": "3D",
    "visual_captioning": "VC",
    "visual_grounding": "VG",
    "doc_understanding": "DU",
    "action_recognition": "AR",
    "pixel_level_perception": "PLP",
    "image-to-image_translation": "I2IT",
    "relation_reasoning": "RR",
    "intelligence_quotient_test": "IQT",
    "emotion": "Emo",
    "visual_illusion": "VI",
    "meme_understanding": "MemU",
    "visual_prompt_understanding": "VPU",
    "anomaly_detection": "AND",
    "keypoint_detection": "KD",
    "visual_commonsense_reasoning": "VCR",
    "image_evaluation_judgement": "IEJ",
    "multiple_image_analysis": "MIA",
    "cross_image_matching": "CIM",
    "temporal_understanding": "TU",
    "visual_code": "VP",
    "medical_understanding": "MedU",
    "autonomous_driving": "AUD",
    "discipline_knowledge_reasoning": "DKR",
    "embodied_ai": "EA",
    "gui_navigation": "GN"
}


def report_acc(df):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)
    res['split'] = list()
    res['Overall'] = list()
    for _, name in abbrs.items():
        res[name] = list()

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits

    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'category', 'l2-category']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]
            res['Overall'].extend([np.mean(df['hit'])])
        elif group not in df:
            continue
        elif group == "category":
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
                res[ab_name].extend([np.mean(sub_df['hit'])])
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                sub_task_name_list = df[df['l2-category'] == ab]['category'].unique()
                sub_task_acc = []
                for sub_task_name in sub_task_name_list:
                    sub_df = df[df['category'] == sub_task_name]
                    sub_task_acc.append([np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']])
                
                new_acc = []
                for i in range(len(sub_task_acc[0])):
                    new_acc.append(sum([_[i] for _ in sub_task_acc]) / len([_ for _ in sub_task_acc]))
                ab_name = abbrs[ab] if ab in abbrs else ab
                res[ab_name] = new_acc

                sub_task_acc = []
                for sub_task_name in sub_task_name_list:
                    sub_df = df[df['category'] == sub_task_name]
                    sub_task_acc.append([np.mean(sub_df['hit'])])
                new_acc = []
                for i in range(len(sub_task_acc[0])):
                    new_acc.append(sum([_[i] for _ in sub_task_acc]) / len([_ for _ in sub_task_acc]))

                res[ab_name].extend(new_acc)

    res['split'].append('ALL')
    return pd.DataFrame(res)


def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret


def prefetch_answer(item):
    choices = build_choices(item)
    return can_infer(item['prediction'], choices)


def prefetch_sub_data(sub_data, answer_map, verbose=False):
    lt = len(sub_data)
    GT, PRED = [], []
    for i in range(lt):
        item = sub_data.iloc[i]
        idx = item['index']
        GT.append(answer_map[idx])
        PRED.append(prefetch_answer(item))
        if PRED[-1] and (PRED[-1] in string.ascii_uppercase):
            
            return dict(opt=PRED[-1])
    flag = True
    for p in PRED:
        if not p:
            flag = False
    ret = (dict(opt=PRED[-1]), ) if flag else (None, )
    ret = ret + (GT, PRED) if verbose else ret
    return ret if len(ret) > 1 else ret[0]


def eval_sub_data(model, sub_data, answer_map):
    res, GT, PRED = prefetch_sub_data(sub_data, answer_map, verbose=True)
    if res is not None:
        return res

    lt = len(sub_data)
    log = ''
    for i in range(lt):
        if PRED[i]:
            log += f'Rolling {i} Matched.\n'
        else:
            res = extract_answer_from_item(model, sub_data.iloc[i])
            opt, match_log = res['opt'], res['log']
            PRED[i] = opt
            return dict(opt = opt)
            # if PRED[i] != GT[i]:
            #     # log += (
            #     #     f"Failed in Rolling {i}: Answer is {GT[i]}; Prediction is {sub_data.iloc[i]['prediction']}; "
            #     #     f'Pre-fetched is {PRED[i]}; Match Log is {match_log}.\n'
            #     # )
            #     return dict(opt = opt)
            # else:
            #     # log += (
            #     #     f"Rolling {i}: Answer is {GT[i]}, Prediction is {sub_data.iloc[i]['prediction']}, "
            #     #     f'Pre-fetched is {PRED[i]}.\n'
            #     # )
            pass

    return dict(opt = opt)


def extract_answer_from_item(model, item):
    logger = get_logger('Evaluation')
    # It will return: (pred, raw, llm_time)
    choices = build_choices(item)
    option_str = build_option_str(choices)

    if cn_string(item['question']):
        prompt = build_prompt_cn(item['question'], option_str, item['prediction'])
    else:
        prompt = build_prompt(item['question'], option_str, item['prediction'])
    retry = 3

    ret = can_infer(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=item['prediction'])

    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            logger.warning('GPT API failed to answer. ')
        else:
            ret = can_infer(ans, choices)
            if ret:
                return dict(opt=ret, log=ans)
            else:
                logger.warning(f'Output includes 0 / > 1 letter among candidates {set(choices)} and Z: {ans}')
        retry -= 1

        if retry == 0:
            options = list(choices) + ['Z'] if 'Z' not in choices else []
            return dict(opt=rd.choice(options), log='Failed to predict, thus randomly generate one. ')


def build_prompt(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)


def eval_data_groups(model, data_groups, answer_map, result, result_file, nproc=16):
    prefetched = [prefetch_sub_data(g, answer_map, verbose=False) for g in data_groups]
    remain = []
    for dg, pf in zip(data_groups, prefetched):
        if pf:
            result[dg.iloc[0]['index'] % 1e6] = pf
        else:
            remain.append(dg)
    dump(result, result_file)
    tups = [(model, x, answer_map) for x in remain]
    keys = [x.iloc[0]['index'] % 1e6 for x in remain]
    if len(tups) == 0:
        return
    
    assert model

    res = track_progress_rich(
        eval_sub_data,
        tups,
        nproc=nproc,
        chunksize=nproc,
        save=result_file,
        keys=keys)
    result = load(result_file)
    for k, v in zip(keys, res):
        if k in result:
            # assert result[k]['hit'] == v['hit'] and result[k]['log'] == v['log']
            pass
        else:
            result[k] = v
    dump(result, result_file)



def MMTBench_result_transfer(eval_file, dataset='default', **judge_kwargs):
    logger = get_logger('Evaluation')
    INTERNAL = os.environ.get('INTERNAL', 0)

    nproc = judge_kwargs.pop('nproc', 4)

    rd.seed(2680)
    suffix = eval_file.split('.')[-1]
    model = judge_kwargs['model']
    assert model in ['chatgpt-0613', 'exact_matching', 'gpt-4-0125']
    name_str_map = {
        'chatgpt-0613': 'openai',
        'gpt-4-0125': 'gpt4'
    }
    name_str = name_str_map[model] if model in name_str_map else model

    if model == 'exact_matching':
        model = None
    else:
        if INTERNAL or gpt_key_set():
            model = build_judge(**judge_kwargs)
        else:
            logger.error('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

    logger.info(f'Evaluating {eval_file}')
    result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_option.pkl')
    result = {}
    if osp.exists(result_file):
        result = load(result_file)

    data = load(eval_file)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    if dataset != 'default':
        meta = TSVDataset(dataset).data
    else:
        logger.warning('Dataset is not provided, try to use the original `eval_file` as meta data. ')
        meta = load(eval_file)
        assert 'index' in meta and 'answer' in meta, 'Essentail columns missing in the eval_file.'

    answer_map = {i: 'A' for i, c in zip(meta['index'], meta['index'])}  # 123
    cate_map = {i: c for i, c in zip(meta['index'], meta['category'])} if 'category' in meta else None
    l2_cate_map = {i: c for i, c in zip(meta['index'], meta['l2-category'])} if 'l2-category' in meta else None
    split_map = {i: c for i, c in zip(meta['index'], meta['split'])} if 'split' in meta else None

    if cate_map is not None and np.all([pd.isna(x) for x in cate_map.values()]):
        cate_map = None
    if l2_cate_map is not None and np.all([pd.isna(x) for x in l2_cate_map.values()]):
        l2_cate_map = None
    if split_map is not None and np.all([pd.isna(x) for x in split_map.values()]):
        split_map = None

    data = data[data['index'].isin(cate_map)]
    data_main = data[data['index'] < int(1e6)]
    meta_idx_set = set(meta['index'])
    data_main = data_main[data_main['index'].isin(meta_idx_set)]

    lt = len(data_main)

    data_groups = []
    for i in tqdm(range(lt)):
        # Dealing with the normal part
        item_main = data_main.iloc[i]
        idx = item_main['index']

        if idx in result:
            continue

        sub_data = data[data['index'] % int(1e6) == idx]
        data_groups.append(sub_data)

    if len(data_groups):
         eval_data_groups(
            model=model,
            data_groups=data_groups,
            answer_map=answer_map,
            nproc=nproc,
            result=result,
            result_file=result_file)

    tmp_pth = f'/tmp/{timestr()}.xlsx'
    dump(data_main, tmp_pth)
    data_main = load(tmp_pth)

    res = load(result_file)
    indices = data_main['index']

    data_main['opt'] = [res[i]['opt'] for i in indices]
    # data_main['log'] = [res[i]['log'] for i in indices]

    main_idx = data_main['index']
    if cate_map is not None:
        data_main['category'] = [cate_map[i] for i in main_idx]
    if l2_cate_map is not None:
        data_main['l2-category'] = [l2_cate_map[i] for i in main_idx]
    if split_map is not None:
        data_main['split'] = [split_map[i] for i in indices]

    # load split
    output_path = eval_file.replace(f'.{suffix}', f'_{name_str}_submission.tsv')
    dump(data_main, eval_file.replace(f'.{suffix}', f'_{name_str}_submission.tsv'))
    return output_path
