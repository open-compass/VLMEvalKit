import pandas as pd
from .matching_util import can_infer, can_infer_lego
from ...smp import *
import numpy as np
import re

MMB_abbrs = {
    'coarse_perception': 'CP',
    'finegrained_perception (instance-level)': 'FP-S',
    'finegrained_perception (cross-instance)': 'FP-C',
    'logic_reasoning': 'LR',
    'relation_reasoning': 'RR',
    'attribute_reasoning': 'AR'
}

MMT_abbrs = {
    'visual_recognition': 'VR',
    'localization': 'Loc',
    'ocr': 'OCR',
    'counting': 'Count',
    'hallucination': 'HLN',
    'image_retrieval': 'IR',
    'threed': '3D',
    'visual_captioning': 'VC',
    'visual_grounding': 'VG',
    'doc_understanding': 'DU',
    'action_recognition': 'AR',
    'pixel_level_perception': 'PLP',
    'image-to-image_translation': 'I2IT',
    'relation_reasoning': 'RR',
    'intelligence_quotient_test': 'IQT',
    'emotion': 'Emo',
    'visual_illusion': 'VI',
    'meme_understanding': 'MemU',
    'visual_prompt_understanding': 'VPU',
    'anomaly_detection': 'AND',
    'keypoint_detection': 'KD',
    'visual_commonsense_reasoning': 'VCR',
    'image_evaluation_judgement': 'IEJ',
    'multiple_image_analysis': 'MIA',
    'cross_image_matching': 'CIM',
    'temporal_understanding': 'TU',
    'visual_code': 'VP',
    'medical_understanding': 'MedU',
    'autonomous_driving': 'AUD',
    'discipline_knowledge_reasoning': 'DKR',
    'embodied_ai': 'EA',
    'gui_navigation': 'GN'
}


def MMMU_preproc(data):
    logger = get_logger('Evaluation')
    cnt = 0
    As, Bs, Ans = list(data['A']), list(data['B']), list(data['answer'])
    lt = len(data)
    for i in range(lt):
        if pd.isna(As[i]):
            As[i] = Ans[i]
            Bs[i] = 'Other Answers'
            Ans[i] = 'A'
            cnt += 1
    logger.info(f'During MMMU_preproc in Evaluation, {cnt} open questions are re-formulated to multi-choice ones. ')
    data['A'] = As
    data['B'] = Bs
    data['answer'] = Ans
    return data


def report_acc_json(df, inds=['category'], key='hit'):
    res = {}
    assert key in df, df.keys()
    scores = list(df[key])
    data = cp.deepcopy(df)
    res['response_err_rate'] = np.mean([x is None for x in scores])
    data[key] = [x if x is not None else 0 for x in scores]
    res['overall'] = np.mean(data[key])
    for group in inds:
        if group not in df:
            continue
        cates = list(set(data[group]))
        for c in cates:
            sub = data[data[group] == c]
            res[f'{c}'] = np.mean(sub[key])
    return res


def report_acc(df, inds=['category', 'l2-category'], key='hit'):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None] + inds:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp][key]) for sp in res['split']]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = MMB_abbrs[ab] if ab in MMB_abbrs else ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
    if len(res['split']) == 1:
        res.pop('split')
    return pd.DataFrame(res)


def report_acc_MMVP(df):
    assert len(df) % 2 == 0
    for i in range(len(df) // 2):
        assert df['question'][2 * i] == df['question'][2 * i + 1]
    res = {}
    res['Average'] = np.mean(df['hit'])
    hits = list(df['hit'])
    both_correct = [hits[2 * i] and hits[2 * i + 1] for i in range(len(df) // 2)]
    res['Overall'] = np.mean(both_correct)
    return d2df(res)


def report_acc_MMT(df):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)
    res['split'] = list()
    res['Overall'] = list()
    for _, name in MMT_abbrs.items():
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
        elif group == 'category':
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
                ab_name = MMT_abbrs[ab] if ab in MMT_abbrs else ab
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


def report_acc_MMSci(df):

    df_filtered = df[df['setting'].isin(['Fig2Cap', 'SubFig2Cap', 'SubCap2Fig'])]

    subject_acc = df_filtered.groupby(['subject', 'setting'])['hit'].mean().unstack(fill_value=0)
    subject_acc['Avg'] = subject_acc.mean(axis=1)
    subject_acc.reset_index(inplace=True)

    category_acc = df_filtered.groupby(['category', 'setting'])['hit'].mean().unstack(fill_value=0)
    category_acc['Avg'] = category_acc.mean(axis=1)
    category_acc.reset_index(inplace=True)
    category_acc['category'] = 'CATEGORY_' + category_acc['category']
    category_acc.rename(columns={'category': 'subject'}, inplace=True)

    overall_acc = df_filtered.groupby(['setting'])['hit'].mean().to_frame().T
    overall_acc['Avg'] = overall_acc.mean(axis=1)
    overall_acc['subject'] = 'Overall'

    full_acc_df = pd.concat([subject_acc, category_acc, overall_acc], ignore_index=True)
    column_order = ['subject', 'Fig2Cap', 'SubFig2Cap', 'SubCap2Fig', 'Avg']
    full_acc_df = full_acc_df[column_order]
    return full_acc_df


def report_topviewrs_acc(df):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)
    print(df.columns)

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'l2-category', 'category']:
        if group is None:

            res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]

            if 'partial_match' in df:
                res['Overall_PM'] = [np.mean(df[df['split'] == sp]['partial_match']) for sp in res['split']]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = MMB_abbrs[ab] if ab in MMB_abbrs else ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
                if 'partial_match' in df:
                    res[f'{ab_name}_PM'] = [
                        np.mean(sub_df[sub_df['split'] == sp]['partial_match'])
                        for sp in res['split']
                    ]

    return pd.DataFrame(res)


def build_prompt(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        '[Question]: What is the main object in image?\n[Options]: A. teddy bear B. rabbit C. cat D. dog\n'
        '[Prediction]: a cute teddy bear\n[Output]: A\n'
        'Example 2: \n'
        '[Question]: What is the main object in image?\n[Options]: A. teddy bear B. rabbit C. cat D. dog\n'
        '[Prediction]: Spider\n[Output]: Z\n'
        'Your Task: \n'
        '[Question]: {}?\n[Options]: {}\n[Prediction]: {}\n[Output]: '
    )
    return tmpl.format(question, options, prediction)


def build_prompt_wemath(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D, E, F, G (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: <start>\nWhat is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n<end>\n'
        'Answer: <start>\na cute teddy bear\n<end>\nYour output: A\n'
        'Example 2: \n'
        'Question: <start>\nWhat is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n<end>\n'
        'Answer: <start>\nSpider\n<end>\nYour output: Z\n'
        'Example 3: \n'
        'Question: <start>\n{}\nOptions: {}\n<end>\nAnswer: <start>\n{}\n<end>\nYour output: '
    )
    question = question.replace(
        (
            "Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n"
            "<Thought process>: <<your thought process>> <Answer>: <<your option>>"
        ),
        '',
    )
    return tmpl.format(question, options, prediction)


def build_prompt_blink(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output Z."
        'If the meaning of all options are significantly different from the answer, '
        'or the answer does not select any option, output Z. '
        'Your should output one of the choices, A, B, C, D (if they are valid options), or Z.\n'
        'Example 1: \n'
        'Question: Which point is closer to the camera?\nSelect from the following choices.\n'
        'Options: A. Point A\nB. Point B\n(Z) Failed\n'
        'Answer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n'
        'Example 2: \n'
        'Question: Which point is closer to the camera?\nSelect from the following choices.\n'
        'Options: (A) Point A\n(B) Point B\n(Z) Failed\n'
        "Answer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
        'Example 3: \n'
        'Question: Which point is corresponding to the reference point?\nSelect from the following choices.\n'
        'Options: (A) Point A\n(B) Point B\n(Z) Failed\n'
        'Answer:The reference point (REF) on the first image is at the tip of the pot, '
        'which is the part used to Poke if the pots were used for that action. Looking at the second image, '
        'we need to find the part of the object that would correspond to poking.\n'
        "(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n"
        '(B) Point B is at the bottom of the spoon, which is not used for poking.\n'
        '(C) Point C is on the side of the pspoonot, which is not used for poking.\n'
        '(D) Point D is at the tip of the spoon, which is not used for poking.\n'
        '\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n'
        'Example 4: \n'
        'Question: {}?\nOptions: {}\n(Z) Failed\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)


def build_prompt_cn(question, options, prediction):
    tmpl = (
        '你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。'
        '你会被提供：一个问题，多个选项，一个模型预测。你的任务是找到与答案意义最相近的选项。'
        '如果所有选项的意义都与模型预测显著不同，则输出 Z。'
        '你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 Z。'
        '例 1:'
        '[问题]: 图中最主要的物体是什么?\n[选项]: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n[模型预测]: 一只可爱的泰迪熊\n[输出]: A\n'
        '例 2: \n'
        '[问题]: 图中最主要的物体是什么?\n[选项]: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n[模型预测]: 蜘蛛\n[输出]: Z\n'
        '你的任务: \n'
        '[问题]: {}?\n[选项]: {}\n[模型预测]: {}\n[输出]: '
    )
    return tmpl.format(question, options, prediction)


def build_prompt_LEGO(question, options, prediction,question_type):
    if question_type == 'sort':
        tmpl = (
            'You are an AI assistant who will help me to arrange options in the correct order. '
            'You are provided with a question, four options, and an answer. '
            'You need to determine the correct ordering of options based on the answer. '
            'Output should be a permutation of ABCD (like DCBA or BADC).\n'
            'Example 1:\n'
            'Question: Arrange these historical events chronologically\n'
            'Options: A. Renaissance B. Middle Ages C. Industrial Revolution D. Digital Age\n'
            'Answer: From Middle Ages to Renaissance, then Industrial Revolution, finally Digital Age\n'
            'Output: BACD\n\n'
            'Example 2:\n'
            'Question: Sort colors by wavelength (longest to shortest)\n'
            'Options: A. Red B. Green C. Blue D. Violet\n'
            'Answer: Red has longest wavelength, followed by green then blue, shortest is violet\n'
            'Output: ABCD\n\n'
            'Example 3:\n'
            'Question: {}\nOptions: {}\nAnswer: {}\nOutput:'
        )
        return tmpl.format(question, options, prediction)
    else:
        return build_prompt(question, options, prediction)


def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret


def prefetch_answer(item):
    choices = build_choices(item)
    return can_infer(item['prediction'], choices)


def extract_answer_from_item(model, item, dataset_name=None):
    logger = get_logger('Evaluation')
    # It will return: (pred, raw, llm_time)
    choices = build_choices(item)
    option_str = build_option_str(choices)

    if len(choices) == 0:
        from .extractor import LLM_Extractor
        EXTRACT_PROMPT = 'Please extract the answer (an uppercase letter) from the response and directly output it. '
        verifier = lambda x: len(str(x)) == 1 and x in string.ascii_uppercase  # noqa: E731
        extractor = LLM_Extractor(model, EXTRACT_PROMPT, verifier=verifier)
        ans = extractor.extract(item['prediction'])
        if ans:
            return dict(opt=ans, log=f"Prediction: {item['prediction']}, Extracted: {ans}")
        else:
            return dict(opt='Z', log=f"Prediction: {item['prediction']}, Failed to extract answer.")

    if dataset_name == 'BLINK':
        prompt = build_prompt_blink(item['question'], option_str, item['prediction'])
    elif dataset_name == 'WeMath':
        prompt = build_prompt_wemath(item['question'], option_str, item['prediction'])
    elif cn_string(item['question']):
        prompt = build_prompt_cn(item['question'], option_str, item['prediction'])
    elif dataset_name is not None and 'LEGO' in dataset_name:
        prompt = build_prompt_LEGO(item['question'], option_str, item['prediction'],item['question_type'])
    else:
        prompt = build_prompt(item['question'], option_str, item['prediction'])
    retry = 3

    if dataset_name is not None and 'LEGO' in dataset_name:
        ret = can_infer_lego(item['prediction'], item['question_type'], choices)
    else:
        ret = can_infer(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=item['prediction'])
    if model is None:
        return dict(opt='Z', log='Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.')

    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            logger.warning('GPT API failed to answer. ')
        else:
            if dataset_name is not None and 'LEGO' in dataset_name:
                ret = can_infer_lego(ans, item['question_type'], choices)
            else:
                ret = can_infer(ans, choices)
            if ret:
                return dict(opt=ret, log=ans)
            else:
                logger.warning(
                    f'Failed to infer: prediction is {ans}, choice labels are {set(choices)}'
                    f', Prompt is {prompt}'
                    f', Answer is {item["answer"]}' if "answer" in item else ""
                )
        retry -= 1

        if retry == 0:
            options = list(choices) + ['Z'] if 'Z' not in choices else []
            return dict(opt=rd.choice(options), log=f'Failed to predict, thus randomly generate one. {FAIL_MSG}')


# For Circular Evaluation
def prefetch_circular_group(sub_data, verbose=False):
    lt = len(sub_data)
    GT, PRED = [], []
    for i in range(lt):
        item = sub_data.iloc[i]
        GT.append(item['answer'])
        PRED.append(prefetch_answer(item))
        if PRED[-1] and (GT[-1] != PRED[-1]):
            log = (
                f'Failed in Prefetching Rolling {i}: Answer is {GT[-1]}, '
                f"Prediction is {item['prediction']}, Pre-fetched is {PRED[-1]}. "
            )
            return dict(hit=0, log=log)
    flag = True
    for g, p in zip(GT, PRED):
        if g != p:
            flag = False
    ret = (dict(hit=1, log='Succeed During Pre-fetching'), ) if flag else (None, )
    ret = ret + (GT, PRED) if verbose else ret
    return ret if len(ret) > 1 else ret[0]


def eval_vanilla(model, item, dataset_name=None):
    item = dict(cp.deepcopy(item))
    if 'options' in item and istype(item['options'], list):
        options = item['options'] if isinstance(item['options'], list) else eval(item['options'])
        for ch, opt in zip(string.ascii_uppercase, options):
            item[ch] = opt
        if item['answer'] in options:
            item['answer'] = string.ascii_uppercase[options.index(item['answer'])]

    res = extract_answer_from_item(model, item, dataset_name=dataset_name)
    opt, match_log = res['opt'], res['log']
    if opt == item['answer']:
        return dict(hit=1, log=f'Match Log: {match_log}. ')
    else:
        return dict(hit=0, log=f'Match Log: {match_log}. ')


# For Circular Evaluation
def eval_circular_group(model, sub_data, dataset_name=None):
    prefetched = prefetch_circular_group(sub_data, verbose=True)
    if isinstance(prefetched, dict) and 'hit' in prefetched:
        return prefetched

    res, GT, PRED = prefetch_circular_group(sub_data, verbose=True)
    if res is not None:
        return res

    lt = len(sub_data)
    log = ''
    for i in range(lt):
        if PRED[i]:
            log += f'Rolling {i} Matched.\n'
        else:
            res = extract_answer_from_item(model, sub_data.iloc[i], dataset_name=dataset_name)
            opt, match_log = res['opt'], res['log']
            PRED[i] = opt
            if PRED[i] != GT[i]:
                log += (
                    f"Failed in Rolling {i}: Answer is {GT[i]}; Prediction is {sub_data.iloc[i]['prediction']}; "
                    f'Pre-fetched is {PRED[i]}; Match Log is {match_log}.\n'
                )
                return dict(hit=0, log=log)
            else:
                log += (
                    f"Rolling {i}: Answer is {GT[i]}, Prediction is {sub_data.iloc[i]['prediction']}, "
                    f'Pre-fetched is {PRED[i]}.\n'
                )

    return dict(hit=1, log=log)


# data is pd.DataFrame, result_file is a path
def mcq_vanilla_eval(model, data, nproc, result_file, dataset_name=None):
    result = {}
    if osp.exists(result_file):
        result = load(result_file)

    if dataset_name is not None and 'MMMU' in dataset_name:
        data = MMMU_preproc(data)
    rows = [row for _, row in data.iterrows() if row['index'] not in result]
    tups = [dict(model=model, item=x, dataset_name=dataset_name) for x in rows]
    keys = [x['index'] for x in rows]

    if len(tups):
        res = track_progress_rich(eval_vanilla, tups, nproc=nproc, chunksize=nproc, save=result_file, keys=keys)
        result = load(result_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v
    data['hit'] = [result[i]['hit'] for i in data['index']]
    data['log'] = [result[i]['log'] for i in data['index']]
    return data


def merge_vanilla_judge(vanilla_data, hit_key='hit', log_key='log'):
    assert 'g_index' in vanilla_data, 'g_index is required for circular evaluation.'
    g_indices = list(set(vanilla_data['g_index']))
    g_indices.sort()
    lines = []
    for g_index in g_indices:
        sub_group = vanilla_data[vanilla_data['g_index'] == g_index]
        hits = list(sub_group[hit_key])
        new_log = {k: v for k, v in zip(sub_group['index'], sub_group[log_key])}
        new_line = dict(sub_group.iloc[0])
        new_line[log_key] = new_log
        new_line[hit_key] = np.all([x > 0 for x in hits])
        lines.append(new_line)
    keys = list(lines[0].keys())
    res = {k: [item[k] for item in lines] for k in keys}
    merged_df = pd.DataFrame(res)
    return merged_df


def extract_characters_regex(s, choices=['(A)', '(B)', '(C)', '(D)', '(E)']):
    if type(s) is dict:
        s = ''
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ''
    return matches[0]


def get_dimension_rating(data_path):
    TASKS = [
        'Reasoning',
        'Perception',
    ]

    SUBTASKS = [
        'Monitoring',
        'Autonomous_Driving',
        'OCR with Complex Context',
        'Diagram and Table',
        'Remote Sensing',
    ]
    data = load(data_path)
    results = {}
    results['Overall'] = {}
    for task in TASKS:
        results[f'{task}'] = {}
        for subtask in SUBTASKS:
            results[f'{task}'][f'{subtask}'] = {}

    for i in range(len(data)):
        question = data.iloc[i]
        Task = question['category'].split('/')[0]
        Subtask = question['category'].split('/')[1]
        Category = question['l2-category'].lower()
        if 'attribute' in Category.lower():
            Category = Category.split('/')[0] + '/attribute'
        if question['score'] >= 0:
            cnt = question['score']
            if Category not in results[Task][Subtask].keys():
                results[Task][Subtask][f'{Category}'] = {'true': cnt, 'false': 1 - cnt}
            else:
                results[Task][Subtask][f'{Category}']['true'] += cnt
                results[Task][Subtask][f'{Category}']['false'] += 1 - cnt

    sum_all, succ_all = 0, 0
    for task, tasks_values in results.items():
        cnt_task, sum_task = 0, 0
        for substask, subtask_value in tasks_values.items():
            cnt_subtask, sum_subtask = 0, 0
            for category, category_dict in subtask_value.items():
                cnt_subtask += category_dict['true']
                sum_subtask += category_dict['false'] + category_dict['true']
                acc = category_dict['true'] / (category_dict['false'] + category_dict['true'])
                results[task][substask][category] = acc
            if sum_subtask == 0:
                acc_subtasks = 0
            else:
                acc_subtasks = cnt_subtask / sum_subtask
            cnt_task += cnt_subtask
            sum_task += sum_subtask
            results[task][substask]['Avg'] = acc_subtasks
        if sum_task == 0:
            acc_task = 0
        else:
            acc_task = cnt_task / sum_task
        succ_all += cnt_task
        sum_all += sum_task
        results[task]['Avg'] = acc_task
    results['Overall'] = succ_all / sum_all
    return results
