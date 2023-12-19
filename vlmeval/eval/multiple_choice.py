import os.path as osp
import pandas as pd
from tqdm import tqdm
from vlmeval.api import OpenAIWrapper, OpenAIWrapperInternal
from vlmeval.utils import can_infer, track_progress_rich, TSVDataset
from vlmeval.smp import *
import numpy as np

fout = None
INTERNAL = os.environ.get('INTERNAL', 0)

abbrs = {
    'coarse_perception': 'CP', 
    'finegrained_perception (instance-level)': 'FP-S', 
    'finegrained_perception (cross-instance)': 'FP-C', 
    'logic_reasoning': 'LR',
    'relation_reasoning': 'RR',
    'attribute_reasoning': 'AR'
}

def report_acc(df):
    # assert group in [None, 'category', 'l2-category']
    res = defaultdict(list)

    if 'split' in df:
        res['split'] = ['full', 'dev', 'test']
    else:
        res['split'] = 'dev'
    
    for group in [None, 'l2-category', 'category']:
        if group is None:
            if 'split' in df:
                res['Overall'] = [np.mean(df['hit']), np.mean(df[df['split'] == 'dev']['hit']), np.mean(df[df['split'] == 'test']['hit'])]
            else:
                res['Overall'] = [np.mean(df['hit'])]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = abbrs[ab] if ab in abbrs else ab
                sub_df = df[df[group] == ab]
                if 'split' in df:
                    res[ab_name] = [np.mean(sub_df['hit']), np.mean(sub_df[sub_df['split'] == 'dev']['hit']), np.mean(sub_df[sub_df['split'] == 'test']['hit'])]
                else:
                    res[ab_name] = [np.mean(sub_df['hit'])]
    return pd.DataFrame(res)

def extract_options(item):
    options = []
    for c in 'ABCD':
        if c in item and not pd.isna(item[c]):
            options.append(item[c])
        else:
            return options
    return options

def build_prompt(question, options, prediction):
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output E. "\
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and E. \n"
        "Example 1: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: a cute teddy bear\nYour output: A\n"
        "Example 2: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: Spider\nYour output: E\n"
        "Example 3: \n"
        "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)

def build_prompt_cn(question, options, prediction):
    tmpl = (
        "你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。"
        "你会被提供：一个问题，多个选项，一个答案。你的任务是找到与答案意义最相近的选项。"
        "如果所有选项的意义都与答案显著不同，则输出 E。"
        "你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 E。"
        "例 1:"
        "问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 一只可爱的泰迪熊\n输出: A\n"
        "例 2: \n"
        "问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 蜘蛛\n输出: E\n"
        "例 3: \n"
        "问题: {}?\n选项: {}\n答案: {}\n输出: "
    )
    return tmpl.format(question, options, prediction)

def build_choices(item):
    ret = {}
    for ch in 'ABCD':
        if not pd.isna(item[ch]):
            ret[ch] = item[ch]
    return ret

def prefetch_answer(item):
    choices = build_choices(item)
    return can_infer(item['prediction'], choices)

def extract_answer_from_item(model, item):
    logger = get_logger('Evaluation')
    # It will return: (pred, raw, llm_time)
    options = extract_options(item)
    option_str = build_options(options)

    if cn_string(item['question']):
        prompt = build_prompt_cn(item['question'], option_str, item['prediction'])
    else:
        prompt = build_prompt(item['question'], option_str, item['prediction'])
    retry = 3
    choices = build_choices(item)

    ret = can_infer(item['prediction'], choices)
    if ret: 
        return dict(opt=ret, log=item['prediction'])
    
    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            msg = 'GPT API failed to answer. '
            logger.warning(msg)
            retry -= 1
        else:
            ret = can_infer(ans, choices)
            if ret:
                return dict(opt=ret, log=ans)
            else:
                logger.warning(f'GPT output includes 0 or more than 1 letter in "ABCD": {ans}')
                retry -= 1

        if retry == 0:
            num_options = sum([ch in item for ch in 'ABCD'])
            if num_options >= 2:
                chars = string.ascii_uppercase[:num_options]
                chars = chars + 'E'
                num_options += 1
                tmp = rd.randint(0, num_options - 1)
                return dict(opt=chars[tmp], log='Failed to predict, thus randomly generate one. ')
            
def prefetch_sub_data(sub_data, answer_map, verbose=False):
    lt = len(sub_data)
    GT, PRED = [], []
    for i in range(lt):
        item = sub_data.iloc[i]
        idx = item['index']
        GT.append(answer_map[idx])
        PRED.append(prefetch_answer(item))
        if PRED[-1] and (GT[-1] != PRED[-1]):
            log = f"Failed in Prefetching Rolling {i}: Answer is {GT[-1]}, Prediction is {item['prediction']}, Pre-fetched is {PRED[-1]}. "
            return dict(hit=0, log=log)
    flag = True
    for g, p in zip(GT, PRED):
        if g != p:
            flag = False 
    ret = (dict(hit=1, log="Succeed During Pre-fetching"), ) if flag else (None, )
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
            if PRED[i] != GT[i]:
                log += f"Failed in Rolling {i}: Answer is {GT[i]}; Prediction is {sub_data.iloc[i]['prediction']}; Pre-fetched is {PRED[i]}; Match Log is {match_log}.\n"
                return dict(hit=0, log=log)
            else:
                log += f"Rolling {i}: Answer is {GT[i]}, Prediction is {sub_data.iloc[i]['prediction']}, Pre-fetched is {PRED[i]}.\n"

    return dict(hit=1, log=log)

def eval_data_groups(model, data_groups, answer_map, result, result_file, nproc=16):
    prefetched = [prefetch_sub_data(g, answer_map) for g in data_groups]
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
            assert result[k]['hit'] == v['hit'] and result[k]['log'] == v['log']
        else:
            result[k] = v
    dump(result, result_file)

def multiple_choice_eval(eval_file, dataset=None, model='chatgpt-0613', nproc=4, verbose=False):
    logger = get_logger('Evaluation')

    assert dataset is not None
    if dataset == 'MMBench_TEST_CN':
        dataset = 'MMBench_CN'
    elif dataset == 'MMBench_TEST_EN':
        dataset = 'MMBench'

    rd.seed(2680)
    suffix = eval_file.split('.')[-1]
    assert model in ['chatgpt-0613', "exact_matching"]
    name_str = 'openai' if model == 'chatgpt-0613' else model

    if model == 'exact_matching':
        model = None
    else:
        model_name = 'gpt-3.5-turbo-0613'
        if INTERNAL:
            model = OpenAIWrapperInternal(model_name, verbose=verbose)
        else:
            model = OpenAIWrapper(model_name, verbose=verbose)
    
    logger.info(f'Evaluating {eval_file}')
    result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')
    result = {}
    if osp.exists(result_file):
        result = load(result_file)
        
    data = load(eval_file)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    for k in data.keys():
        data[k.lower() if k not in 'ABCD' else k] = data.pop(k)

    meta = TSVDataset(dataset).data

    cate_map = {i: c for i, c in zip(meta['index'], meta['category'])}
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    l2_cate_map = {i: c for i, c in zip(meta['index'], meta['l2-category'])} if 'l2-category' in meta else None
    split_map = {i: c for i, c in zip(meta['index'], meta['split'])} if 'split' in meta else None

    data = data[data['index'].isin(answer_map)]
    data_main = data[data['index'] < int(1e6)]
    meta_idx_set = set(meta['index'])
    data_main = data_main[data_main['index'].isin(meta_idx_set)]
    
    lt = len(data_main)
    hit, tot = 0, 0

    data_groups = []
    for i in tqdm(range(lt)):
        # Dealing with the normal part
        item_main = data_main.iloc[i]
        idx = item_main['index']
        
        if idx in result:
            correct = result[idx]['hit']
            assert correct in [0, 1]
            hit += correct
            tot += 1
            continue
            
        sub_data = data[data['index'] % int(1e6) == idx]
        data_groups.append(sub_data)

    if len(data_groups):
        if model is not None:
            eval_data_groups(
                model=model, 
                data_groups=data_groups, 
                answer_map=answer_map,
                nproc=nproc, 
                result=result, 
                result_file=result_file)
        else:
            logger.warning("Exact Matching mode, will not do GPT-based answer matching. ")
            keys = [x.iloc[0]['index'] % 1e6 for x in data_groups]
            for k in keys:
                result[k] = dict(hit=0, log="Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.")
            dump(result, result_file)
        
    tmp_pth = f'/tmp/{timestr()}.xlsx'
    dump(data_main, tmp_pth)
    data_main = load(tmp_pth)

    res = load(result_file)
    indices = data_main['index']

    data_main['hit'] = [res[i]['hit'] for i in indices]
    data_main['log'] = [res[i]['log'] for i in indices]

    main_idx = data_main['index']
    data_main['category'] = [cate_map[i] for i in main_idx]
    if l2_cate_map is not None:
        data_main['l2-category'] = [l2_cate_map[i] for i in main_idx]
    if split_map is not None:
        data_main['split'] = [split_map[i] for i in indices]
    
    # load split
    dump(data_main, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
    data_main = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
    
    acc = report_acc(data_main)
    score_file = eval_file.replace(f'.{suffix}', f'_acc.csv')
    dump(acc, score_file)
    logger.info(f'multiple_choice_eval successfully finished evaluating {eval_file}, results saved in {score_file}')
    logger.info(f'Score: ')
    logger.info(acc)
    return acc

def parse_args():
    parser = argparse.ArgumentParser(description="Inference LLM Answers. ")
    parser.add_argument("data", type=str, help="The question set for inference, in excel / tsv / json format. ")
    parser.add_argument("--model", type=str, help="The LLM (GPT) used for inference. ", default='chatgpt-0613', choices=['chatgpt-0613', 'exact_matching'])
    parser.add_argument(
        "--dataset", 
        type=str, 
        default='MMBench', 
        help='The dataset to evaluate', 
        choices=['MMBench', 'MMBench_CN', 'MMBench_DEV_EN', 'MMBench_DEV_CN', 'SEEDBench_IMG', 'CCBench', 'MMBench_TEST_CN', 'MMBench_TEST_EN'])
    parser.add_argument("--nproc", type=int, default=6)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    acc = multiple_choice_eval(eval_file=args.data, model=args.model, dataset=args.dataset, nproc=args.nproc, verbose=args.verbose)
    