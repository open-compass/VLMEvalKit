import re
import string
from tqdm import tqdm
from ...smp import *
import pandas as pd
from collections import defaultdict
import logging


FAIL_MSG = 'Failed to obtain answer via API.'


DEFAULT_PROMPT_TEMPLATE = """
You are an evaluation model. Your task is to judge whether a predicted answer
should be considered correct based on the question and ground truth.

Be objective and strict.

Question: {question}
Ground Truth: {gold}
Predicted Answer: {pred}

Please output exactly one character:
- Output '1' if the predicted answer should be considered correct.
- Output '0' if the predicted answer should be considered incorrect.
"""



NEGATIVE_WORDS = ['not', 'no', 'never', "n't", 'without']
WINDOW_SIZE = 3

def standardize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def check_negation_around_match(ground_truth, predicted):

    standardized_gt = standardize_text(ground_truth)
    gt_word_list = standardized_gt.split()
    
    standardized_pred = standardize_text(predicted)
    pred_word_list = standardized_pred.split()
    
    if not gt_word_list:
        return False

    match_indices = []
    for i in range(len(pred_word_list) - len(gt_word_list) + 1):
        if pred_word_list[i:i + len(gt_word_list)] == gt_word_list:
            match_indices.append(i)
            
    if not match_indices:
        return False

    gt_len = len(gt_word_list)
    for start_index in match_indices:
        pre_start = max(0, start_index - WINDOW_SIZE)
        for i in range(pre_start, start_index):
            if pred_word_list[i] in NEGATIVE_WORDS:
                return True
        
        post_end = min(len(pred_word_list), start_index + gt_len + WINDOW_SIZE)
        for i in range(start_index + gt_len, post_end):
            if pred_word_list[i] in NEGATIVE_WORDS:
                return True
                
    return False


def check_format(llm_output):
    if '<think>' not in llm_output or '</think>' not in llm_output:
        return 0.0
   
    if llm_output.count('<think>') != llm_output.count('</think>') or llm_output.count('<think>') != 1:
        return 0.0
    
    final_answer = llm_output.split('</think>', 1)[1].strip()
    
    if final_answer:
        return 1.0
    else:
        return 0.0

def VRSBench_auxeval(model, line):
    """
    1. 子串匹配 + 否定词检查
    2. 'yes'/'no'/数字 精确匹配
    3. LLM 模型判断
    """
    log = ''
    response = line['prediction']
    gt = line['answer'].strip() if isinstance(line['answer'], str) else line['answer']
    if not gt :
        log += 'Invalid ground truth format, returning False.'
        return dict(log=log, res=False)
    if not response or not isinstance(response, str):
        log += 'Invalid response format, returning False.'
        return dict(log=log, res=False)
    
    if '</think>' in response:
        final_ans = response.split('</think>')[1].lower()
    else:
        final_ans = response.lower()
    if final_ans is None or not isinstance(final_ans, str) or final_ans.strip() == "":
        log += 'Invalid final answer, returning False.'
        return dict(log=log, res=False)
    # 1. ground_truth 是 predicted 的子串
    if gt in final_ans:
        if check_negation_around_match(gt, final_ans):
            log += 'Decorated by a negative word, returning False.'
            return dict(log=log, res=False)
        else:
            log += 'Prefetch succeed. Matching successfully.'
            return dict(log=log, res=True)
    # 2. ground_truth 是 'yes', 'no' 或数字，需要精确匹配
    try:
        
        gt_num = float(gt)
        final_ans_num = float(final_ans)
        if gt_num == final_ans_num:
            log += 'Prefetch succeed. Matching successfully.'
            return dict(log=log, res=True)
        else:
            log += 'Not a number, returning False.'
            return dict(log=log, res=False)
        
    except ValueError:
        # 如果不是数字，则继续原始的 'yes'/'no' 判断
        if gt in ['yes', 'no']:
            if gt == final_ans:
                log += 'Prefetch succeed. Matching successfully.'
                return dict(log=log, res=True)
            else:
                log += 'Not a yes or no, returning False.'
                return dict(log=log, res=False)
        # 3. 复杂情况，调用 LLM 模型进行判断
        else:
            question = line['question']
            validation_prompt = DEFAULT_PROMPT_TEMPLATE.format(
                question=question,
                gold=gt,
                pred=final_ans
            )
            
            retry = 5
            for i in range(retry):
                prediction = line['prediction']
                res = model.generate(validation_prompt, temperature=0.1)#temperature=i * 0.5
                match = re.search(r'[01]', res)
                if FAIL_MSG in res or not match:
                    log += f'Try {i}: output is {prediction}, answer is {final_ans}. judge_model response is {res}, failed to eval with judge_model.\n'
                else:
                    log += 'Judge model evaluate Succeessfully.'
                    if match.group(0) == '1':
                        re_score = True 
                        logging.info(f"Judge model evaluate Succeessfully. Response is: {res.strip()}")
                    else:
                        re_score = False
                        logging.info(f"Judge model evaluate Succeessfully. Response is: {res.strip()}")
                    return dict(log=log, res=re_score)
            log += 'All 5 retries failed.\n'
        
    return dict(log=log, res=False)
            
def VRSBench_acc(result_file):
    data = load(result_file)
    if result_file.endswith('.json'):
        data = pd.DataFrame(data)
    tot = defaultdict(int)
    hit = defaultdict(int)
    fetch = defaultdict(int)
    lt = len(data)

    for i in tqdm(range(lt)):
        item = data.iloc[i]
        cate = item.get('category', 'Overall')

        tot['Overall'] += 1
        tot[cate] += 1
        if 'log' in item:
            log_value = item['log']
        else:
            log_value = item['answer_match_log']
        if 'Prefetch succeed' in log_value:
            fetch['Overall'] += 1
            fetch[cate] += 1
        if item.get('result'):
            hit['Overall'] += 1
            hit[cate] += 1

    res = defaultdict(list)
    for k in tot:
        res['Subject'].append(k)
        res['total'].append(tot[k])
        # res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100 if tot[k] else 0.0)
        # res['rule_prefetch'].append(fetch[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)

    return pd.DataFrame(res).sort_values('Subject', ignore_index=True)





