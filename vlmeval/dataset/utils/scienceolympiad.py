import requests
from math_verify import verify, parse
import pandas as pd
from tqdm import tqdm
from ...smp import *
from collections import defaultdict
import logging


FAIL_MSG = 'Failed to obtain answer via API.'

DEFAULT_PROMPT_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and return a letter "A" or "B" to indicate whether the predicted answer is correct or incorrect.

# Input
[Question]
{question}

[Reference Answer]
{gold}

[Predicted Answer]
{pred}

# Evaluation Rules
- The predicted answer of the model may contain the reasoning process, you should spot the final answer from it.
- Evaluate the model's answer based on correctness compared to the reference answer.
- Ignore language differences: If the core meaning of the predicted answer (after extracting the final result) is consistent with the reference answer (even if one is in English and the other in Chinese), it is considered correct.
- Formula/Chemical Expression Evaluation: For chemical formulas, chemical equations, or physical formulas, judge based on core consistency:
  * Chemical formulas: Consistent elemental composition and valence (e.g., H2O and H₂O are equivalent, NaOH, sodium hydroxide, and 氢氧化钠 are equivalent if the reference/predicted uses name/formula respectively).
  * Chemical equations: Consistent reactants, products, and balanced stoichiometry (ignore minor formatting differences like spaces or superscript/subscript display).
  * Physical formulas: Consistent variables, mathematical relationships, and key constants (ignore formatting differences like parentheses or symbol case if the core logic is identical).
- For questions with multiple minor issues, the predicted answer is determined to be correct only if it meets the reference answers for all minor issues; otherwise, it is considered incorrect.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct  

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
""" 

def parse_boxed(final_ans, ground_truth):
    ground_truth = "\\boxed{"+ ground_truth + "}"
    pred_result= last_boxed_only_string(final_ans)
    if pred_result is None:
        return None, None
    
    extract_ans = None
    if pred_result.startswith("\\boxed{"):
        extract_ans = pred_result[7:-1] 
    elif pred_result.startswith("\\fbox{"):
        extract_ans = pred_result[6:-1]  
    elif pred_result.startswith("\\boxed "):
        extract_ans = pred_result[7:] 
        
    is_pass = eval_math_verify(pred_result, ground_truth)
    return is_pass, extract_ans

def eval_math_verify(predicted, ground_truth):
    predicted = parse(predicted, parsing_timeout=None)
    ground_truth = parse(ground_truth, parsing_timeout=None)
    is_correct = verify(predicted, ground_truth,timeout_seconds=None)
    return is_correct

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval

def ScienceOlympiad_auxeval(model, line):
    
    log = ''
    response = line['prediction']
    if not response or not isinstance(response, str):
        log += 'Invalid response format, returning False.'
        return dict(log=log, res=False)
    
    if '</think>' in response:
        final_ans = response.split('</think>')[1]
    else:
        final_ans = response
        
    gt = line['answer'].strip()
    if final_ans is None or not isinstance(final_ans, str) or final_ans.strip() == "":
        log += 'Invalid final answer, returning False.'
        return dict(log=log, res=False)
    is_pass, extract_ans = parse_boxed(final_ans, gt)
    if  is_pass == True:
        log += 'Prefetch succeed. Math_verify evaluate Succeessfully.'
        return dict(log=log, res=True)
    # if extract_ans is None:
    #     log += 'Invalid extract answer, returning False.'
    #     return dict(log=log, res=False)
    
    # llm judge
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

            if FAIL_MSG in res or res.strip() not in ['A', 'B']:
                log += f'Try {i}: output is {prediction}, answer is {extract_ans}. judge_model response is {res}, failed to eval with judge_model.\n'
            else:
                log += 'Judge model evaluate Succeessfully.'
                if res.strip() == 'A':
                    re_score = True 
                    logging.info(f"Judge model evaluate Succeessfully. Response is: {res.strip()}")
                else:
                    re_score = False
                    logging.info(f"Judge model evaluate Succeessfully. Response is: {res.strip()}")
                return dict(log=log, res=re_score)
        log += 'All 5 retries failed.\n'
        
    return dict(log=log, res=False)
            
def ScienceOlympiad_acc(result_file):
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