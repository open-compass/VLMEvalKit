from vlmeval.smp import *
from vlmeval.utils import can_infer
import re
import json
import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict

FAIL_MSG = 'Failed to obtain answer via API.'

#************** Answer Evaluation ****************


def get_ICE():
    example_1 = """
Ground truth answer: 26.7kg \n
Predicted answer: The mass of block \( B \) is: 
\[ 
\boxed{26.7 \, \text\{kg\}}
\] \n
Judegement: 1
"""

    example_2 = """
Ground truth answer: 46.3 kN \n
Predicted answer: The tension \( T_B \) in the cable is approximately:
\[
\boxed{46300 \, \text{N}}
\] \n
Judegement: 1
"""

    example_3 = """
Ground truth answer: 12 m/s \n
Predicted answer: The speed of the box after 2.00 seconds is:
\[
\boxed{11.3 \, \text{m/s}}
\] \n
Judegement: 0
"""

    example_4 = """
Ground truth answer: 36.00 kg \n
Predicted answer: The mass of the hanging block \( m_2 \) must be approximately:
\[
\boxed{36.1 \, \text\{kg\}}
\] \n
Judegement: 1
"""

    example_5 = """
Ground truth answer: 4.7 m \n
Predicted answer: The stuntman and villain slide approximately **4.69 meters**.
Judegement: 1
"""

    return [example_1, example_2, example_3, example_4, example_5]

def get_ICE_MC():
    example_1 = """
Ground truth answer: A \n
Predicted answer: A \n
Judegement: 1
"""

    example_2 = """
Ground truth answer: B \n
Predicted answer: A \n
Judegement: 0
"""

    example_3 = """
Ground truth answer: C \n
Predicted answer: ### Step 1: Calculate \( l_1 \)
The lightbulb is \( 2.50 \, \text{m} \) above the floor, and the bottom of the mirror is \( 0.50 \, \text{m} \) above the floor. The vertical distance from the lightbulb to the bottom of the mirror is:
\[
\Delta y_1 = 2.50 \, \text{m} - 0.50 \, \text{m} = 2.00 \, \text{m}.
\] \n
Judegement: 0
"""

    example_4 = """
Ground truth answer: D \n
Predicted answer: The correct option is D. \n
Judegement: 1
"""

    return [example_1, example_2, example_3, example_4]

def build_phyx_gpt4_prompt(line):
    task_description = """
Please read the following example. Given predicted answer and ground truth answer, 
compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt. 
If the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If the given predicted mentions "approximately", then allow the Approximation Error, such as 0.49 and approximately 0.5, 0.81 and approximately 0.8. \n
"""
    gt_answer = line['answer']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Ground truth answer: {} \n'.format(gt_answer)
    prompt += 'Predicted answer: {} \n'.format(prediction)
    prompt += 'Judegement:'
    return prompt

def build_phyx_gpt4_prompt_MC(line):
    task_description = """
Please read the following example. Given predicted answer and ground truth answer for Multi-Choice question.
The ground truth answer would be A/B/C/D. The predicted answer would be some words containing A/B/C/D.
Please compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt. \n
"""
    gt_answer = line['answer']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_ICE_MC()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Ground truth answer: {} \n'.format(gt_answer)
    prompt += 'Predicted answer: {} \n'.format(prediction)
    prompt += 'Judegement:'
    return prompt


def PhyX_auxeval(model, line):
    prompt = build_phyx_gpt4_prompt(line)
    log = ''
    retry = 5

    gt_answer = str(line['answer'])
    prediction = line['prediction']

    # try extract final answer using re rules
    tmp = PhyX_process_line(line)
    if tmp["extracted"] != "SAME as predict":
        prediction = tmp["extracted"] 

    # judge via LLM
    if gt_answer.strip().lower() == prediction.strip().lower():
        return dict(log="Matched at string level", res=1, extracted=prediction)
    
    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)
        if FAIL_MSG in res:
            log += f'Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n'
        else:
            log += 'Compared at semantic level. '
            # print(res)
            if "1" in res or 1 == res:
                log += "Semantic equal via LLM."
                return dict(log=log, res=1, extracted=prediction)
            elif "0" in res or 0 == res:
                log += "LLM judgement {}".format(res)
                return dict(log=log, res=0, extracted=prediction)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res=0, extracted=prediction)

def PhyX_auxeval_MC(model, line):
    prompt = build_phyx_gpt4_prompt_MC(line)
    log = ''
    retry = 5

    gt_answer = str(line['answer'])
    prediction = line['prediction']


    tmp = PhyX_process_line_MC(line)
    if tmp["extracted"] != "SAME as predict":
        prediction = tmp["extracted"] # rule extracted

    # judge via LLM
    # match at string level
    if gt_answer.strip().lower() == prediction.strip().lower():
        return dict(log="Matched at string level", res=1, extracted=prediction)
    else:
        # prediction is A/B/C/D, then labeled as unmatch
        if prediction.strip() in ["A", "B", "C", "D"]:
            return dict(log="Unmatched at string level", res=0, extracted=prediction)
    
    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)
        if FAIL_MSG in res:
            log += f'Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n'
        else:
            log += 'Compared at semantic level. '
            if "1" in res or 1 == res:
                log += "Semantic equal via LLM."
                return dict(log=log, res=1, extracted=prediction)
            elif "0" in res or 0 == res:
                log += "LLM judgement {}".format(res)
                return dict(log=log, res=0, extracted=prediction)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res=0, extracted=prediction)

def PhyX_acc(result_file):
    data = load(result_file)
    lt = len(data)
    res = {}
    hit = 0
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        if cate in res.keys():
            res[cate].append(item['res'])
        else:
            res[cate] = [item['res']]
        hit += item['res']
    
    final_res = {}
    final_res["Overall Acc"] = hit/lt
    for k,v in res.items():
        final_res[k] = sum(v)/len(v)
    df = pd.DataFrame(final_res, index=[0])
    return df


def PhyX_process_line(line):
    ret = {}

    answers = str(line['answer'])

    ret["index"] = line["index"]
    ret['gt'] = answers
    ret['pred'] = line['prediction'].strip()

    pattern = r'\b(?:final\s+answer|correct\s+answer)\b[^:：]*[:：]\s*(.*?)(?=\n\n\n|\Z)'
    flags = re.IGNORECASE | re.DOTALL
    match = re.search(pattern, ret['pred'], flags=flags)

    if match:
        extracted_answer=match.group(1)
        # compare string
        ret["extracted"] = extracted_answer
        if ret['gt'].strip().lower() == extracted_answer.strip().lower():
            ret['match'] = 1
            return ret
    else:
        ret["extracted"] = "SAME as predict"


    if ret['gt'] in ret['pred']:
        ret['match'] = 1
    else:
        ret['match'] = 0
            
    return ret

def PhyX_process_line_MC(line):
    ret = {}

    answers = str(line['answer'])

    ret["index"] = line["index"]
    ret['gt'] = answers
    ret['pred'] = line['prediction'].strip()

    pattern = r'\b(?:correct|answer|option|Answer|Option|Correct)\b[\s\S]*?([A-D])'
    match = re.search(pattern, ret['pred'])

    if match:
        extracted_answer=match.group(1)
        # compare string
        ret["extracted"] = extracted_answer
        if ret['gt'].strip().lower() == extracted_answer.strip().lower():
            ret['match'] = 1
            return ret
    else:
        # try another match strategy 
        matches = re.findall(r'([ABCD]):', ret['pred'])
        if matches:
            extracted_answer=matches[-1]
            ret["extracted"] = extracted_answer
            if ret['gt'].strip().lower() == extracted_answer.strip().lower():
                ret['match'] = 1
                return ret
        else:
            ret["extracted"] = "SAME as predict"

    if ret['gt'] + ":" in ret['pred'] or ret['gt'] + "**" in ret['pred'] or "**" + ret['gt'] in ret['pred']:
        ret['match'] = 1
    else:
        ret['match'] = 0
            
    return ret
