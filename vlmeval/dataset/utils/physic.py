from ...smp import *
from ...utils import can_infer
import timeout_decorator
import logging

try:
    from latex2sympy2 import latex2sympy
except Exception as e:
    logging.critical(f'{type(e)}: {e}')
    logging.critical('Please install latex2sympy2 by running "pip install latex2sympy2"')


FAIL_MSG = 'Failed to obtain answer via API.'


def is_equal(asw: str, gt_asw: str) -> bool:
    if type(asw) != str or type(gt_asw) != str:
        print('Warning: input is not string')
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    return False


def get_gpt4_physic_ICE():
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.
Question: Which number is missing?
Model response: The number missing in the sequence is 14.
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.
Question: What is the fraction of females facing the camera?
Model response: The fraction is 0.6.
Extracted answer: 0.6
"""

    return [example_1, example_2]


def build_physic_gpt4_prompt(line):
    task_description = "Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.\n"
    prompt = task_description
    for example in get_gpt4_physic_ICE():
        prompt += example + '\n'
    prompt += "Question: " + line['question'] + '\n'
    prompt += "Model response: " + str(line['prediction']) + '\n'
    prompt += "Extracted answer:"
    return prompt


def post_check_physic(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']

    
    res = str(response).strip()
    ans = str(ans).strip()

    if is_equal(res, ans):
        return res if prefetch else True
    else:
        return False


def PHYSIC_auxeval(model, line):
    prompt = build_physic_gpt4_prompt(line)
    log = ''
    retry = 5

    if post_check_physic(line, prefetch=True):
        res = post_check_physic(line, prefetch=True)
        return dict(log='Prefetch succeed', res=res)

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)

    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')


def PHYSIC_acc(result_file):
    import pandas as pd
    from collections import defaultdict

    data = load(result_file)
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)

    for i in range(lt):
        item = data.iloc[i]
        cate = item['category'] if 'category' in item else 'Overall'
        tot['Overall'] += 1
        tot[cate] += 1
        if item['log'] == 'Prefetch succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1
        if post_check_physic(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)

    res = pd.DataFrame(res).sort_values('Subject', ignore_index=True)
    return res
