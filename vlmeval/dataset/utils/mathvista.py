from ...smp import *
from .matching_util import can_infer


def check_equal_judge(model, a, b):
    prompt = f'Are {a} and {b} equal? (Yes/No)'
    res = model.generate(prompt, temperature=0)
    return 'yes' in res.lower()


def is_equal(asw, gt_asw, judge=None) -> bool:
    if not isinstance(asw, str) != str or not isinstance(gt_asw, str):
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
    if judge is not None:
        return check_equal_judge(judge, asw, gt_asw)
    return False


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_mathvista_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}


def MathVista_auxeval_MCQ(model, line, choices):
    choices = list_to_dict(choices)
    prompt = build_mathvista_gpt4_prompt(line)
    answer_label = None
    for k in choices:
        if str(choices[k]) == line['answer']:
            answer_label = k
            break
    assert answer_label is not None, (choices, line['answer'])
    log, retry = '', 5
    res = can_infer(line['prediction'], choices)
    if res:
        return dict(log='Prefetch succeed', res=res, hit=is_equal(res, answer_label))

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            res2 = can_infer(res, choices)
            if res2:
                return dict(log=log, res=res2, hit=is_equal(res2, answer_label))
            else:
                return dict(log=log, res=res, hit=is_equal(res, answer_label, judge=model))
    log += f'All 5 retries failed. {FAIL_MSG}\n'
    return dict(log=log, res='', hit=False)


def MathVista_auxeval_Open(model, line):
    log, retry = '', 5
    prompt = build_mathvista_gpt4_prompt(line)
    if is_equal(line['prediction'], line['answer']):
        return dict(log='Prefetch succeed', res=line['prediction'], hit=True)

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res, hit=is_equal(res, line['answer'], judge=model))

    log += f'All 5 retries failed. {FAIL_MSG}\n'
    return dict(log=log, res='', hit=False)


def MathVista_auxeval(model, line):
    line['answer'], line['prediction'] = str(line['answer']), str(line['prediction'])
    if line['question_type'] == 'multi_choice':
        choices = eval(line['choices'])
        return MathVista_auxeval_MCQ(model, line, choices)
    else:
        return MathVista_auxeval_Open(model, line)


def MathVista_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    skill_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['task']
        tot['Overall'] += 1
        try:
            skills = eval(item['skills'])
        except SyntaxError:
            skills = [item['skills']]
        for skill in skills:
            if skill not in skill_list:
                skill_list.append(skill)
            tot[skill] += 1
        tot[cate] += 1
        if item['hit']:
            hit['Overall'] += 1
            hit[cate] += 1
            for skill in skills:
                hit[skill] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res['Task&Skill'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100)
    res = pd.DataFrame(res)
    return res
