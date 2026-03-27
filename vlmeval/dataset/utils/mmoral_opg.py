import random
from collections import defaultdict

import pandas as pd

from ...smp.file import load

# for MMOral-OPG-Bench


def build_mmoral_opg_gpt4_prompt(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])
    # Keep this prompt readable and flake8-friendly (avoid overly long lines).
    prompt = """
Given the question, compare the ground truth and prediction from AI
models, to generate a correctness score for the prediction.
The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
How many teeth are visualized in the radiograph? | 30 teeth are visualized with clear anatomical
definition. | 30 | 1.0
How many teeth are visualized in the radiograph? | 30 teeth are visualized with clear anatomical
definition. | 29 teeth are visualized with clear anatomical definition. | 0.0
What is the status of the wisdom teeth in the radiograph? | Three
wisdom teeth are detected, all of which are impacted: #18, #28, and #48.
| #18: impacted, #28: impacted, #48: erupted | 0.7
What is the condition of the teeth #26 and #14? | Teeth #26 and #14
show signs of periapical abscesses. | Teeth #26 and #23 show signs
of periapical abscesses. | 0.5
What is the condition of the bone architecture and visible structures in
the jaw? | No apparent bone loss is observed. Bilateral mandibular
canals and maxillary sinuses are clearly visible. | Bilateral
mandibular canals and maxillary sinuses are clearly visible. | 0.5
What is the clinical priority concern regarding the periapical lesions?
| Periapical cysts at #11 and #12, and granuloma at #46 require
endodontic evaluation. | Periapical lesions at #11, #12, and #46
require endodontic evaluation. | 0.8
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n
{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n
{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n
] | Crown | 0.8
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n
{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n
{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n
] | Crown at position: [1230, 627, 1276, 750] | 0.9
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n
{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n
{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n
] | Teeth at position: {\"point_2d\": [1242, 726]}},\n
{Crown at position: {\"box_2d\": [1230, 627, 1276, 750]}} | 1.0
"""
    gpt4_prompt = prompt + '\n' + ' | '.join(
        [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])
    return gpt4_prompt


def MMOral_opg_auxeval(model, line):
    def float_cvt(s):
        try:
            return float(s)
        except ValueError:
            return None

    prompt = build_mmoral_opg_gpt4_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        output = model.generate(prompt, temperature=i * 0.5)
        score = float_cvt(output)
        if score is None:
            log += f'Try {i}: output is {output}, failed to parse.\n'
        elif score < 0 or score > 1:
            log += f'Try {i}: output is {output}, invalid score: {score}.\n'
        else:
            log += 'Succeed'
            return dict(log=log, score=score)
    log += 'All 5 retries failed.\n'
    return dict(log=log, score=0.0)


def MMOral_opg_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    lt = len(data)
    cate2_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        cate2 = cate.replace(',', '_')
        if cate2 not in cate2_list:
            cate2_list.append(cate2)
        grade = float(item['score'])
        cate_list = ['Teeth', 'Patho', 'HisT', 'Jaw', 'SumRec', 'Report']
        for capa in cate_list:
            if capa in cate:
                tot[capa] += 1
                score[capa] += grade
        tot['Overall'] += 1
        tot[cate2] += 1
        score['Overall'] += grade
        score[cate2] += grade

    res = defaultdict(list)
    res2 = defaultdict(list)
    cate_list.append('Overall')
    cate2_list.append('Overall')
    for k in cate_list:
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['acc'].append(score[k] / tot[k] * 100)
    for v in cate2_list:
        res2['Category'].append(v)
        res2['tot'].append(tot[v])
        res2['acc'].append(score[v] / tot[v] * 100)
    res = pd.DataFrame(res)
    res2 = pd.DataFrame(res2)
    return res, res2


def get_single_choice_prediction(response, all_choices, index2ans):
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = ' ' + response + ' '  # add space to avoid partial match

    candidates = []

    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)
            elif f' {choice}.' in response:
                candidates.append(choice)
            elif f' {choice},' in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for index, ans in index2ans.items():
            ans_str = str(ans)
            if ans_str in response:
                candidates.append(index)

    if len(candidates) > 0:
        positions = {}
        for c in candidates:
            pos = response.find(f' {c} ')
            if pos == -1:
                pos = response.find(f'({c})')
            if pos == -1:
                pos = response.find(str(index2ans[c]))
            if pos != -1:
                positions[c] = pos

        if positions:
            return min(positions.items(), key=lambda x: x[1])[0]

    return random.choice(all_choices)
