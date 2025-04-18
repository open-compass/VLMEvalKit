import logging
import re
import timeout_decorator
from sympy import simplify, expand, trigsimp
from sympy.parsing.latex import parse_latex
from ...smp import *
from ...utils import can_infer
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from .physics_eval_utils import extract_final_answer_allform, is_equiv

FAIL_MSG = 'Failed to obtain answer via API.'


def build_physic_prompt(line):
    prompt_text = (
        "You are a physics expert assistant. Solve the following question step-by-step.\n\n"
        "At the VERY END of your answer, output ONLY the FINAL ANSWER in this format:\n\n"
        "\\[\n\\boxed{{your_final_answer_here}}\n\\]\n\n"
        "✅ You MUST put the final answer in the `\\boxed{}` environment.\n"
        "✅ This applies even if the answer is a text explanation like \"The singlet state is lower in energy.\"\n"
        "✅ Do NOT include multiple boxes.\n"
        "✅ Do NOT include \\boxed anywhere else in your reasoning.\n"
        "✅ The box must appear on the last line of the response.\n\n"
        "⚠️ WARNING: DO NOT forget to include \boxed{} with the final answer. Responses without it will be considered INVALID.\n\n"  # noqa: E501
        "Example:\n\n"
        "Question: What is the energy difference between n=2 and n=1 in hydrogen?\n"
        "Answer:\nThe energy levels are E_n = -13.6 / n² (in eV).\n"
        "E_2 = -13.6 / 4 = -3.4 eV\n"
        "E_1 = -13.6 eV\n"
        "ΔE = 13.6 - 3.4 = 10.2 eV\n"
        "\\[\n\\boxed{10.2\\ \\text{eV}}\n\\]\n\n"
        "Question: Which energy state is lower in hydrogen molecule?\n"
        "Answer:\nBased on spin multiplicity, the singlet state lies lower in energy than the triplet.\n"
        "\\[\n\\boxed{The singlet state is lower in energy}\n\\]\n\n"
        f"Question: {line['question']}\nAnswer:"
    )
    return [{"type": "text", "value": prompt_text}]


def post_check(line, prefetch=False):
    try:
        response = line['prediction'] if prefetch else line.get('res', '')
        if not response or not isinstance(response, str):
            return False

        pred_boxed = extract_final_answer_allform(response)
        gt = line['answer'].strip()

        flat_preds = [item.strip() for group in pred_boxed for item in (group if isinstance(group, list) else [group])]

        if gt in flat_preds:
            return True

        for pred in flat_preds:
            if is_equiv(pred, gt):
                return True

        return False
    except Exception as e:
        logging.warning(f'post_check error: {e}')
        return False


def PHYSIC_auxeval(model, line, i=None):
    log = ''
    retry = 3

    if post_check(line, prefetch=True):
        return dict(log='Prefetch succeed', res=line.get("prediction", ""))

    for i in range(retry):

        prediction = model.generate(line, temperature=0.5 * i)

        line_copy = line.copy()
        line_copy['res'] = prediction

        if FAIL_MSG in prediction:
            log += f'Try {i}: output failed to parse.\n'
        else:
            if post_check(line_copy):
                return dict(log='Succeed', res=prediction)
            else:
                log += f'Try {i}: wrong result.\n'

    return dict(log=log, res=prediction)


def PHYSIC_acc(result_file):
    data = load(result_file)
    tot = defaultdict(int)
    fetch = defaultdict(int)
    hit = defaultdict(int)
    lt = len(data)

    for i in tqdm(range(lt)):
        item = data.iloc[i]
        cate = item.get('category', 'Overall')

        tot['Overall'] += 1
        tot[cate] += 1

        if item.get('log') == 'Prefetch succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1

        if post_check(item):
            hit['Overall'] += 1
            hit[cate] += 1

        pred_raw = item.get("res", "")
        gt = item.get("answer", "").strip()  # noqa: F841
        pred_boxed = extract_final_answer_allform(str(pred_raw))
        flat_pred = [ans.strip() for group in pred_boxed for ans in (group if isinstance(group, list) else [group])]  # noqa: F841, E501

    res = defaultdict(list)
    for k in tot:
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100 if tot[k] else 0.0)
        res['acc'].append(hit[k] / tot[k] * 100 if tot[k] else 0.0)

    return pd.DataFrame(res).sort_values('Subject', ignore_index=True)
