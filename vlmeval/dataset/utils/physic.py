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
        "You MUST put the final answer in the `\\boxed{}` environment.\n"
        "This applies even if the answer is a text explanation like \"The singlet state is lower in energy.\"\n"
        "Do NOT include multiple boxes.\n"
        "Do NOT include \\boxed anywhere else in your reasoning.\n"
        "The box must appear on the last line of the response.\n\n"
        "WARNING: DO NOT forget to include \boxed{} with the final answer. Responses without it will be considered INVALID.\n\n"  # noqa: E501
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


def PHYSIC_auxeval(model, line):
    equiv_data = {}
    try:
        response = line['prediction']
        if not response or not isinstance(response, str):
            equiv_data['LOG'] = 'Invalid response format, returning False.'
            return dict(log=equiv_data, res=False)

        pred_boxed = extract_final_answer_allform(response)
        gt = line['answer'].strip()

        flat_preds = [item.strip() for group in pred_boxed for item in (group if isinstance(group, list) else [group])]

        if gt in flat_preds:
            equiv_data['LOG'] = 'GT found in prediction, returning True.'
            return dict(log=equiv_data, res=True)

        for pred in flat_preds:
            equiv_data = is_equiv(model, pred, gt)
            if equiv_data['llm_result']:
                equiv_data['LOG'] = 'Equivalence found, returning True.'
                return dict(log=equiv_data, res=True)

        equiv_data['LOG'] = 'No equivalence found, returning False.'
        return dict(log=equiv_data, res=False)
    except Exception as e:
        logging.warning(f'post_check error: {e}')
        equiv_data['LOG'] = f'Exception occurred: {e}'
        return dict(log=equiv_data, res=False)


def PHYSIC_acc(result_file):
    data = load(result_file)
    tot = defaultdict(int)
    hit = defaultdict(int)
    lt = len(data)

    for i in tqdm(range(lt)):
        item = data.iloc[i]
        cate = item.get('category', 'Overall')

        tot['Overall'] += 1
        tot[cate] += 1

        if item.get('res'):
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
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100 if tot[k] else 0.0)

    return pd.DataFrame(res).sort_values('Subject', ignore_index=True)
