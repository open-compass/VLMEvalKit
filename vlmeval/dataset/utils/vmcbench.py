import hashlib

import numpy as np
import pandas as pd


def _row_choices(row):
    all_choices = []
    for i in range(9):
        if chr(65 + i) in row and not pd.isna(row[chr(65 + i)]):
            all_choices.append(chr(65 + i))
    return all_choices


def _collect_candidates(response, all_choices, index2ans):
    """Collect answer candidates from ``response``.

    Returns ``(candidates, index_ans, ans_with_brack)`` exactly as the three
    stages of ``parse_multi_choice_response`` did; shared with the fallback
    counter in ``report_vmc_acc`` so the two cannot drift apart.
    """
    response = str(response)
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response or f'{choice}. ' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    return candidates, index_ans, ans_with_brack


def parse_multi_choice_response(response, all_choices, index2ans, row_key=None):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.

    ``row_key`` optionally identifies the current row (e.g., the question
    index). When nothing in the response can be parsed, the fallback guess is
    derived from a hash of ``(row_key, response)`` so that rescoring the same
    predictions file is deterministic (#1867). Passing ``row_key`` also keeps
    byte-identical responses (e.g., repeated refusal strings) from all
    collapsing onto the same guessed option.
    """
    candidates, index_ans, ans_with_brack = _collect_candidates(response, all_choices, index2ans)

    if len(candidates) == 0:  # still not get answer: deterministic guess.
        seed_src = f'{row_key}|{response}' if row_key is not None else response
        digest = hashlib.sha256(seed_src.encode('utf-8')).digest()
        pred_index = all_choices[int.from_bytes(digest[:8], 'big') % len(all_choices)]
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def get_mc_score(row, use_parse=True):
    if use_parse:
        if pd.isna(row["A"]):
            return False
        response = row["prediction"]
        all_choices = []
        for i in range(9):
            if chr(65 + i) in row and not pd.isna(row[chr(65 + i)]):
                all_choices.append(chr(65 + i))
        index2ans = {index: row[index] for index in all_choices}
        row_key = str(row['index']) if 'index' in row and not pd.isna(row['index']) else str(row.name)
        pred_index = parse_multi_choice_response(response, all_choices, index2ans, row_key=row_key)
    else:
        pred_index = row["output"]
    return int(pred_index == row["answer"])


def report_vmc_acc(data):
    general_datasets = ["SEEDBench", "MMStar", "A-OKVQA", "VizWiz", "MMVet", "VQAv2", "OKVQA"]
    reason_datasets = ["MMMU", "MathVista", "ScienceQA", "RealWorldQA", "GQA", "MathVision"]
    ocr_datasets = ["TextVQA", "OCRVQA"]
    doc_datasets = ["AI2D", "ChartQA", "DocVQA", "InfoVQA", "TableVQABench"]
    results = {}
    for category in data['category'].unique():
        results[category] = data[data['category'] == category]['hit'].mean()
    results = pd.DataFrame(results, index=[0])
    results["Overall"] = data['hit'].mean()
    # Surface how many rows fell back to a guessed option (#1867): those rows
    # contribute an expected +0.25 hit rate each, so the number is needed to
    # read the overall score.
    results['Unparsed'] = data.apply(
        lambda row: float(len(_collect_candidates(str(row['prediction']), _row_choices(row), {
            index: row[index] for index in _row_choices(row)})[0]) == 0), axis=1).mean()
    results['General'] = results[general_datasets].mean(axis=1)
    results['Reasoning'] = results[reason_datasets].mean(axis=1)
    results['OCR'] = results[ocr_datasets].mean(axis=1)
    results['Doc & Chart'] = results[doc_datasets].mean(axis=1)
    for key in results:
        results[key] = round(results[key] * 100, 2)
    results = results[['Overall', 'General', 'Reasoning', 'OCR', 'Doc & Chart', 'Unparsed']
                      + general_datasets + reason_datasets + ocr_datasets + doc_datasets]
    return results
