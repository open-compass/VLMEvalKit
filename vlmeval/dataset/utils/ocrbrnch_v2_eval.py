# flake8: noqa
import string
import re
import threading
from .Ocrbench_v2.vqa_metric import vqa_evaluation, cn_vqa_evaluation, math_expression_evaluation, vqa_evaluation_case_sensitive, counting_evaluation, cn_math_expression_evaluation
from .Ocrbench_v2.IoUscore_metric import vqa_with_position_evaluation, calculate_iou, extract_coordinates
from .Ocrbench_v2.TEDS_metric import TEDS, convert_markdown_table_to_html, convert_str_to_dict, convert_str_to_multi_dict, generate_combinations, dict_to_html, compute_f1_score, doc_parsing_evaluation, wrap_html_table
from .Ocrbench_v2.page_ocr_metric import cal_per_metrics
from .Ocrbench_v2.spotting_metric import extract_bounding_boxes_robust, spotting_evaluation
from .extractor import LLM_Extractor, LLM_VERIFIER
from vlmeval.smp import *

LOCK = threading.Lock()

MCQ_EXTRACT_PROMPT = 'Please extract the choice label (an uppercase character) from the response and directly output it. '
MCQ_VERIFIER = lambda x: isinstance(x, str) and len(x) == 1 and (x in string.ascii_uppercase)
QAP_VERIFY_PROMPT = """
Given the question: {question}, do the following two answers have exactly the same meaning?
Answer1: {answer}; Answer2: {prediction}. Please respond with 'Yes' or 'No'."""

def QAP_VERIFY_VERIFIER(resp):
    resp = resp.strip().lower()
    if 'yes' in resp and 'no' not in resp:
        return 1
    elif 'no' in resp and 'yes' not in resp:
        return 0
    else:
        return None


def is_nan_value(value):
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == 'nan':
        return True
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except:
        pass
    return False


def get_value_or_zero(value):
    return 0.0 if value is None else value


def calculate_average(scores_dict):
    averages = {key: sum(values) / len(values) for key, values in scores_dict.items() if len(values) > 0}
    return averages


# Handle MCQ, EN / CN VQA Eval, w. case sensitive
def basic_eval(model, data_item):
    score = None
    if "eval" in data_item.keys():
        if data_item["eval"] == "multiple choice":
            if not isinstance(data_item["answers"], list):
                data_item["answers"] = [data_item["answers"]]
            assert len(data_item["answers"]) == 1
            extractor = LLM_Extractor(model=model, prompt=MCQ_EXTRACT_PROMPT, verifier=MCQ_VERIFIER)
            predict = extractor.extract(data_item["predict"])
            if predict == data_item["answers"][0]:
                score = 1
            else:
                score = 0
        elif data_item["eval"] == "case sensitive":
            score = vqa_evaluation_case_sensitive(data_item["predict"], data_item["answers"])
        else:
            raise ValueError("No such evaluation method")
    else:
        cate = data_item['type']
        if cate.endswith(' en'):
            score = vqa_evaluation(data_item["predict"], data_item["answers"])
        elif cate.endswith(' cn'):
            score = cn_vqa_evaluation(data_item["predict"], data_item["answers"])
    return score


# 简答 Holisitic Eval, others LLM verifier
def handwritten_answer_extraction_eval(model, data_item):
    if "简答" in data_item["question"]:
        with LOCK:
            ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
        score = (
            get_value_or_zero(ocr_metric["bleu"]) +
            get_value_or_zero(ocr_metric["meteor"]) +
            get_value_or_zero(ocr_metric["f_measure"]) +
            (1 - get_value_or_zero(ocr_metric["edit_dist"]))
        ) / 4
    else:
        assert len(data_item["answers"]) == 1
        answer = data_item["answers"][0]
        prediction = data_item["predict"]
        veri = LLM_VERIFIER(model, QAP_VERIFY_PROMPT, QAP_VERIFY_VERIFIER)
        struct = {
            'question': data_item['question'],
            'answer': answer,
            'prediction': prediction
        }
        score = veri.verify(struct)
        if score is None:
            score = 0
    return score


# Math Expression Eval, EN / CN
def formula_recognition_eval(model, data_item):
    if data_item["type"] == "formula recognition cn":
        score = cn_math_expression_evaluation(data_item["predict"], data_item["answers"])
    elif data_item["type"] == 'formula recognition en':
        score = math_expression_evaluation(data_item["predict"], data_item["answers"])
    return score


# Counting Eval
def text_counting_eval(model, data_item):
    score = counting_evaluation(data_item["predict"], data_item["answers"], data_item["eval"])
    return score


# Table Parsing Eval based on TEDS
def table_parsing_eval(model, data_item):
    teds = TEDS(n_jobs=32)
    assert type(data_item["answers"])==list and len(data_item["answers"]) == 1, data_item['answers']
    if not isinstance(data_item["predict"], str):
        score = 0
    elif "html" in data_item["question"].lower():
        no_find = False
        predict_table = data_item["predict"].replace('\n','')
        if "<body" in predict_table:
            predict_table = re.findall('<body.*', predict_table)[0]
        elif "<table" in predict_table:
            predict_table = re.findall('<table.*', predict_table)[0]
        else:
            no_find = True

        if no_find:
            score = 0
        else:
            pred_table_html = wrap_html_table(predict_table)
            gold_table_html = wrap_html_table(data_item["answers"][0])
            try:
                score = teds.evaluate(pred_table_html, gold_table_html)
            except:
                score = 0

    elif "markdown" in data_item["question"].lower():
        pred_table_html = convert_markdown_table_to_html(data_item["predict"])
        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
        score = teds.evaluate(pred_table_html, gt_table_html)
    return score


# Chart Parsing Eval (Key Info to dict) based on TEDS
def chart_parsing_eval(model, data_item):
    teds = TEDS(n_jobs=32)
    answer = data_item["answers"][0]
    pred_chart_dict = convert_str_to_multi_dict(data_item["predict"])
    if len(pred_chart_dict) == 0:
        score = 0
    else:
        pred_chart_html = dict_to_html(pred_chart_dict)
        gt_chart_html = dict_to_html(answer)
        score = teds.evaluate(pred_chart_html, gt_chart_html)
    return score


# Doc Parsing Eval
def document_parsing_eval(model, data_item):
    assert type(data_item["answers"])==list and len(data_item["answers"]) == 1, data_item['answers']
    score = doc_parsing_evaluation(data_item["predict"], data_item["answers"][0])
    return score


# Key Information Extraction Eval, based on F1 score
def key_information_extraction_eval(model, data_item):
    assert len(data_item["answers"]) == 1, data_item['answers']
    answers = generate_combinations(data_item["answers"][0])

    if type(answers)==list and len(answers) == 1:
        if not isinstance(data_item["predict"], str):
            score = 0
        else:
            pred_kie_dict = convert_str_to_dict(data_item["predict"])
            score = compute_f1_score(pred_kie_dict, answers[0])
    else:
        max_score = 0
        for answer in answers:
            pred_kie_dict = convert_str_to_dict(data_item["predict"])
            score = compute_f1_score(pred_kie_dict, answer)
            max_score = max(max_score, score)
        score = max_score
    return score


# VQA w. Position Eval
def vqa_with_position_eval(model, data_item):
    pred_dict = convert_str_to_dict(data_item["predict"])
    score = vqa_with_position_evaluation(pred_dict, data_item)
    return score


# Text Translation Eval, w. holistic metrics
def holistic_metric_eval(model, data_item):
    assert len(data_item["answers"][0]) > 0, data_item['answers']
    with LOCK:
        ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
    score = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4
    return score


def text_grounding_eval(model, data_item):
    predict_bbox = extract_coordinates(data_item["predict"])
    if not predict_bbox:
        score = 0
    else:
        score = calculate_iou(predict_bbox, data_item["answers"])
    return score


def text_spotting_eval(model, data_item):
    predict_bbox = extract_bounding_boxes_robust(data_item["predict"])
    if not predict_bbox:
        score = 0
    else:
        score = spotting_evaluation(predict_bbox, data_item)
    return score


CATE_ROUTER = {
    'APP agent en': basic_eval,
    'ASCII art classification en': basic_eval,
    'math QA en': basic_eval,
    'reasoning VQA en': basic_eval,
    'science QA en': basic_eval,
    'text recognition en': basic_eval,
    'document classification en': basic_eval,
    'cognition VQA en': basic_eval,
    'diagram QA en': basic_eval,
    'cognition VQA cn': basic_eval,
    'reasoning VQA cn': basic_eval,
    'handwritten answer extraction cn': handwritten_answer_extraction_eval,
    'formula recognition cn': formula_recognition_eval,
    'formula recognition en': formula_recognition_eval,
    'text counting en': text_counting_eval,
    'table parsing en': table_parsing_eval,
    'table parsing cn': table_parsing_eval,
    'chart parsing en': chart_parsing_eval,
    'document parsing en': document_parsing_eval,
    'document parsing cn': document_parsing_eval,
    'key information extraction en': key_information_extraction_eval,
    'key information mapping en': key_information_extraction_eval,
    'key information extraction cn': key_information_extraction_eval,
    'VQA with position en': vqa_with_position_eval,
    'text translation cn': holistic_metric_eval,
    'fine-grained text recognition en': holistic_metric_eval,
    'full-page OCR en': holistic_metric_eval,
    'full-page OCR cn': holistic_metric_eval,
    'text grounding en': text_grounding_eval,
    'text spotting en': text_spotting_eval,
}


def evaluate_item_tup(model, data_item):
    assert data_item['type'] in CATE_ROUTER, data_item
    score_fn = CATE_ROUTER[data_item['type']]
    score = score_fn(model, data_item)
    if score < 0 or score > 1:
        import warnings
        warnings.warn(f'Score {score} of data_item {data_item} is out of range [0, 1], will clip. ')
        score = np.clip(score, 0, 1)
    return score


def process_predictions(model, predict_file, nproc=16):
    input_tups = [dict(model=model, data_item=data_item) for data_item in predict_file]
    scores = track_progress_rich(
        evaluate_item_tup,
        input_tups,
        nproc=nproc,
        desc='Evaluating OCRBench_v2'
    )
    return scores


def ocrbench_v2_aggregate_accuracy(data_list):
    en_text_recognition_list, en_text_detection_list, en_text_spotting_list, en_relationship_extraction_list = [], [], [], []
    en_element_parsing_list, en_mathematical_calculation_list, en_visual_text_understanding_list = [], [], []
    en_knowledge_reasoning_list = []

    cn_text_recognition_list, cn_relationship_extraction_list = [], []
    cn_element_parsing_list, cn_visual_text_understanding_list = [], []
    cn_knowledge_reasoning_list = []

    res_list = []
    for item in data_list:
        if "ignore" in item.keys():
            assert item["ignore"] == "True"

        elif item["type"] == "text recognition en" or item["type"] == "fine-grained text recognition en" or item["type"] == "full-page OCR en":
            en_text_recognition_list.append(item["score"])

        elif item["type"] == "text grounding en" or item["type"] == "VQA with position en":
            en_text_detection_list.append(item["score"])

        elif item["type"] == "text spotting en":
            en_text_spotting_list.append(item["score"])

        elif item["type"] == "key information extraction en" or item["type"] == "key information mapping en":
            en_relationship_extraction_list.append(item["score"])

        elif item["type"] == "document parsing en" or item["type"] == "chart parsing en" \
        or item["type"] == "table parsing en" or item["type"] == "formula recognition en":
            en_element_parsing_list.append(item["score"])

        elif item["type"] == "math QA en" or item["type"] == "text counting en":
            en_mathematical_calculation_list.append(item["score"])

        elif item["type"] == "document classification en" \
        or item["type"] == "cognition VQA en" or item["type"] == "diagram QA en":
            en_visual_text_understanding_list.append(item["score"])

        elif item["type"] == "reasoning VQA en" or item["type"] == "science QA en" \
        or item["type"] == "APP agent en" or item["type"] == "ASCII art classification en":
            en_knowledge_reasoning_list.append(item["score"])

        elif item["type"] == "full-page OCR cn":
            cn_text_recognition_list.append(item["score"])

        elif item["type"] == "key information extraction cn" or item["type"] == "handwritten answer extraction cn":
            cn_relationship_extraction_list.append(item["score"])

        elif item["type"] == "document parsing cn" or item["type"] == "table parsing cn" or item["type"] == "formula recognition cn":
            cn_element_parsing_list.append(item["score"])

        elif item["type"] == "cognition VQA cn":
            cn_visual_text_understanding_list.append(item["score"])

        elif item["type"] == "reasoning VQA cn" or item["type"] == "text translation cn":
            cn_knowledge_reasoning_list.append(item["score"])

        else:
            raise ValueError("Unknown task type!")

    en_scores = {
        "en_text_recognition": en_text_recognition_list,
        "en_text_detection": en_text_detection_list,
        "en_text_spotting": en_text_spotting_list,
        "en_relationship_extraction": en_relationship_extraction_list,
        "en_element_parsing": en_element_parsing_list,
        "en_mathematical_calculation": en_mathematical_calculation_list,
        "en_visual_text_understanding": en_visual_text_understanding_list,
        "en_knowledge_reasoning": en_knowledge_reasoning_list
    }

    cn_scores = {
        "cn_text_recognition": cn_text_recognition_list,
        "cn_relationship_extraction": cn_relationship_extraction_list,
        "cn_element_parsing": cn_element_parsing_list,
        "cn_visual_text_understanding": cn_visual_text_understanding_list,
        "cn_knowledge_reasoning": cn_knowledge_reasoning_list
    }

    en_averages = calculate_average(en_scores)
    cn_averages = calculate_average(cn_scores)

    return en_averages,cn_averages
