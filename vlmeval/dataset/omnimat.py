import ast
import json
import os
import re
import warnings
from decimal import Decimal, InvalidOperation, getcontext, localcontext
from pathlib import Path
from typing import Any

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load, toliststr
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import build_judge

getcontext().prec = 50


class OmniMat(ImageBaseDataset):
    TYPE = 'VQA'
    DEFAULT_JUDGE = 'gemini-2.5-flash'

    DATASET_URL = {
        'OmniMat_QA': '',
        'OmniMat_CAL': '',
    }

    QA_PROMPT = (
        'You are a materials science expert. Answer the question clearly and accurately.\n\n'
        'Return both sections exactly:\n'
        'step-by-step reasoning\n'
        '<answer>a concise final answer that directly addresses the question</answer>\n\n'
        'Question:\n{question}'
    )

    CAL_PROMPT = (
        'You are a materials science expert. Solve the calculation accurately.\n\n'
        'Return both sections exactly:\n'
        'step-by-step reasoning\n'
        '<answer>ONLY the final answer content</answer>\n\n'
        'If final_answer_format is provided, the answer inside <answer> must match its '
        'grouping, order, and number of slots exactly. Replace each empty string slot '
        'with one final number, expression, or formula only.\n\n'
        'Question:\n{question}'
    )

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def load_data(self, dataset):
        candidates = [
            Path(LMUDataRoot()) / f'{dataset}.tsv',
            Path.cwd() / f'{dataset}.tsv',
            Path(__file__).resolve().parents[2] / f'{dataset}.tsv',
        ]
        for path in candidates:
            if path.exists():
                self.data_path = str(path)
                data = load(str(path))
                if 'image' in data:
                    data['image'] = data['image'].fillna('')
                if 'image_path' in data:
                    data['image_path'] = data['image_path'].fillna('')
                _normalize_ids(data)
                return data

        raise FileNotFoundError(
            f'{dataset}.tsv was not found. Run `python scripts/convert_omnimat.py` '
            'from the repository root first, or place the TSV under LMUData.'
        )

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_paths = self.dump_image(line) if _has_image_data(line) else _line_image_paths(line)
        if self.dataset_name == 'OmniMat_QA':
            prompt = self.QA_PROMPT.format(question=line['question'])
        elif self.dataset_name == 'OmniMat_CAL':
            prompt = self.CAL_PROMPT.format(question=line['question'])
            final_answer_format = _parse_json_value(line.get('final_answer_format'), default=None)
            if final_answer_format is not None:
                prompt += (
                    '\n\n--- Final Answer Format Constraint ---\n'
                    f'final_answer_format: {json.dumps(final_answer_format, ensure_ascii=False)}\n'
                    'Inside <answer>, output only the filled answer structure.'
                )
        else:
            raise NotImplementedError(self.dataset_name)

        msgs = [dict(type='image', value=path) for path in image_paths]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        if self.dataset_name == 'OmniMat_QA':
            return self.evaluate_qa(eval_file, **judge_kwargs)
        if self.dataset_name == 'OmniMat_CAL':
            return self.evaluate_cal(eval_file, **judge_kwargs)
        raise NotImplementedError(self.dataset_name)

    def evaluate_qa(self, eval_file, **judge_kwargs):
        eval_file = str(eval_file)
        data = _ensure_dataframe(load(eval_file))
        _normalize_ids(data)
        assert 'prediction' in data

        data['llm_answer'] = [extract_answer_text(x) for x in data['prediction']]
        items = [_qa_item(data.iloc[i]) for i in range(len(data))]

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_model = judge_kwargs.pop('model', self.DEFAULT_JUDGE)
        if judge_model == 'exact_matching':
            raise ValueError('OmniMat_QA requires an LLM judge; exact_matching is not supported.')
        judge = build_judge(model=judge_model, **judge_kwargs)

        precision_file = get_intermediate_file_path(eval_file, '_omnimat_precision', 'pkl')
        recall_file = get_intermediate_file_path(eval_file, '_omnimat_recall', 'pkl')
        precision = _run_cached_judge(precision_file, omnimat_qa_precision, judge, items, nproc)
        recall = _run_cached_judge(recall_file, omnimat_qa_recall, judge, items, nproc)

        data['precision'] = [precision[item['index']]['score'] for item in items]
        data['recall'] = [recall[item['index']]['weighted_score'] for item in items]
        data['f1'] = [_f1_score(p, r) for p, r in zip(data['precision'], data['recall'])]
        data['precision_details'] = [
            json.dumps(precision[item['index']].get('precision_details', {}), ensure_ascii=False)
            for item in items
        ]
        data['recall_details'] = [
            json.dumps(recall[item['index']].get('eval_details', {}), ensure_ascii=False)
            for item in items
        ]

        detail_file = get_intermediate_file_path(eval_file, '_omnimat_qa_details', 'csv')
        dump(data, detail_file)

        summary = _qa_summary(data)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(summary, score_file)
        return summary

    def evaluate_cal(self, eval_file, **judge_kwargs):
        _ = judge_kwargs
        eval_file = str(eval_file)
        data = _ensure_dataframe(load(eval_file))
        _normalize_ids(data)
        assert 'prediction' in data
        _warn_if_missing_answers(data)

        scored_rows = [_score_cal_line(data.iloc[i]) for i in range(len(data))]
        scored = pd.DataFrame(scored_rows)
        detail = pd.concat([data.reset_index(drop=True), scored], axis=1)
        detail_file = get_intermediate_file_path(eval_file, '_omnimat_cal_details', 'csv')
        dump(detail, detail_file)

        summary = _cal_summary(detail)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(summary, score_file)
        return summary

    @classmethod
    def report_primary_metric(cls, metrics: dict | None) -> dict:
        if not isinstance(metrics, dict):
            return {}
        primary_candidates = (
            (
                'Overall F1(Macro)',
                (
                    'Category=Overall|F1(Macro)',
                    'Category ID=Overall|Category=Overall|F1(Macro)',
                    'Overall F1(Macro)',
                ),
            ),
            (
                'Overall Slot Accuracy Exact',
                (
                    'Category=Overall|Slot Accuracy Exact',
                    'Category ID=Overall|Category=Overall|Slot Accuracy Exact',
                    'Overall Slot Accuracy Exact',
                ),
            ),
            (
                'Overall Slot Accuracy Threshold',
                (
                    'Category=Overall|Slot Accuracy Threshold',
                    'Category ID=Overall|Category=Overall|Slot Accuracy Threshold',
                    'Overall Slot Accuracy Threshold',
                ),
            ),
        )
        for primary_name, keys in primary_candidates:
            for key in keys:
                if key in metrics:
                    return {primary_name: metrics[key]}
        return super().report_primary_metric(metrics)


def _ensure_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def _format_id(value: Any, width: int) -> str:
    if _is_missing(value):
        return ''
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text.zfill(width) if text.isdigit() else text


def _normalize_ids(data: pd.DataFrame) -> None:
    if 'category_id' in data:
        data['category_id'] = [_format_id(value, 2) for value in data['category_id']]
    if 'id' in data:
        data['id'] = [_format_id(value, 3) for value in data['id']]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_json_value(value: Any, default: Any):
    if _is_missing(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none'}:
        return default
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return default


def _line_image_paths(line) -> list[str]:
    if 'image_path' not in line or _is_missing(line.get('image_path')):
        return []
    value = line.get('image_path')
    try:
        paths = toliststr(value)
    except Exception:
        paths = [str(value)]
    return [path for path in paths if str(path).strip()]


def _has_image_data(line) -> bool:
    if 'image' not in line or _is_missing(line.get('image')):
        return False
    value = line.get('image')
    if isinstance(value, list):
        return any(not _is_missing(item) and str(item).strip() for item in value)
    return bool(str(value).strip())


def extract_answer_text(response: Any) -> str:
    text = '' if response is None else str(response)
    match = re.search(r'<answer>(.*?)</answer>', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _qa_item(line) -> dict[str, Any]:
    return {
        'index': str(line['index']),
        'id': str(line.get('id', line['index'])),
        'category_id': str(line.get('category_id', '')),
        'category': str(line.get('category', '')),
        'question': str(line.get('question', '')),
        'gt_answer': str(line.get('answer', '')),
        'llm_answer': str(line.get('llm_answer', '')),
        'key_points': _parse_json_value(line.get('key_points'), default=[]),
        'scoring_weights': _parse_json_value(line.get('scoring_weights'), default={}),
    }


def _key_point_lines(item: dict[str, Any]) -> list[str]:
    key_points = item.get('key_points', [])
    if key_points and isinstance(key_points, list) and isinstance(key_points[0], dict):
        return [
            f"[{kp.get('id', '')}] {kp.get('description', '')}"
            for kp in key_points
        ]
    return [str(value) for value in key_points]


def _precision_prompt(item: dict[str, Any]) -> str:
    formatted_points = '\n'.join(
        f'{index}. {point}' for index, point in enumerate(_key_point_lines(item), start=1)
    )
    return f"""
You are a rigorous, fair, and professional benchmark evaluator.

Your task is to calculate the Precision of the model answer:
Precision = TP / (TP + FP)

Definitions:
- TP: a specific information unit in the model answer that directly matches a key scoring point.
- FP: irrelevant, incorrect, redundant, or filler information in the model answer.
- Missed key points are false negatives and do not affect Precision.

Ground Truth Answer:
{item['gt_answer']}

Key Scoring Points:
{formatted_points}

Model Answer:
{item['llm_answer']}

Return only one JSON object with exactly these keys:
{{
  "tp_string": "one TP per line",
  "fp_string": "one FP per line, include [FP-Type] when possible"
}}
"""


def _recall_prompt(item: dict[str, Any]) -> str:
    key_points = item.get('key_points', [])
    formatted_points = '\n'.join(
        f"{index}. [{kp.get('id', '')}] {kp.get('description', '')}"
        for index, kp in enumerate(key_points, start=1)
        if isinstance(kp, dict)
    )
    return f"""
You are a strict and meticulous grader specializing in materials science.

Evaluate each Key Scoring Point sequentially.

For each point:
- met: 1 if the model answer clearly covers the point; otherwise 0.
- quality_score: if met is 0, use 0.0. If met is 1, use 1.0 for excellent,
  0.5 for acceptable but shallow/imprecise, and 0.1 for poor or partly wrong.

Return only one JSON object with exactly these keys:
{{
  "met": [0 or 1 for each key point],
  "quality_score": [float score for each key point],
  "reasoning": "brief critical explanation"
}}

Key Scoring Points:
{formatted_points}

Ground Truth Answer:
{item['gt_answer']}

Model Answer:
{item['llm_answer']}
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            return json.loads(match.group(0))
        raise


def _split_nonempty_lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value or '').splitlines() if line.strip()]


def omnimat_qa_precision(model, item: dict[str, Any]) -> dict[str, Any]:
    try:
        response = model.generate(_precision_prompt(item))
        parsed = _extract_json_object(response)
        tp_list = _split_nonempty_lines(parsed.get('tp_string', ''))
        fp_list = _split_nonempty_lines(parsed.get('fp_string', ''))
        denom = len(tp_list) + len(fp_list)
        score = len(tp_list) / denom if denom else 0.0
        return {
            'score': round(score, 3),
            'judge_response': response,
            'precision_details': {
                'tp_list': tp_list,
                'fp_list': fp_list,
                'counts': {'count_tp': len(tp_list), 'count_fp': len(fp_list)},
            },
        }
    except Exception as exc:
        return {'score': 0.0, 'error': f'{type(exc).__name__}: {exc}'}


def _compute_weighted_score(item: dict[str, Any], quality_scores: list[Any]) -> float:
    key_points = item.get('key_points', [])
    weights = item.get('scoring_weights', {}) or {}
    if not key_points:
        return 0.0
    if not weights:
        weights = {
            kp.get('id', str(index)): 1.0 / len(key_points)
            for index, kp in enumerate(key_points)
            if isinstance(kp, dict)
        }

    total = 0.0
    for index, kp in enumerate(key_points):
        if not isinstance(kp, dict) or index >= len(quality_scores):
            continue
        try:
            quality = float(quality_scores[index])
        except (TypeError, ValueError):
            quality = 0.0
        total += quality * float(weights.get(kp.get('id', ''), 0.0))
    return round(total, 4)


def omnimat_qa_recall(model, item: dict[str, Any]) -> dict[str, Any]:
    try:
        response = model.generate(_recall_prompt(item))
        parsed = _extract_json_object(response)
        quality_scores = parsed.get('quality_score', [])
        return {
            'weighted_score': _compute_weighted_score(item, quality_scores),
            'judge_response': response,
            'eval_details': parsed,
        }
    except Exception as exc:
        return {'weighted_score': 0.0, 'error': f'{type(exc).__name__}: {exc}'}


def _run_cached_judge(cache_file: str, func, judge, items: list[dict[str, Any]], nproc: int) -> dict[str, dict]:
    cache = load(cache_file) if os.path.exists(cache_file) else {}
    pending = [item for item in items if item['index'] not in cache]
    if pending:
        keys = [item['index'] for item in pending]
        tasks = [(judge, item) for item in pending]
        results = track_progress_rich(func, tasks, nproc=nproc, save=cache_file, keys=keys)
        for key, result in zip(keys, results):
            cache[key] = result
    return cache


def _f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _qa_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category_id, sub in sorted(data.groupby('category_id'), key=lambda item: item[0]):
        rows.append(_qa_summary_row(str(category_id), sub['category'].iloc[0], sub))
    rows.append(_qa_overall_summary_row(data, rows))
    return pd.DataFrame(rows)


def _qa_summary_row(category_id: str, category: str, data: pd.DataFrame) -> dict[str, Any]:
    precision = float(data['precision'].mean()) if len(data) else 0.0
    recall = float(data['recall'].mean()) if len(data) else 0.0
    f1_macro = float(data['f1'].mean()) if len(data) else 0.0
    return {
        'Category ID': category_id,
        'Category': category,
        'Items': len(data),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1(Agg)': round(_f1_score(precision, recall), 4),
        'F1(Macro)': round(f1_macro, 4),
    }


def _qa_overall_summary_row(data: pd.DataFrame, category_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'Category ID': 'Overall',
        'Category': 'Overall',
        'Items': len(data),
        'Precision': _mean_row_metric(category_rows, 'Precision'),
        'Recall': _mean_row_metric(category_rows, 'Recall'),
        'F1(Agg)': _mean_row_metric(category_rows, 'F1(Agg)'),
        'F1(Macro)': _mean_row_metric(category_rows, 'F1(Macro)'),
    }


def _mean_row_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


def clean_text(value: Any) -> str:
    text = '' if value is None else str(value)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = text.replace('$', '')
    text = text.replace(r'\left', '').replace(r'\right', '')
    text = text.replace(r'\,', '').replace(' ', '')
    text = text.replace(',', '')
    text = text.replace('−', '-').replace('–', '-').replace('—', '-')
    text = text.replace('×', 'x').replace(r'\times', 'x')
    text = text.replace(r'\cdot', 'x').replace('∙', 'x').replace('·', 'x')
    text = re.sub(r'(?<!\\)frac', r'\\frac', text)
    text = re.sub(r'\\(?:mathrm|operatorname|text|textrm|mathbf|mathit|mathrmbf)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\boxed\{([^{}]*)\}', r'\1', text)
    return text


def to_decimal(value: Any) -> Decimal | None:
    text = clean_text(value)
    if not text:
        return None

    for pattern in (r'[+-]?\d+(\.\d+)?', r'[+-]?\d+(\.\d+)?[eE][+-]?\d+'):
        if re.fullmatch(pattern, text):
            try:
                return Decimal(text)
            except InvalidOperation:
                return None

    match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)x10\^?\{?([+-]?\d+)\}?', text)
    if match:
        return Decimal(match.group(1)) * (Decimal(10) ** int(match.group(2)))

    match = re.fullmatch(r'10\^?\{?([+-]?\d+)\}?', text)
    if match:
        return Decimal(10) ** int(match.group(1))

    match = re.fullmatch(r'\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}', text)
    if match:
        numerator = Decimal(match.group(1))
        denominator = Decimal(match.group(2))
        return None if denominator == 0 else numerator / denominator

    match = re.fullmatch(r'([+-])?(\d+)\\frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}', text)
    if match:
        sign = -1 if match.group(1) == '-' else 1
        integer = Decimal(match.group(2))
        numerator = Decimal(match.group(3))
        denominator = Decimal(match.group(4))
        return None if denominator == 0 else Decimal(sign) * (integer + numerator / denominator)

    match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)', text)
    if match:
        numerator = Decimal(match.group(1))
        denominator = Decimal(match.group(2))
        return None if denominator == 0 else numerator / denominator

    return None


def numeric_equal_exact(gt: Any, pred: Any) -> int:
    gt_decimal = to_decimal(gt)
    pred_decimal = to_decimal(pred)
    if gt_decimal is None or pred_decimal is None:
        return 0

    match = re.search(r'\.(\d+)', str(gt).strip())
    decimals = len(match.group(1)) if match else 0
    digits = max(len(gt_decimal.as_tuple().digits), len(pred_decimal.as_tuple().digits))
    with localcontext() as ctx:
        ctx.prec = max(getcontext().prec, digits + decimals + 2)
        pred_rounded = pred_decimal.quantize(Decimal(f'1e-{decimals}'))
    return 1 if gt_decimal == pred_rounded else 0


def numeric_equal_threshold(gt: Any, pred: Any, *, rel_tol: float, zero_tol: float) -> int:
    gt_decimal = to_decimal(gt)
    pred_decimal = to_decimal(pred)
    if gt_decimal is None or pred_decimal is None:
        return 0

    diff = abs(gt_decimal - pred_decimal)
    max_abs = max(abs(gt_decimal), abs(pred_decimal))
    if max_abs == 0:
        return 1
    if abs(gt_decimal) == 0 or abs(pred_decimal) == 0:
        return 1 if diff <= Decimal(str(zero_tol)) else 0
    return 1 if diff <= Decimal(str(rel_tol)) * max_abs else 0


def text_equal(gt: Any, pred: Any) -> int:
    return 1 if clean_text(gt) == clean_text(pred) else 0


def flatten_answer(value: Any) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_answer(item))
        return out
    return [str(value).strip()]


def parse_prediction_string(text: str) -> list[str]:
    text = (text or '').strip()
    if not text:
        return []

    boxed = re.search(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        return [boxed.group(1).strip()]

    try:
        return flatten_answer(json.loads(text))
    except Exception:
        pass

    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        inner = answer_match.group(1).strip()
        try:
            return flatten_answer(json.loads(inner))
        except Exception:
            return [inner]

    if '</think>' in text:
        return parse_prediction_string(text.split('</think>')[-1].strip())
    return [text]


def normalize_prediction(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return flatten_answer(raw)
    return parse_prediction_string(str(raw))


def _score_cal_line(line, rel_tol: float = 0.1, zero_tol: float = 1e-12) -> dict[str, Any]:
    gt_answers = flatten_answer(_parse_json_value(line.get('final_answer_list'), default=[]))
    pred_answers = normalize_prediction(line.get('prediction', ''))

    pair_count = max(len(gt_answers), len(pred_answers))
    slot_exact = []
    slot_threshold = []
    for index in range(pair_count):
        gt = gt_answers[index] if index < len(gt_answers) else ''
        pred = pred_answers[index] if index < len(pred_answers) else ''
        try:
            exact = numeric_equal_exact(gt, pred)
        except InvalidOperation:
            slot_exact.append(0)
            slot_threshold.append(0)
            continue
        if exact == 0:
            exact = text_equal(gt, pred)
        slot_exact.append(exact)

        threshold = numeric_equal_threshold(gt, pred, rel_tol=rel_tol, zero_tol=zero_tol)
        if threshold == 0:
            threshold = text_equal(gt, pred)
        slot_threshold.append(threshold)

    exact_match = int(bool(gt_answers) and len(gt_answers) == len(pred_answers) and all(slot_exact))
    threshold_match = int(bool(gt_answers) and len(gt_answers) == len(pred_answers) and all(slot_threshold))
    return {
        'pred_answer': json.dumps(pred_answers, ensure_ascii=False),
        'gt_answer_list': json.dumps(gt_answers, ensure_ascii=False),
        'slot_scores_exact': json.dumps(slot_exact, ensure_ascii=False),
        'slot_scores_threshold': json.dumps(slot_threshold, ensure_ascii=False),
        'score_exact': exact_match,
        'score_threshold': threshold_match,
    }


def _cal_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category_id, sub in sorted(data.groupby('category_id'), key=lambda item: item[0]):
        rows.append(_cal_summary_row(str(category_id), sub['category'].iloc[0], sub))
    rows.append(_cal_summary_row('Overall', 'Overall', data))
    return pd.DataFrame(rows)


def _slot_acc(series: pd.Series) -> float:
    total = 0
    correct = 0
    for value in series:
        scores = _parse_json_value(value, default=[])
        total += len(scores)
        correct += sum(int(x) for x in scores)
    return correct / total if total else 0.0


def _cal_summary_row(category_id: str, category: str, data: pd.DataFrame) -> dict[str, Any]:
    total = len(data)
    correct_exact = int(data['score_exact'].sum()) if total else 0
    correct_threshold = int(data['score_threshold'].sum()) if total else 0
    return {
        'Category ID': category_id,
        'Category': category,
        'Items': total,
        'Correct Exact': correct_exact,
        'Accuracy Exact': round(correct_exact / total if total else 0.0, 4),
        'Slot Accuracy Exact': round(_slot_acc(data['slot_scores_exact']), 4),
        'Correct Threshold': correct_threshold,
        'Accuracy Threshold': round(correct_threshold / total if total else 0.0, 4),
        'Slot Accuracy Threshold': round(_slot_acc(data['slot_scores_threshold']), 4),
    }


def _warn_if_missing_answers(data: pd.DataFrame) -> None:
    missing = data['final_answer_list'].apply(lambda value: not _parse_json_value(value, default=[]))
    if missing.any():
        warnings.warn(f'OmniMat_CAL has {int(missing.sum())} rows without final_answer_list.')
