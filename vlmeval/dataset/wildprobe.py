import ast
import json
import os
import os.path as osp
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import d2df, dump, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

WILDPROBE_LLM_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process:**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `True`. Otherwise, conclude `False`.

**Output format:**

Return exactly two lines:

[reason] A concise reason for the judgment.
[judge] True

or

[reason] A concise reason for the judgment.
[judge] False

**Input:**

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}
"""


def _safe_model_name(model_name):
    return str(model_name).replace('/', '_').replace(':', '_')


def _json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def _coerce_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def _maybe_load_struct(value):
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _is_external_image_url(path):
    return str(path).startswith(('http://', 'https://', 'data:'))


def _resolve_image_path(path, base_dir):
    path = str(path)
    if osp.isabs(path) or _is_external_image_url(path):
        return path
    return str(Path(base_dir).expanduser() / path)


def _image_url_value(part):
    image_url = part.get('image_url')
    if isinstance(image_url, dict):
        return image_url.get('url', '')
    if isinstance(image_url, str):
        return image_url
    return part.get('value', '')


def _normalize_question(question, base_dir):
    question = _maybe_load_struct(question)
    if isinstance(question, dict):
        question = [question]
    if not isinstance(question, list):
        return [dict(type='text', value=_coerce_text(question))]

    messages = []
    for part in question:
        if not isinstance(part, dict):
            messages.append(dict(type='text', value=_coerce_text(part)))
            continue

        part_type = part.get('type')
        if part_type == 'text':
            messages.append(dict(type='text', value=_coerce_text(part.get('text', part.get('value', '')))))
        elif part_type == 'image_url':
            image_path = _image_url_value(part)
            messages.append(dict(type='image', value=_resolve_image_path(image_path, base_dir)))
        elif part_type == 'image':
            image_path = part.get('value', part.get('image', ''))
            messages.append(dict(type='image', value=_resolve_image_path(image_path, base_dir)))
        else:
            messages.append(dict(type='text', value=_coerce_text(part)))

    return [message for message in messages if message.get('value')]


def _extract_image_paths(messages):
    return [message['value'] for message in messages if message.get('type') == 'image' and message.get('value')]


def _normalize_dimension(dimension):
    dimension = _maybe_load_struct(dimension)
    if dimension is None or (isinstance(dimension, float) and pd.isna(dimension)):
        return []
    if isinstance(dimension, list):
        return [_coerce_text(x) for x in dimension]
    return [_coerce_text(dimension)]


def _extract_golden_answer(answer):
    answer = _maybe_load_struct(answer)
    if isinstance(answer, dict):
        for key in ('golden', 'golden_answer', 'ground_truth', 'groundtruth', 'gold', 'gt', 'answer'):
            if key in answer:
                return _coerce_text(answer[key])
    return _coerce_text(answer)


def _question_to_text(question):
    question = _maybe_load_struct(question)
    if not isinstance(question, list):
        return _coerce_text(question)

    parts = []
    for message in question:
        if not isinstance(message, dict):
            parts.append(_coerce_text(message))
            continue
        if message.get('type') == 'text':
            parts.append(_coerce_text(message.get('value', message.get('text', ''))))
        elif message.get('type') == 'image':
            parts.append(f"[Image: {message.get('value', '')}]")
        elif message.get('type') == 'image_url':
            parts.append(f"[Image: {_image_url_value(message)}]")
    return '\n'.join(part for part in parts if part)


def _build_judge_prompt(question, groundtruth, modeloutput):
    return WILDPROBE_LLM_JUDGE_PROMPT.format(
        question=_question_to_text(question),
        groundtruth=groundtruth,
        modeloutput=modeloutput,
    )


def _parse_judge_response(response):
    response = str(response).strip()
    judge_match = re.search(r'^\s*\[judge\]\s*(true|false)\s*$', response, flags=re.IGNORECASE | re.MULTILINE)
    if judge_match:
        return judge_match.group(1).lower() == 'true'

    response_lower = response.lower()
    if response_lower in {'true', 'false'}:
        return response_lower == 'true'
    return False


def _parse_judge_reason(response):
    response = str(response).strip()
    reason_match = re.search(r'^\s*\[reason\]\s*(.*?)\s*$', response, flags=re.IGNORECASE | re.MULTILINE)
    return reason_match.group(1) if reason_match else ''


class WildprobeDataset(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE_MODEL = 'gpt-5.2'
    DATASET_URL = {
        'wildprobe_multiimg_reasoning': 'WILDPROBE_MULTIIMG_REASONING_DATA_PATH',
    }
    DATASET_MD5 = {}
    force_use_dataset_prompt = True

    def __init__(self, dataset='wildprobe_multiimg_reasoning', **kwargs):
        super().__init__(dataset=dataset, skip_noimg=False)

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    @classmethod
    def _data_path(cls, dataset):
        if dataset not in cls.DATASET_URL:
            raise KeyError(f'Unsupported WildProbe dataset: {dataset}')
        env_var = cls.DATASET_URL[dataset]
        data_path = os.environ.get(env_var, '')
        if not data_path:
            raise EnvironmentError(
                f'Please set {env_var} to the WildProbe jsonl path before loading {dataset}.'
            )
        data_path = Path(data_path).expanduser()
        if not data_path.exists():
            raise FileNotFoundError(f'WildProbe jsonl does not exist: {data_path}')
        if data_path.suffix.lower() != '.jsonl':
            raise ValueError(f'WildProbe data path must be a jsonl file: {data_path}')
        return data_path

    def load_data(self, dataset):
        data_path = self._data_path(dataset)
        base_dir = data_path.parent
        records = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = _normalize_question(item.get('question', ''), base_dir)
                dimension = _normalize_dimension(item.get('dimension', []))
                image_paths = _extract_image_paths(messages)
                index = item.get('idx', item.get('index', line_no))

                record = {
                    'index': index,
                    'idx': item.get('idx', index),
                    'question': _json_dumps(messages),
                    'question_text': _question_to_text(messages),
                    'answer': _extract_golden_answer(item.get('answer', '')),
                    'dimension': _json_dumps(dimension),
                    'dimension_0': dimension[0] if dimension else '',
                }
                if image_paths:
                    record['image_path'] = image_paths[0] if len(image_paths) == 1 else image_paths

                for key, value in item.items():
                    if key in record or key in {'question', 'answer', 'dimension'}:
                        continue
                    record[key] = _json_dumps(value) if isinstance(value, (dict, list)) else value

                records.append(record)

        return pd.DataFrame(records)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return _maybe_load_struct(line['question'])

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) if pd.notna(x) else '' for x in data['prediction']]
        data['answer'] = [str(x) if pd.notna(x) else '' for x in data['answer']]

        judge_model = judge_kwargs.pop('model', self.DEFAULT_JUDGE_MODEL)
        safe_model = _safe_model_name(judge_model)
        storage = get_intermediate_file_path(eval_file, f'_{safe_model}_reason_judge')
        tmp_file = get_intermediate_file_path(eval_file, f'_{safe_model}_reason_judge_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{safe_model}_reason_score', 'json')
        nproc = judge_kwargs.pop('nproc', 4)

        if osp.exists(storage):
            judged = load(storage)
        else:
            judge_kwargs.setdefault('max_tokens', 65535)
            model = build_judge(model=judge_model, **judge_kwargs)
            if not model.working():
                raise RuntimeError('Judge model is not working properly. ' + DEBUG_MESSAGE)

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            lines = [data.iloc[i] for i in range(len(data))]
            indices = [line['index'] for line in lines]
            tasks = [(model, line) for line in lines if line['index'] not in ans]
            keys = [line['index'] for line in lines if line['index'] not in ans]
            if tasks:
                track_progress_rich(
                    self._evaluate_single,
                    tasks,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=keys,
                    save=tmp_file,
                )
                ans = load(tmp_file)

            judged = data.copy()
            judged['ModelResult'] = judged['prediction']
            judged['GroundTruth'] = judged['answer']
            judged['LLMJudgeResult'] = [ans[idx]['llm_judge_result'] for idx in indices]
            judged['JudgeReason'] = [ans[idx]['judge_reason'] for idx in indices]
            judged['JudgeResponse'] = [ans[idx]['judge_response'] for idx in indices]
            judged['JudgePrompt'] = [ans[idx]['judge_prompt'] for idx in indices]
            dump(judged, storage)

        metrics = self.compute_metrics(judged)
        dump(metrics, score_file)
        return metrics

    @staticmethod
    def _evaluate_single(model, line):
        question = line.get('question_text', line.get('question', ''))
        answer = str(line['answer'])
        prediction = str(line['prediction'])
        judge_prompt = _build_judge_prompt(
            question=question,
            groundtruth=answer,
            modeloutput=prediction,
        )
        judge_response = model.generate(judge_prompt)
        llm_judge_result = _parse_judge_response(judge_response)
        return {
            'judge_prompt': judge_prompt,
            'judge_response': judge_response,
            'judge_reason': _parse_judge_reason(judge_response),
            'llm_judge_result': llm_judge_result,
            'score': int(llm_judge_result),
        }

    @staticmethod
    def compute_metrics(results):
        total = len(results)
        correct = sum(1 for _, row in results.iterrows() if bool(row['LLMJudgeResult']))
        metrics = {'Overall': correct / total * 100 if total > 0 else 0.0}

        if 'dimension_0' in results:
            categories = sorted(str(x) for x in results['dimension_0'].dropna().unique())
        else:
            categories = sorted({
                (dimension[0] if dimension else '')
                for dimension in (_normalize_dimension(x) for x in results.get('dimension', []))
            })

        for category in categories:
            if not category:
                continue
            if 'dimension_0' in results:
                subset = results[results['dimension_0'].astype(str) == category]
            else:
                subset = results[
                    results['dimension'].apply(lambda value: (_normalize_dimension(value) or [''])[0] == category)
                ]
            metrics[f'Dimension/{category}'] = np.mean([bool(x) for x in subset['LLMJudgeResult']]) * 100

        return {k: round(float(v), 4) for k, v in metrics.items()}

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'Overall' in metrics:
            return {'Overall': metrics['Overall']}
        return super().report_primary_metric(metrics)

    @staticmethod
    def metrics_to_frame(metrics):
        return d2df(metrics)
