import json
import os.path as osp
import re

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge
from .utils.llavabench import build_prompt as build_llavabench_prompt

HERON_BENCH_DATASET = 'Japanese-Heron-Bench'
HERON_BENCH_REPO = 'turing-motors/Japanese-Heron-Bench'
HERON_BENCH_REVISION = 'dabd13314f2eac26204b982ec098049fdbbd1322'
HERON_BENCH_SAMPLES = 103
HERON_CATEGORIES = ('conv', 'detail', 'complex')
HERON_SYSTEM_PROMPT = 'You are a helpful and precise assistant for checking the quality of the answer.'
HERON_SCORE_PATTERN = re.compile(
    r'^\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:,|\s)\s*'
    r'(-?(?:\d+(?:\.\d*)?|\.\d+))\s*$'
)


def _load_jsonl(path):
    records = []
    with open(path, encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f'Invalid JSON in {path} at line {line_number}') from err
    return records


def _safe_model_name(model_name):
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', str(model_name))


def parse_heron_score(review):
    """Parse the official two-score first line returned by the judge."""
    first_line = str(review).strip().splitlines()[0] if str(review).strip() else ''
    match = HERON_SCORE_PATTERN.fullmatch(first_line)
    if match is None:
        return [-1.0, -1.0]

    scores = [float(match.group(1)), float(match.group(2))]
    if not all(1 <= score <= 10 for score in scores):
        return [-1.0, -1.0]
    return scores


def build_heron_judge_prompt(line):
    """Build the pairwise prompt used by the official Heron evaluator."""
    prompt = build_llavabench_prompt(line)
    return (
        prompt
        + 'If it is not relevant to the context, does not answer directly, or says the wrong thing, '
        + 'give it a low score.\n\n'
    )


def heron_bench_atomeval(model, prompt):
    return parse_heron_score(model.generate(prompt))


def heron_bench_score(data):
    """Aggregate category-relative scores following the official Heron protocol."""
    required = {'category', 'gpt4_score', 'score'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'Heron-Bench score data is missing columns: {sorted(missing)}')

    rows = []
    for category in HERON_CATEGORIES:
        category_data = data[data['category'] == category]
        valid = category_data[
            category_data['gpt4_score'].between(1, 10)
            & category_data['score'].between(1, 10)
        ]
        parse_errors = len(category_data) - len(valid)
        if len(valid):
            reference_score = float(valid['gpt4_score'].mean() * 10)
            model_score = float(valid['score'].mean() * 10)
            relative_score = float(model_score / reference_score * 100)
        else:
            reference_score = np.nan
            model_score = np.nan
            relative_score = np.nan
        rows.append({
            'split': category,
            'Relative Score (main)': relative_score,
            'VLM Score': model_score,
            'GPT4 Score': reference_score,
            'Valid Samples': len(valid),
            'Parse Errors': parse_errors,
        })

    valid_rows = [row for row in rows if not np.isnan(row['Relative Score (main)'])]
    overall = {
        'split': 'overall',
        'Relative Score (main)': (
            float(np.mean([row['Relative Score (main)'] for row in valid_rows]))
            if valid_rows else np.nan
        ),
        'VLM Score': (
            float(np.mean([row['VLM Score'] for row in valid_rows]))
            if valid_rows else np.nan
        ),
        'GPT4 Score': (
            float(np.mean([row['GPT4 Score'] for row in valid_rows]))
            if valid_rows else np.nan
        ),
        'Valid Samples': sum(row['Valid Samples'] for row in rows),
        'Parse Errors': sum(row['Parse Errors'] for row in rows),
    }
    return pd.DataFrame([overall] + rows)


class HeronBench(ImageBaseDataset):
    """Japanese Heron-Bench VQA dataset and LLM-as-a-judge evaluation.

    Paper: https://arxiv.org/abs/2404.07824
    Dataset: https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench
    The downloaded images retain the per-item terms documented in ``LICENCE.md``.
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE = 'gpt-4o'
    DATASET_URL = {
        HERON_BENCH_DATASET: f'https://huggingface.co/datasets/{HERON_BENCH_REPO}',
    }

    @classmethod
    def supported_datasets(cls):
        return [HERON_BENCH_DATASET]

    def load_data(self, dataset):
        if dataset != HERON_BENCH_DATASET:
            raise ValueError(f'Unsupported Heron-Bench dataset: {dataset}')

        dataset_path = snapshot_download(
            repo_id=HERON_BENCH_REPO,
            repo_type='dataset',
            revision=HERON_BENCH_REVISION,
            allow_patterns=[
                'questions_ja.jsonl',
                'answers_gpt4.jsonl',
                'images/*.jpg',
                'LICENCE.md',
                'README.md',
            ],
        )
        question_path = osp.join(dataset_path, 'questions_ja.jsonl')
        answer_path = osp.join(dataset_path, 'answers_gpt4.jsonl')
        self.data_path = question_path

        questions = _load_jsonl(question_path)
        answers = _load_jsonl(answer_path)
        if len(questions) != HERON_BENCH_SAMPLES or len(answers) != HERON_BENCH_SAMPLES:
            raise ValueError(
                f'Expected {HERON_BENCH_SAMPLES} Heron-Bench questions and answers, '
                f'got {len(questions)} and {len(answers)}'
            )

        answer_map = {}
        for answer in answers:
            question_id = answer.get('question_id')
            if question_id in answer_map:
                raise ValueError(f'Duplicate Heron-Bench answer question_id: {question_id}')
            answer_map[question_id] = answer

        rows = []
        seen_question_ids = set()
        for question in questions:
            question_id = question.get('question_id')
            if question_id in seen_question_ids:
                raise ValueError(f'Duplicate Heron-Bench question_id: {question_id}')
            seen_question_ids.add(question_id)

            if question_id not in answer_map:
                raise ValueError(f'Missing Heron-Bench reference answer for question_id {question_id}')
            if question.get('category') not in HERON_CATEGORIES:
                raise ValueError(
                    f'Unknown Heron-Bench category for question_id {question_id}: '
                    f'{question.get("category")}'
                )

            image_path = osp.join(dataset_path, 'images', str(question.get('image', '')))
            if not osp.isfile(image_path):
                raise FileNotFoundError(
                    f'Missing Heron-Bench image for question_id {question_id}: {image_path}'
                )

            answer = answer_map[question_id]
            rows.append({
                'index': question_id,
                'image_path': image_path,
                'question': str(question.get('text', '')),
                'caption': str(question.get('context', '')),
                'category': question['category'],
                'image_category': str(question.get('image_category', '')),
                'answer': str(answer.get('text', '')),
                'gpt4_ans': str(answer.get('text', '')),
            })

        extra_answers = set(answer_map).difference(seen_question_ids)
        if extra_answers:
            raise ValueError(
                f'Heron-Bench has reference answers without questions: {sorted(extra_answers)}'
            )
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_paths = self.dump_image(line)
        messages = [dict(type='image', value=path) for path in image_paths]
        messages.append(dict(type='text', value=str(line['question'])))
        return messages

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        judge_model_name = judge_kwargs.pop('model', cls.DEFAULT_JUDGE)
        nproc = int(judge_kwargs.pop('nproc', 4))
        judge_kwargs.setdefault('temperature', 0)

        safe_model_name = _safe_model_name(judge_model_name)
        record_file = get_intermediate_file_path(
            eval_file, f'_{safe_model_name}_heron_judge'
        )
        checkpoint_file = get_intermediate_file_path(
            eval_file, f'_{safe_model_name}_heron_judge_tmp', 'pkl'
        )
        score_file = get_intermediate_file_path(
            eval_file, f'_{safe_model_name}_heron_score', 'csv'
        )

        if not osp.exists(record_file):
            data = load(eval_file)
            required = {
                'index', 'caption', 'question', 'gpt4_ans', 'prediction', 'category',
            }
            missing = required.difference(data.columns)
            if missing:
                raise ValueError(f'Heron-Bench eval file is missing columns: {sorted(missing)}')

            scores = load(checkpoint_file) if osp.exists(checkpoint_file) else {}
            pending = [data.iloc[i] for i in range(len(data)) if data.iloc[i]['index'] not in scores]
            if pending:
                judge = build_judge(
                    model=judge_model_name,
                    system_prompt=HERON_SYSTEM_PROMPT,
                    **judge_kwargs,
                )
                assert judge.working(), (
                    'Heron-Bench evaluation requires a working judge API\n' + DEBUG_MESSAGE
                )
                keys = [line['index'] for line in pending]
                prompts = [build_heron_judge_prompt(line) for line in pending]
                new_scores = track_progress_rich(
                    heron_bench_atomeval,
                    [(judge, prompt) for prompt in prompts],
                    nproc=nproc,
                    chunksize=nproc,
                    keys=keys,
                    save=checkpoint_file,
                )
                scores.update(zip(keys, new_scores))

            data['gpt4_score'] = [scores[index][0] for index in data['index']]
            data['score'] = [scores[index][1] for index in data['index']]
            dump(data, record_file)

        result = heron_bench_score(load(record_file)).round(2)
        dump(result, score_file)
        return result
