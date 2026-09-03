"""C4 Bench: cross-concept creativity evaluation with Chinese chengyu.

Paper: https://arxiv.org/abs/2608.06501
Dataset: https://huggingface.co/datasets/sci-m-wang/C4-Eval

Citation::

    @misc{wang2026mllmsdecodecreativeleap,
          title={Can MLLMs Decode the Creative Leap? Introducing C4 for
                 Cross-Concept Understanding},
          author={Ming Wang and Yuqing Zhang and Tingna Xie and Xiangju Li and
                  Xiaocui Yang and Daling Wang and Shi Feng and Yifei Zhang},
          year={2026},
          eprint={2608.06501},
          archivePrefix={arXiv},
          primaryClass={cs.AI},
          url={https://arxiv.org/abs/2608.06501},
    }
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset

HF_REPO_ID = 'sci-m-wang/C4-Eval'
EVAL_FILENAME = 'data/eval.jsonl'
PRIMARY_TASKS = ('H0', 'H1', 'H4', 'E0')
EXPLANATION_TASKS = ('E0', 'E1')
DATASET_TASKS = {
    'C4Bench': PRIMARY_TASKS,
    'C4Bench_H0': ('H0', ),
    'C4Bench_H1': ('H1', ),
    'C4Bench_H4': ('H4', ),
    'C4Bench_E0': ('E0', ),
    'C4Bench_E1': ('E1', ),
}

PUNCT_RE = re.compile(r'[\s\n\r\t，,。.!！?？:：;；、\'"“”‘’`·]+')
ANSWER_PATTERNS = (
    re.compile(
        r'["\']?answer["\']?\s*[:：]\s*["“”\']?([\u4e00-\u9fff]{4})',
        re.IGNORECASE,
    ),
    re.compile(r'(?:答案|成语)\s*(?:是|为|[:：])\s*["“”\']?([\u4e00-\u9fff]{4})'),
)


def normalize_answer(text):
    return PUNCT_RE.sub('', str(text or '')).strip()


def strip_code_fence(text):
    cleaned = str(text or '').strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


def recover_explicit_answer(text):
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(str(text or ''))
        if matches:
            return matches[-1]
    lines = [line.strip().strip('"“”') for line in str(text or '').splitlines() if line.strip()]
    if lines and re.fullmatch(r'[\u4e00-\u9fff]{4}', lines[-1]):
        return lines[-1]
    return ''


def parse_task_answer(task, output):
    if task in EXPLANATION_TASKS:
        try:
            parsed = json.loads(strip_code_fence(output))
        except (json.JSONDecodeError, TypeError):
            return recover_explicit_answer(output), False
        if not isinstance(parsed, dict):
            return '', False
        return str(parsed.get('answer', '')).strip(), True

    lines = [line.strip() for line in str(output or '').splitlines() if line.strip()]
    answer = lines[0].strip('"“”') if len(lines) == 1 else recover_explicit_answer(output)
    return answer, None


def _answer_aliases(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value]
    if not value or pd.isna(value):
        return []
    if isinstance(value, str) and value.startswith('['):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return [str(value)]


class C4Bench(ImageBaseDataset):
    """C4 Bench primary and task-specific image-text-to-text evaluations."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    @classmethod
    def supported_datasets(cls):
        return list(DATASET_TASKS)

    def load_data(self, dataset):
        if dataset not in DATASET_TASKS:
            raise ValueError(f'Unsupported C4 Bench dataset: {dataset}')
        dataset_root = Path(LMUDataRoot()) / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            local_dir=dataset_root,
            allow_patterns=EVAL_FILENAME,
        )
        data_path = dataset_root / EVAL_FILENAME
        data = pd.read_json(data_path, lines=True)
        data = data[data['task'].isin(DATASET_TASKS[dataset])].copy()
        data['index'] = data['instance_id']
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            local_dir=self.img_root,
            allow_patterns='images/**',
        )
        self._image_root = Path(self.img_root)
        return data.reset_index(drop=True)

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_path = self._image_root / str(line['image_path'])
        if not image_path.is_file():
            raise FileNotFoundError(f'C4 Bench image is missing: {image_path}')
        return [str(image_path)]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [
            {
                'type': 'image',
                'value': self.dump_image(line)[0]
            },
            {
                'type': 'text',
                'value': str(line['question'])
            },
        ]

    @staticmethod
    def _accepted_answers(line):
        aliases = _answer_aliases(line.get('answer_aliases', []))
        answers = [line.get('answer', ''), *aliases]
        return {normalize_answer(answer) for answer in answers if normalize_answer(answer)}

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).copy()
        metadata = {str(row['index']): row for _, row in self.data.iterrows()}
        details = []

        for _, row in data.iterrows():
            meta = metadata[str(row['index'])]
            task = str(meta['task'])
            parsed_answer, valid_json = parse_task_answer(task, row.get('prediction', ''))
            exact = normalize_answer(parsed_answer) in self._accepted_answers(meta)
            details.append({
                'index': str(row['index']),
                'task': task,
                'prediction': str(row.get('prediction', '')),
                'parsed_answer': parsed_answer,
                'answer': str(meta['answer']),
                'exact_match': int(exact),
                'json_valid': valid_json,
            })

        detail_frame = pd.DataFrame(details)
        detail_file = get_intermediate_file_path(eval_file, '_c4_result')
        dump(detail_frame, detail_file)

        metrics = {}
        primary = detail_frame[detail_frame['task'].isin(PRIMARY_TASKS)]
        if len(primary):
            metrics['Overall'] = float(primary['exact_match'].mean() * 100)
        for task in DATASET_TASKS[self.dataset_name]:
            subset = detail_frame[detail_frame['task'] == task]
            if not len(subset):
                continue
            if task != 'E1':
                metrics[f'{task} Exact Match'] = float(subset['exact_match'].mean() * 100)
            if task in EXPLANATION_TASKS:
                json_valid = subset['json_valid'].astype(bool).mean() * 100
                metrics[f'{task} JSON Valid'] = float(json_valid)
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if not isinstance(metrics, dict):
            return {}
        if 'Overall' in metrics:
            return {'Primary Score': metrics['Overall']}
        if 'E1 JSON Valid' in metrics:
            return {'E1 JSON Valid': metrics['E1 JSON Valid']}
        return super().report_primary_metric(metrics)
