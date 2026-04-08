import re

import pandas as pd

from ..smp.file import dump, get_intermediate_file_path, load
from ..smp.misc import d2df, toliststr
from .image_base import ImageBaseDataset


def _extract_option(pred):
    s = str(pred or "")
    # Align with SafeWork-R1 extraction: boxed answer first.
    boxed = re.findall(r"\\boxed\{\s*([A-Z])\s*\}", s)
    if boxed:
        return boxed[-1]
    # Fallback: use the last parenthesized option.
    m = re.findall(r"\(([A-Z])\)", s)
    if m:
        return m[-1]
    return ""


class M3oralBenchDataset(ImageBaseDataset):
    TYPE = 'MCQ'
    MODALITY = 'IMAGE'
    DATASET_URL = {'M3oralBench': 'https://opencompass.openxlab.space/utils/VLMEval/M3oralBench.tsv'}
    DATASET_MD5 = {'M3oralBench': '0b8eacfdef15e1c1a510059910f3b2dc'}

    @classmethod
    def supported_datasets(cls):
        return ['M3oralBench']

    def __init__(self, dataset='M3oralBench', **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self._instruction_map = {}
        if self.QUERY_JSON.exists():
            try:
                items = load(str(self.QUERY_JSON))
                self._instruction_map = {
                    int(item['id']): str(item.get('instruction', '')).strip()
                    for item in items
                    if 'id' in item and str(item.get('instruction', '')).strip()
                }
            except Exception:
                self._instruction_map = {}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line) if not self.meta_only else toliststr(line['image_path'])
        prompt = None
        line_id = line.get('sample_id', None)
        if line_id is None:
            line_id = line.get('id', None)
        if line_id is None and line.get('index', None) is not None:
            try:
                line_id = int(line['index']) + 1
            except Exception:
                line_id = None
        if line_id is not None:
            try:
                prompt = self._instruction_map.get(int(line_id), None)
            except Exception:
                prompt = None
        if (
            not prompt
            and 'instruction' in line
            and not pd.isna(line['instruction'])
            and str(line['instruction']).strip()
        ):
            prompt = str(line['instruction'])
        if prompt is None:
            options = {k: line[k] for k in 'ABCDEFG' if k in line and not pd.isna(line[k])}
            prompt = f"Question: {line['question']}\n"
            if len(options):
                prompt += 'Options:\n'
                for k, v in options.items():
                    prompt += f'{k}. {v}\n'
                letters = "/".join([f"({k})" for k in options.keys()])
                prompt += f'Please answer with only one option in this format: {letters}.'
        # Align with SafeWork-R1 M3oralBench input construction: user content is image first, then text.
        msgs = [dict(type='image', value=p) for p in tgt_path]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'prediction' in data and 'answer' in data

        data['pred_option'] = [_extract_option(x) for x in data['prediction']]
        data['answer'] = [str(x).strip().upper() for x in data['answer']]
        data['correct'] = [int(p == a) for p, a in zip(data['pred_option'], data['answer'])]

        ret = {'Overall': round(data['correct'].mean() * 100 if len(data) else 0, 2)}

        if 'task_type' in data:
            for t in sorted(set(data['task_type'])):
                sub = data[data['task_type'] == t]
                ret[f'task_{t}'] = round(sub['correct'].mean() * 100 if len(sub) else 0, 2)

        if 'category' in data:
            for c in sorted(set(data['category'])):
                sub = data[data['category'] == c]
                ret[f'foundation_{c}'] = round(sub['correct'].mean() * 100 if len(sub) else 0, 2)

        detailed_file = get_intermediate_file_path(eval_file, '_detailed', 'xlsx')
        dump(data, detailed_file)
        score = d2df(ret)
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        return score
