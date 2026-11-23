import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from vlmeval.smp import d2df, dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset

ANSWER_FORMATS = {
    'count': 'Answer with a single integer.',
    'yesno': 'Answer with yes or no.',
    'choice': 'Answer with the letter of the correct option.',
    'numeric': 'Answer with a single number.',
}

NUMBER_WORDS = {
    w: i for i, w in enumerate(
        'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
        'fifteen sixteen seventeen eighteen nineteen twenty'.split()
    )
}


def _final_segment(text):
    tagged = re.findall(r'<answer>(.*?)</answer>', text, flags=re.S | re.I)
    if tagged:
        return tagged[-1].strip()
    m = re.search(r'(?:final answer|answer)\s*(?:is|:)\s*(.+)', text, flags=re.I)
    return m.group(1).strip() if m else text.strip()


def parse_answer(text, answer_type, options=None):
    """Extract a normalized answer string, or None if the response contains no valid answer."""
    seg = _final_segment(text or '')
    low = seg.lower()
    if answer_type in ('count', 'numeric'):
        m = re.search(r'-?\d+(?:\.\d+)?', seg)
        if m:
            value = float(m.group())
            return str(int(round(value))) if answer_type == 'count' else f'{value:g}'
        for word, value in NUMBER_WORDS.items():
            if re.search(rf'\b{word}\b', low):
                return str(value)
        return None
    if answer_type == 'yesno':
        m = re.search(r'\b(yes|no)\b', low)
        return m.group(1) if m else None
    if answer_type == 'choice':
        m = re.search(r'^\(?([A-D])\)?(?:[\s.:)]|$)', seg.strip())
        if m:
            return m.group(1)
        m = re.search(r'\(([A-D])\)', seg)
        if m:
            return m.group(1)
        if options:
            hits = [i for i, option in enumerate(options) if option.lower() in low]
            if len(hits) == 1:
                return 'ABCD'[hits[0]]
        m = re.search(r'\b([A-D])\b', seg)
        return m.group(1) if m else None
    raise ValueError(f'Unknown answer type: {answer_type}')


def mean_relative_accuracy(pred, target):
    """VSI-Bench MRA: mean over theta in {0.50, ..., 0.95} of 1[|pred - gt| / gt < 1 - theta]."""
    if target == 0:
        return float(pred == 0)
    rel = abs(pred - target) / abs(target)
    return float(np.mean([rel < 1 - t for t in np.arange(0.50, 0.951, 0.05)]))


def score(answer, answer_type, prediction):
    """Exact match for counts, yes/no and choices; MRA for metric distances."""
    if prediction is None:
        return 0.0
    if answer_type == 'numeric':
        return mean_relative_accuracy(float(prediction), float(answer))
    if answer_type == 'count':
        return float(int(prediction) == int(float(answer)))
    return float(prediction == str(answer))


def _options(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == '':
        return []
    return json.loads(value)


class SpatialSceneBench(ImageBaseDataset):
    """
    Spatial Scene Bench.

    Counting and relational reasoning over procedurally rendered 3D scenes, from single images
    to 16-frame orbit videos, with answers computed from the 3D scene itself.

    Reference:
      https://github.com/selfishout/spatial-scene-bench
    """

    TYPE = 'VQA'
    force_use_dataset_prompt = True
    DATASET_URL = {
        'SpatialSceneBench': 'https://huggingface.co/datasets/alithema/spatial-scene-bench/resolve/main/vlmevalkit/SpatialSceneBench.tsv',  # noqa: E501
        'SpatialSceneBench_mini': 'https://huggingface.co/datasets/alithema/spatial-scene-bench/resolve/main/vlmevalkit/SpatialSceneBench_mini.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'SpatialSceneBench': '96c2683ccaf92cd140c7d8daa8ef2e72',
        'SpatialSceneBench_mini': 'b7b1feafa0f96728797918a11ae11f66',
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        prompt = f"{line['question']}\n{ANSWER_FORMATS[line['answer_type']]}"
        msgs = [dict(type='image', value=p) for p in tgt_path]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    @staticmethod
    def compute_metrics(data):
        """Per-type accuracy plus macro averages over all, image-only and video-only types."""
        by_type = defaultdict(list)
        count_mra = []
        for _, row in data.iterrows():
            options = _options(row.get('options'))
            pred = parse_answer(str(row['prediction']), row['answer_type'], options)
            by_type[row['question_type']].append(score(row['answer'], row['answer_type'], pred))
            if row['answer_type'] == 'count':
                gt = float(row['answer'])
                count_mra.append(0.0 if pred is None else mean_relative_accuracy(float(pred), gt))

        per_type = {t: 100 * float(np.mean(v)) for t, v in sorted(by_type.items())}
        image = [v for t, v in per_type.items() if not t.startswith('video_')]
        video = [v for t, v in per_type.items() if t.startswith('video_')]
        metrics = {'Overall': float(np.mean(list(per_type.values())))}
        if image:
            metrics['Image'] = float(np.mean(image))
        if video:
            metrics['Video'] = float(np.mean(video))
        if count_mra:
            metrics['Count_MRA'] = 100 * float(np.mean(count_mra))
        metrics.update(per_type)
        return {k: round(v, 2) for k, v in metrics.items()}

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'prediction' in data and 'answer' in data
        data['prediction'] = [str(x) if not pd.isna(x) else '' for x in data['prediction']]
        meta = self.data.set_index('index')
        for col in ('question_type', 'answer_type', 'options'):
            if col not in data:
                data[col] = [meta.loc[i, col] for i in data['index']]
        metrics = self.compute_metrics(data)
        dump(d2df(metrics), get_intermediate_file_path(eval_file, '_acc', 'csv'))
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'Overall' in metrics:
            return {'Overall': metrics['Overall']}
        return super().report_primary_metric(metrics)
