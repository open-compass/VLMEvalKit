"""PointerBench adapter for https://github.com/warmwindOS/pointerbench.

Scoring follows the MIT-licensed official evaluators at commit
35e34f57462262ffc321b4ea0a6eb3bae2995fb4; see pointerbench.LICENSE.
Images and annotations are CC BY 4.0. No LLM judge is required.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load, md5
from vlmeval.smp.status_report import is_number, to_number
from ..image_base import ImageBaseDataset

SYSTEM_PROMPT = (
    'You are evaluating Pointerbench, a GUI grounding benchmark. '
    'You will receive one 1024x768 screenshot and one task instruction. '
    'Use absolute pixel coordinates with origin at the top-left of the image. '
    'Do not return normalized coordinates. Do not crop or resize the coordinate frame. '
    'For point tasks, return JSON like {"point": [x, y]}. '
    'For bounding-box tasks, return JSON like {"bbox": [x0, y0, x1, y1]}.'
)


def parse_prediction(value, answer_type):
    """Unwrap model JSON, retaining raw text for the official bbox fallback."""
    if isinstance(value, dict):
        return value

    if isinstance(value, (list, tuple)):
        return {answer_type: value}

    if not isinstance(value, str):
        return {}

    try:
        parsed = json.loads(value)
    except ValueError:
        # Accept a JSON object inside a code fence or an explanatory response.
        parsed = None
        decoder = json.JSONDecoder()
        for match in re.finditer(r'\{', value):
            try:
                parsed, _ = decoder.raw_decode(value[match.start():])
                break
            except ValueError:
                continue

    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, list):
        return {answer_type: parsed}

    return {'raw': value}


def _point(record):
    for key in ('point', 'pred', 'coordinate', 'prediction'):
        value = record.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]

    if 'x' in record and 'y' in record:
        return [float(record['x']), float(record['y'])]

    return None


def _numbers(value):
    if isinstance(value, str):
        return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', value)]

    if isinstance(value, (list, tuple)):
        return [float(x) for x in value if isinstance(x, (int, float))]

    return []


def _bbox(record):
    for key in ('bbox', 'box', 'pred', 'prediction'):
        numbers = _numbers(record.get(key))
        if len(numbers) >= 4:
            return numbers[:4]

    if all(key in record for key in ('x0', 'y0', 'x1', 'y1')):
        return [float(record[key]) for key in ('x0', 'y0', 'x1', 'y1')]

    numbers = _numbers(record.get('text') or record.get('raw') or record.get('answer'))
    return numbers[:4] if len(numbers) >= 4 else None


def _ordered_bbox(box):
    x0, y0, x1, y1 = box
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def score_prediction(prediction, example):
    """Return (hit, missing) using absolute pixels and per-example thresholds."""
    answer_type = example['answer_type']
    record = parse_prediction(prediction, answer_type)

    try:
        coordinates = _bbox(record) if answer_type == 'bbox' else _point(record)
        if coordinates is None or not all(math.isfinite(x) for x in coordinates):
            return False, True
    except (ValueError, TypeError, OverflowError):
        return False, True

    x0, y0, x1, y1 = _ordered_bbox(example['bbox'])
    if answer_type == 'point':
        return x0 <= coordinates[0] <= x1 and y0 <= coordinates[1] <= y1, False

    px0, py0, px1, py1 = _ordered_bbox(coordinates)
    intersection = (
        max(0, min(x1, px1) - max(x0, px0))
        * max(0, min(y1, py1) - max(y0, py0))
    )
    gt_area = (x1 - x0) * (y1 - y0)
    pred_area = (px1 - px0) * (py1 - py0)
    coverage = intersection / gt_area if gt_area else 0.0
    precision = intersection / pred_area if pred_area else 0.0

    rule = example.get('eval') or {}
    hit = (
        coverage >= float(rule.get('min_coverage', 0.90))
        and precision >= float(rule.get('min_precision', 0.70))
    )
    return hit, False


class PointerBench(ImageBaseDataset):
    TYPE = 'GUI'
    force_use_dataset_prompt = True
    HF_REPO = 'WarmwindOS/pointerbench'
    HF_REVISION = '5a421276ad8bfe813116df04482f81a6201fef13'
    METADATA_MD5 = {
        'sheets': '0312a15bf3c68cebc0c1fba398d6b3cd',
        'text': '70105469d41067d70f4ddea8dddaa62e',
        'pro': '2e7c99a737218710f56334a0e851b3cc',
    }

    SUBSETS = {
        'PointerBench_Sheets': 'sheets',
        'PointerBench_Text': 'text',
        'PointerBench_Pro': 'pro',
    }
    AXES = {
        'sheets': (
            'category',
            'ui_style',
            'data_type',
        ),
        'text': (
            'answer_type',
            'data_type',
            'category',
            'surface',
            'language',
            'difficulty',
        ),
        'pro': (
            'element_type',
            'app_category',
            'platform',
            'app',
        ),
    }

    def __init__(self, dataset='PointerBench', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    @classmethod
    def supported_datasets(cls):
        return ['PointerBench', *cls.SUBSETS]

    def load_data(self, dataset):
        if dataset not in self.supported_datasets():
            raise ValueError(f'Unsupported PointerBench dataset: {dataset}')

        subsets = (
            list(self.SUBSETS.values())
            if dataset == 'PointerBench'
            else [self.SUBSETS[dataset]]
        )

        root = Path(LMUDataRoot()) / 'PointerBench'
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(
                repo_id=self.HF_REPO,
                repo_type='dataset',
                revision=self.HF_REVISION,
                allow_patterns=[
                    f'pointerbench-{subset}/data/test/*'
                    for subset in subsets
                ],
            ))

        records = []
        for subset in subsets:
            subset_root = root / f'pointerbench-{subset}' / 'data' / 'test'
            metadata = subset_root / 'metadata.jsonl'
            if not metadata.is_file():
                raise FileNotFoundError(f'PointerBench metadata is missing: {metadata}')

            expected_md5 = self.METADATA_MD5[subset]
            actual_md5 = md5(str(metadata))
            if actual_md5 != expected_md5:
                raise ValueError(
                    f'PointerBench {subset} metadata MD5 mismatch: '
                    f'expected {expected_md5}, got {actual_md5} ({metadata})')

            with metadata.open(encoding='utf-8') as stream:
                examples = [json.loads(line) for line in stream if line.strip()]

            for example in examples:
                image_path = subset_root / example['file_name']
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f'PointerBench screenshot is missing: {image_path}')

                rule = example.get('eval') or {}
                example['answer_type'] = example.get('answer_type') or (
                    'bbox' if rule.get('type') == 'iou' else 'point')
                records.append(dict(
                    example,
                    index=example['id'],
                    question=example['instruction'],
                    subset=subset,
                    image_path=str(image_path.resolve()),
                ))

        data = pd.DataFrame(records)
        if data.empty or data['index'].duplicated().any():
            raise ValueError(
                'PointerBench metadata must contain nonempty, unique example IDs.')

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        return [
            dict(role='system', type='text', value=SYSTEM_PROMPT),
            dict(role='user', type='image', value=line['image_path']),
            dict(role='user', type='text', value=line['question']),
        ]

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and is_number(metrics.get('accuracy')):
            return {'accuracy': to_number(metrics['accuracy'])}

        return super().report_primary_metric(metrics)

    def evaluate(self, eval_file, **judge_kwargs):
        predictions = load(eval_file)
        if predictions['index'].duplicated().any():
            raise ValueError('PointerBench predictions contain duplicate example IDs.')

        prediction_map = dict(zip(
            predictions['index'].astype(str),
            predictions['prediction'],
        ))
        unknown = set(prediction_map) - set(self.data['index'])
        if unknown:
            raise ValueError(
                f'Unknown PointerBench prediction IDs: {sorted(unknown)}')

        details = []
        reports = {}
        for subset, examples in self.data.groupby('subset', sort=False):
            by = {
                axis: defaultdict(lambda: [0, 0])
                for axis in self.AXES[subset]
            }
            hits = missing = 0
            for example in examples.to_dict('records'):
                prediction = prediction_map.get(example['index'])
                hit, absent = score_prediction(prediction, example)
                hits += int(hit)
                missing += int(absent)
                details.append(dict(
                    index=example['index'],
                    subset=subset,
                    answer_type=example['answer_type'],
                    prediction=prediction,
                    hit=bool(hit),
                    missing=bool(absent),
                ))
                for axis in by:
                    value = example.get(axis, '?')
                    if pd.isna(value):
                        value = '?'
                    by[axis][value][0] += int(hit)
                    by[axis][value][1] += 1

            reports[subset] = dict(
                n=len(examples),
                accuracy=round(hits / len(examples), 4),
                hits=hits,
                missing_predictions=missing,
            )
            for axis, groups in by.items():
                reports[subset][f'by_{axis}'] = {
                    key: dict(acc=round(count[0] / count[1], 4), n=count[1])
                    for key, count in sorted(groups.items())
                }

        if self.dataset_name == 'PointerBench':
            n = sum(report['n'] for report in reports.values())
            hits = sum(report['hits'] for report in reports.values())
            result = dict(
                n=n,
                accuracy=round(hits / n, 4),
                hits=hits,
                missing_predictions=sum(
                    report['missing_predictions']
                    for report in reports.values()
                ),
                by_subset=reports,
            )
        else:
            result = next(iter(reports.values()))

        dump(
            pd.DataFrame(details),
            get_intermediate_file_path(eval_file, '_scored', 'tsv'),
        )
        dump(
            result,
            get_intermediate_file_path(eval_file, '_score', 'json'),
        )
        return result
