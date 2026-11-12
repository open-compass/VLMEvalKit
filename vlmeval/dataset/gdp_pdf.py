"""GDP.pdf page-image adapter and official per-criterion grading protocol.

Reference: https://github.com/surge-ai/gdp-pdf (7a72a514).
The official harness supplies native PDFs; this adapter supplies every page as
an image, without OCR, page selection, or additional question instructions.
"""

import hashlib
import json
import os
import re
import tempfile
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from vlmeval.smp import LMUDataRoot, dump, load, md5, read_ok
from .image_base import ImageBaseDataset
from .utils.judge_cache import (get_judge_cache_file, get_judge_detail_file, get_judge_score_file,
                                load_judge_cache, run_cached_tasks)
from .utils.judge_util import build_judge

JUDGE_PROMPT_TEMPLATE = """Read this json object:
{
  response: {{ rubric.response_to_grade }},
  criteria: {{ rubric.criterion }},
}

Score the response based on the criteria.
If the response perfectly satisfies the criteria, give it a score of "1".
If it does not satisfy the criteria at all, give it a score of "0".
Return the score and rationale in a JSON object with two keys, 'score' and 'rationale'.
'score' should be a string that's either "0" or "1".
'rationale' should be a description of why you gave the response that score.

Here are some examples of how to score a response:
{ score: "1", rationale: "The response is a perfect match for the criteria." }
{ score: "0", rationale: "The response does not satisfy the criteria at all." }
Unlike these examples, when you create a rationale, be specific and detailed. Give reasons specific to this response and criteria. Don't be generic."""  # noqa: E501

RESPONSE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'individual_criteria_score',
        'schema': {
            'type': 'object',
            'required': ['rationale', 'score'],
            'properties': {
                'rationale': {
                    'type': 'string',
                    'description': 'Text describing the evaluation criterion',
                },
                'score': {
                    'type': 'string',
                    'enum': ['0', '1'],
                    'description': (
                        "Score value: '0' (does not satisfy) or '1' (perfectly satisfies)"
                    ),
                },
            },
            'additionalProperties': False,
        },
    },
}

_RUBRIC_RE = re.compile(r'rubric\s*-\s*(\d+)\.\s*criterion', re.IGNORECASE)
UNSCORED = 'GDP.pdf judge output unscored'
FAIL_MSG = 'Failed to obtain answer via API.'


def _rubrics_from_row(row):
    found = []
    for column, value in row.items():
        match = _RUBRIC_RE.fullmatch(str(column).strip())
        if match and pd.notna(value) and value != '':
            found.append((int(match.group(1)), str(value)))
    return [
        {'id': f'r{number}', 'criterion': text}
        for number, text in sorted(found)
    ]


def _judge_prompt(response, criterion):
    return JUDGE_PROMPT_TEMPLATE.replace(
        '{{ rubric.response_to_grade }}', response
    ).replace(
        '{{ rubric.criterion }}', criterion
    )


def _parse_criterion_result(raw):
    if not isinstance(raw, str):
        raise ValueError('Judge output must be text')
    cleaned = re.sub(r'\A```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    payload = json.loads(re.sub(r'\s*```\Z', '', cleaned))
    if not isinstance(payload, dict) or not {'score', 'rationale'} <= payload.keys():
        raise ValueError('Judge output must contain score and rationale')
    score = payload['score']
    if isinstance(score, bool) or not isinstance(score, (int, float, str)):
        raise ValueError('Score must be a number or string')
    if not isinstance(payload['rationale'], str):
        raise ValueError('Rationale must be text')
    score = Decimal(str(score).strip())
    if not score.is_finite():
        raise ValueError('Score must be finite')
    return score, payload['rationale']


def _criterion_passes(score):
    return score.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP) >= 1


def _score_criterion(model, response, criterion):
    prompt = _judge_prompt(response, criterion)
    attempts = []
    # The official scorer retries parsing four times after the first attempt.
    for _ in range(5):
        raw = model.generate(prompt, response_format=RESPONSE_FORMAT)
        attempts.append(raw)
        try:
            score, rationale = _parse_criterion_result(raw)
            passed = _criterion_passes(score)
        except (ValueError, InvalidOperation):
            continue
        return dict(
            score=str(score),
            passed=passed,
            rationale=rationale,
            judge_response=attempts,
        )
    return dict(
        score=None,
        passed=None,
        rationale=UNSCORED,
        judge_response=attempts,
    )


def _sample_scores(grades):
    scored = [grade for grade in grades if grade['passed'] is not None]
    passed = sum(grade['passed'] is True for grade in scored)
    fully_scored = bool(grades) and len(scored) == len(grades)
    return {
        'all_pass': float(passed == len(grades)) if fully_scored else float('nan'),
        'mean_criteria': passed / len(scored) if scored else float('nan'),
        'num_scored_criteria': len(scored),
        'num_unscored_criteria': len(grades) - len(scored),
    }


class GDPPDF(ImageBaseDataset):
    """GDP.pdf (100 held-out tasks), rendered as complete PDF page images."""

    TYPE = 'VQA'
    DEFAULT_JUDGE_MODEL = 'gemini-3.5-flash'
    HF_REPO_ID = 'cloudybean/GDP.pdf'
    HF_REVISION = '1b73b865da6be76dca2e4f3ad66a53d6709ac9f5'
    HF_PARQUET_MD5 = 'aa6d3ce9b6285b767c888da35c2cb7c8'
    force_use_dataset_prompt = True
    JUDGE_FAIL_MARKERS = (FAIL_MSG, UNSCORED)
    # ``all_pass`` is 1.0/0.0 once every criterion was scored and NaN otherwise; NaN cannot pass.
    PASS_K_HIT_COL = 'all_pass'
    DEFAULT_N_RUNS = 5

    def __init__(self, dataset='GDP_pdf', dpi=144):
        if dataset not in self.supported_datasets():
            raise ValueError(f'Unsupported GDP.pdf dataset: {dataset}')
        if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
            raise ValueError('dpi must be a positive integer')
        self.dpi = dpi
        super().__init__(dataset=dataset, skip_noimg=False)

    @classmethod
    def supported_datasets(cls):
        return ['GDP_pdf']

    def _local_file(self, filename):
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Expected a repository-relative attachment path: {filename!r}')
        path = self.data_root / relative
        if not path.is_file():
            hf_hub_download(
                repo_id=self.HF_REPO_ID,
                repo_type='dataset',
                revision=self.HF_REVISION,
                filename=relative.as_posix(),
                local_dir=str(self.data_root),
            )
        return path

    def load_data(self, dataset):
        self.data_root = Path(LMUDataRoot()) / dataset
        self.data_root.mkdir(parents=True, exist_ok=True)
        data_file = self._local_file('data.parquet')
        data_md5 = md5(str(data_file))
        if data_md5 != self.HF_PARQUET_MD5:
            raise ValueError(
                f'GDP.pdf data.parquet MD5 mismatch: expected {self.HF_PARQUET_MD5}, '
                f'got {data_md5} ({data_file})'
            )
        source = pd.read_parquet(data_file)
        rows = []
        for index, row in enumerate(source.to_dict('records')):
            rubrics = _rubrics_from_row(row)
            if not rubrics:
                raise ValueError(f'GDP.pdf row {index} has no rubric criteria')
            if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError(f'GDP.pdf row {index} has no prompt')
            if not isinstance(row.get('pdf_path'), str) or not row['pdf_path']:
                raise ValueError(f'GDP.pdf row {index} has no PDF path')
            task_id = row.get('task_id')
            if pd.isna(task_id) or task_id == '':
                task_id = row.get('id')
            if pd.isna(task_id) or task_id == '':
                task_id = index
            rows.append(dict(
                index=str(task_id),
                question=row['prompt'],
                pdf_path=row['pdf_path'],
                rubrics=json.dumps(rubrics, ensure_ascii=False),
                category=row.get('domain', ''),
            ))
        data = pd.DataFrame(rows)
        if data.empty or data['index'].duplicated().any():
            raise ValueError('GDP.pdf must have nonempty data and unique task identifiers')
        for filename in data['pdf_path'].unique():
            self._local_file(filename)
        return data

    def dump_image(self, line):
        try:
            import fitz
        except ImportError as err:
            raise ImportError(
                'GDP.pdf page rendering requires PyMuPDF: pip install pymupdf'
            ) from err
        pdf = self.data_root / line['pdf_path']
        directory = Path(self.img_root) / f'{md5(str(pdf))}_{self.dpi}dpi'
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        with fitz.open(str(pdf)) as doc:
            if not len(doc):
                raise ValueError(f'PDF has no pages: {pdf}')
            for number in range(len(doc)):
                path = directory / f'page_{number + 1:05d}.png'
                if not read_ok(str(path)):
                    pixmap = doc[number].get_pixmap(dpi=self.dpi, alpha=False)
                    with tempfile.NamedTemporaryFile(
                        dir=directory, suffix='.png', delete=False
                    ) as stream:
                        stream.write(pixmap.tobytes('png'))
                        temporary = stream.name
                    os.replace(temporary, path)
                paths.append(str(path))
        return paths

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [dict(type='text', value=line['question'])] + [
            dict(type='image', value=path) for path in self.dump_image(line)
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        predictions = load(eval_file)
        predictions['index'] = predictions['index'].astype(str)
        metadata = self.data.copy()
        metadata['index'] = metadata['index'].astype(str)
        if (
            predictions['index'].duplicated().any()
            or set(predictions['index']) != set(metadata['index'])
        ):
            raise ValueError('GDP.pdf predictions must contain each dataset index exactly once')
        # Use canonical metadata, not possibly edited rubric columns in predictions.
        data = metadata.merge(
            predictions[['index', 'prediction']], on='index', validate='one_to_one'
        )
        judge_name = judge_kwargs.pop('model', self.DEFAULT_JUDGE_MODEL)
        nproc = judge_kwargs.pop('nproc', 4)
        cache_file = get_judge_cache_file(eval_file, 'rubric_v1', judge_name)
        cache = load_judge_cache(cache_file)
        tasks, keys, sample_keys = [], [], []
        for _, row in data.iterrows():
            response = '' if pd.isna(row['prediction']) else str(row['prediction'])
            row_keys = []
            for rubric in json.loads(row['rubrics']):
                fingerprint = json.dumps([row['index'], response, rubric], ensure_ascii=False)
                key = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()
                row_keys.append((key, rubric['id']))
                if FAIL_MSG in response:
                    cache[key] = dict(
                        score='0',
                        passed=False,
                        rationale='Inference failed.',
                        judge_response=[],
                    )
                elif key not in cache or cache[key]['passed'] is None:
                    tasks.append((response, rubric['criterion']))
                    keys.append(key)
            sample_keys.append(row_keys)
        if tasks:
            judge_kwargs.setdefault('max_tokens', 4096)
            model = build_judge(model=judge_name, **judge_kwargs)
            # Criterion-level tasks allow nproc concurrent calls even for long rubrics.
            judged = run_cached_tasks(
                _score_criterion,
                [(model, *task) for task in tasks],
                keys,
                cache_file,
                nproc=nproc,
            )
            cache.update(judged)
        grades = [
            [
                dict(rubric_id=rubric_id, **cache[key])
                for key, rubric_id in row_keys
            ]
            for row_keys in sample_keys
        ]
        scores = pd.DataFrame([_sample_scores(row) for row in grades])
        for column in scores:
            data[column] = scores[column].values
        data['judge_log'] = [json.dumps(row, ensure_ascii=False) for row in grades]
        self._judge_name = judge_name
        dump(data, get_judge_detail_file(eval_file, 'rubric_v1', judge_name, 'tsv'))

        def summarize(frame):
            return {
                'all_pass': float(frame['all_pass'].mean()),
                'mean_criteria': float(frame['mean_criteria'].mean()),
                'num_samples': len(frame),
                'num_fully_scored': int(frame['all_pass'].notna().sum()),
                'num_scored_criteria': int(frame['num_scored_criteria'].sum()),
                'num_unscored_criteria': int(frame['num_unscored_criteria'].sum()),
            }

        result = summarize(data)
        for category, frame in data.groupby('category', dropna=False):
            result.update({
                f'domain/{category}/{key}': value
                for key, value in summarize(frame).items()
            })
        dump(result, get_judge_score_file(eval_file, judge_name, 'json'))
        return result

    def get_pass_k_detail_paths(self, sub_file):
        # The detail file is named after the judge that produced it, so it cannot be derived from
        # a fixed suffix.
        return [get_judge_detail_file(sub_file, 'rubric_v1', self._judge_name, 'tsv')]

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and pd.notna(metrics.get('all_pass')):
            return {'all_pass': metrics['all_pass']}
        return {}
