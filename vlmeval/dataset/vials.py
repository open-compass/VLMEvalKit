import hashlib
import html
import json
import math
import re
from pathlib import Path

import pandas as pd

from vlmeval.smp import LMUDataRoot, d2df, dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset
from .utils import build_judge, warn_if_judge_model_changed
from .utils.judge_cache import (get_judge_cache_file, get_judge_detail_file, load_judge_cache,
                                run_cached_tasks)

VIALS_DATASET = 'VIALS'
VIALS_HF_REPO = 'handshake-ai-research/VIALS'
VIALS_HF_CONFIG = 'tasks'
VIALS_HF_SPLIT = 'test'
VIALS_EXPECTED_TASKS = 161
VIALS_INFO_FILE = 'dataset_info.json'
# Bump this when the judge prompt, response schema, or grading contract changes.
VIALS_JUDGE_CACHE_STAGE = 'vials_v1'
# Pin the data version used by this integration. The official project keeps
# updating `main`, so recording the commit keeps reported scores reproducible.
VIALS_HF_REVISION = '21b67dcc71b75cef9457ef4b2a76fb661fb7ca7d'
VIALS_HF_URL = f'https://huggingface.co/datasets/{VIALS_HF_REPO}/tree/{VIALS_HF_REVISION}'

VIALS_SYSTEM_PROMPT = (
    'You are an expert scientist. You will be shown a scientific image and a '
    'question. Study the image carefully, then respond with your reasoning '
    'and a single final answer.'
)

VIALS_USER_INSTRUCTION_TAIL = (
    'Reason through the image and task, then commit to a single final answer. '
    'Keep the reasoning concise so your response ends with the ANSWER line. '
    'The answer must appear on its own line prefixed by \'ANSWER:\' and contain '
    'only the final answer: no LaTeX brackets, no extra explanation, no units '
    'unless the question asks for them.'
    '\n\nFormat your response EXACTLY as:\n'
    'REASONING: <your reasoning>\n'
    'ANSWER: <final answer>'
)

VIALS_JUDGE_MAX_TOKENS = 512
VIALS_JUDGE_MAX_RETRIES = 3
VIALS_JUDGE_SYSTEM = (
    'You are grading a scientific answer. Given the question, the '
    'ground-truth final answer (GTFA), and a candidate answer, decide '
    'whether the candidate expresses the same intended answer as the GTFA '
    'in the context of the question. You do NOT see the image.\n\n'
    'IMPORTANT: You are grading, not solving. Do NOT attempt to work out '
    'the correct answer to the question yourself and then compare the '
    'candidate to your own derivation. Treat the GTFA as the sole source '
    'of truth. Your only decision is whether the candidate expresses the '
    'same answer as the GTFA.\n\n'
    'Apply the standard of a competent domain scientist. ACCEPT variations '
    'that an expert would recognize as immaterial to the answer:\n'
    '  - Formatting: punctuation, whitespace, dash / hyphen style, '
    'thousands separators, and equivalent scientific notation for the '
    'same underlying value\n'
    '  - Capitalization: when the identity is otherwise unambiguous\n'
    '  - Units: omitted when unambiguous from the question or GTFA — '
    'but only when the numeric value itself matches\n'
    '  - Lead bounds: when an inclusive LeadBoundLower/LeadBoundUpper range '
    'is provided, a numeric candidate inside that range is equivalent '
    'even if it differs from the exact GTFA value\n'
    '  - Order and duplicates: irrelevant when the answer is a set or '
    'unordered list\n'
    '  - Terminology: synonymous scientific terms of the same specificity\n\n'
    'Numeric answers are STRICT. Do NOT invent tolerances, relative-error '
    'allowances, or "same order of magnitude" acceptance.\n'
    '  - The candidate must express the same numeric value as GTFA '
    '(after the formatting/unit rules above), unless LeadBounds apply\n'
    '  - If the question asks for a stated reporting precision, any '
    'difference at that precision is NOT equivalent\n'
    '  - Exact counts of discrete objects/events must match exactly; '
    'off-by-one is NEVER equivalent\n'
    '  - Different coefficients at the same power of ten are NOT '
    'equivalent\n\n'
    'REJECT when the candidate materially disagrees with GTFA on any of:\n'
    '  - Missing or extra items in a list\n'
    '  - The identity, count, direction, mechanism, or sign of a '
    'biological entity\n'
    '  - Any numeric mismatch under the strict numeric rules above '
    '(unless inside LeadBounds)\n'
    '  - A broader or narrower term when the question required a specific '
    'level of detail\n'
    '  - Extra conflicting numeric values or alternate counts that '
    'disagree with GTFA\n\n'
    'Refusals, meta-answers ("the provided image", "cannot determine"), '
    'and empty responses are ALWAYS not equivalent, regardless of GTFA.\n\n'
    'If evaluating equivalence would require seeing the image — for example, '
    'GTFA uses one labelling system (condition names, colour labels) and the '
    'candidate uses another (lane numbers, positional indices), and the '
    'mapping is not in the question text — return equivalent=false with '
    'your reasoning prefixed "UNVERIFIABLE:". Downstream tooling treats '
    'these as needing a multimodal regrade (or human review), not as '
    'confirmed model errors.\n\n'
    'Reply with the JSON schema. Keep the reasoning to ≤3 sentences and '
    'name the specific principle you applied.'
)

VIALS_MULTIMODAL_JUDGE_SYSTEM = (
    VIALS_JUDGE_SYSTEM.replace(
        'You do NOT see the image.',
        'You are also shown the task image. Use it ONLY to resolve '
        'labelling ambiguities between the candidate and the GTFA. '
        'Do NOT re-solve the question from the image.',
    ).replace(
        'If evaluating equivalence would require seeing the image — for example, '
        'GTFA uses one labelling system (condition names, colour labels) and the '
        'candidate uses another (lane numbers, positional indices), and the '
        'mapping is not in the question text — return equivalent=false with '
        'your reasoning prefixed "UNVERIFIABLE:". Downstream tooling treats '
        'these as needing a multimodal regrade (or human review), not as '
        'confirmed model errors.\n\n',
        'With the image available, decide equivalence directly. Do NOT tag '
        'your reasoning as UNVERIFIABLE.\n\n',
    )
)

VIALS_JUDGE_RESPONSE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'FreeFormJudgeResponse',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'reasoning': {'type': 'string'},
                'equivalent': {'type': 'boolean'},
            },
            'required': ['reasoning', 'equivalent'],
            'additionalProperties': False,
        },
    },
}

_ANCHORED_ANSWER_RE = re.compile(
    r'(?mi)^[ \t]*(?:[-*+\u2022\u00B7]|\d+[.)])?[ \t]*\**[ \t]*ANSWER\**[ \t]*:\**[ \t]*'
)
_LOOSE_ANSWER_RE = re.compile(r'ANSWER[ \t]*:[ \t]*', re.IGNORECASE)
_KNOWN_IMAGE_EXTS = frozenset({'.png', '.jpg', '.jpeg', '.webp', '.gif'})


def extract_answer_block(response):
    """Return the free-form answer after the last official ``ANSWER:`` marker."""
    if not response:
        return None

    matches = list(_ANCHORED_ANSWER_RE.finditer(response))
    if not matches:
        matches = list(_LOOSE_ANSWER_RE.finditer(response))
        if not matches:
            return None

    tail = response[matches[-1].end():].strip()
    tail = re.sub(r'^\\?\[\s*', '', tail)
    tail = re.sub(r'\s*\\?\]\s*$', '', tail)
    tail = re.sub(r'^\$\$?\s*', '', tail)
    tail = re.sub(r'\s*\$\$?$', '', tail)
    tail = re.sub(r'^```(?:\w+)?\s*', '', tail)
    tail = re.sub(r'\s*```$', '', tail)
    tail = re.sub(r'^\*\*(.+)\*\*$', r'\1', tail.strip())
    return tail.strip() or None


def _safe_filename(name, fallback):
    stem = Path((name or '').strip()).name
    if not stem or stem in {'.', '..'}:
        stem = fallback
    if Path(stem).suffix.lower() not in _KNOWN_IMAGE_EXTS:
        stem = f'{stem}.png'
    return stem


def _numeric_bound(value):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _lead_bound_block(line):
    lower = _numeric_bound(line.get('LeadBoundLower'))
    upper = _numeric_bound(line.get('LeadBoundUpper'))
    if lower is None and upper is None:
        return ''
    return (
        'Lead-assigned acceptable numeric range (inclusive): '
        f'lower={"n/a" if lower is None else lower}, '
        f'upper={"n/a" if upper is None else upper}. '
        'If the candidate is a numeric answer inside this range, '
        'mark it equivalent to the GTFA.\n\n'
    )


def build_judge_user_message(line, answer):
    return (
        f"Question:\n{line['Question']}\n\n"
        f"Ground-truth final answer:\n{line['GTFA']}\n\n"
        f"{_lead_bound_block(line)}"
        f"Candidate final answer:\n{answer}\n\n"
        'Are the candidate and ground-truth final answers equivalent?'
    )


def build_multimodal_judge_user_message(line, answer):
    prefix = build_judge_user_message(line, answer).rsplit(
        'Are the candidate and ground-truth final answers equivalent?', 1
    )[0]
    return (
        prefix
        + 'Using the image only to resolve labelling ambiguities, are the '
          'candidate and ground-truth final answers equivalent?'
    )


def _salvage_judge_json(response):
    if not isinstance(response, str):
        return None
    candidates = [response]
    stripped = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', response).strip()
    if stripped and stripped != response:
        candidates.append(stripped)
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match and match.group(0) not in candidates:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict) or not isinstance(parsed.get('reasoning'), str):
            continue
        equivalent = parsed.get('equivalent')
        if isinstance(equivalent, str) and equivalent.lower() in {'true', 'false'}:
            equivalent = equivalent.lower() == 'true'
        if isinstance(equivalent, bool):
            return {'reasoning': parsed['reasoning'], 'equivalent': equivalent}
    return None


def parse_judge_response(response):
    parsed = _salvage_judge_json(response)
    if parsed is None:
        return None
    reasoning = parsed['reasoning']
    return {
        'correct': parsed['equivalent'],
        'judge_reasoning': reasoning,
        'judge_unverifiable': reasoning.lstrip().upper().startswith('UNVERIFIABLE:'),
    }


class VIALS(ImageBaseDataset):
    """Visual Life Sciences benchmark (ATLAS/VIALS)."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE_MODEL = 'gpt-5-mini'
    DEFAULT_N_RUNS = 3
    PASS_K_HIT_COL = 'correct'
    PASS_K_DETAIL_SUFFIX = '_results'
    PASS_K_DETAIL_FORMAT = 'tsv'
    JUDGE_FAIL_MARKERS = ('[JudgeError', '[MultimodalJudgeError')
    force_use_dataset_prompt = True
    DATASET_URL = {VIALS_DATASET: VIALS_HF_URL}

    def __init__(self, dataset=VIALS_DATASET, **kwargs):
        super().__init__(dataset=dataset, skip_noimg=False, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [VIALS_DATASET]

    def _data_dir(self):
        return Path(LMUDataRoot()) / VIALS_DATASET

    def _data_file(self):
        return self._data_dir() / f'{VIALS_DATASET}.tsv'

    @staticmethod
    def _dataset_metadata():
        return {
            'repo_id': VIALS_HF_REPO,
            'config': VIALS_HF_CONFIG,
            'split': VIALS_HF_SPLIT,
            'revision': VIALS_HF_REVISION,
            'num_tasks': VIALS_EXPECTED_TASKS,
        }

    def _image_paths(self, line):
        return [Path(self.img_root) / path for path in toliststr(line['image_path'])]

    def _local_data_is_complete(self):
        data_file = self._data_file()
        metadata_file = self._data_dir() / VIALS_INFO_FILE
        if not data_file.exists() or not metadata_file.exists():
            return False
        try:
            cached_metadata = json.loads(metadata_file.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if cached_metadata != self._dataset_metadata():
            return False
        data = load(str(data_file))
        return all(
            path.is_file()
            for row in data.to_dict('records')
            for path in self._image_paths(row)
        )

    def load_data(self, dataset):
        assert dataset == VIALS_DATASET, dataset
        if self._local_data_is_complete():
            return load(str(self._data_file()))
        return self._download_dataset()

    def _download_dataset(self):
        try:
            from datasets import Image, load_dataset
        except ImportError as exc:
            raise ImportError(
                'VIALS requires the `datasets` package. Install it with `pip install datasets`.'
            ) from exc

        source = load_dataset(
            VIALS_HF_REPO,
            VIALS_HF_CONFIG,
            split=VIALS_HF_SPLIT,
            revision=VIALS_HF_REVISION,
        )
        source = source.cast_column('images', [Image(decode=False)])

        image_root = Path(self.img_root)
        image_root.mkdir(parents=True, exist_ok=True)
        self._data_dir().mkdir(parents=True, exist_ok=True)

        records = []
        for row in source:
            task_id = row['id']
            metadata = row['metadata']
            task_dir = image_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            images = list(row['images'])
            declared = list(metadata['image_filenames'])
            if not images:
                raise RuntimeError(f'VIALS task {task_id} has no image.')
            if len(declared) != len(images):
                raise RuntimeError(
                    f'VIALS task {task_id} declares {len(declared)} image names for '
                    f'{len(images)} images.'
                )

            filenames = [
                _safe_filename(
                    declared[i],
                    f'{task_id}.png' if len(images) == 1 else f'{task_id}_{i + 1}.png',
                )
                for i in range(len(images))
            ]
            relative_paths = []
            for filename, image in zip(filenames, images):
                image_bytes = image.get('bytes')
                if image_bytes is None:
                    image_bytes = Path(image['path']).read_bytes()
                image_file = task_dir / filename
                if not image_file.exists() or image_file.read_bytes() != image_bytes:
                    image_file.write_bytes(image_bytes)
                relative_paths.append(str(Path(task_id) / filename))

            records.append({
                'index': task_id,
                'question': html.unescape(row['question']),
                'answer': html.unescape(row['answer']),
                'image_path': json.dumps(relative_paths, ensure_ascii=False),
                'domain': row['domain'],
                'secondary_tags': json.dumps(
                    list(metadata.get('secondary_tags') or []), ensure_ascii=False
                ),
                'answer_lower_bound': _numeric_bound(metadata.get('answer_lower_bound')),
                'answer_upper_bound': _numeric_bound(metadata.get('answer_upper_bound')),
            })

        data = pd.DataFrame(records)
        if len(data) != VIALS_EXPECTED_TASKS:
            raise RuntimeError(
                f'VIALS expected {VIALS_EXPECTED_TASKS} tasks, got {len(data)}.'
            )
        dump(data, str(self._data_file()))
        (self._data_dir() / VIALS_INFO_FILE).write_text(
            json.dumps(self._dataset_metadata(), indent=2, sort_keys=True) + '\n'
        )
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_paths = self._image_paths(line)
        if not image_paths:
            raise FileNotFoundError(f'VIALS task {line["index"]} has no image path.')
        messages = [dict(role='system', type='text', value=VIALS_SYSTEM_PROMPT)]
        messages.extend(dict(type='image', value=str(path)) for path in image_paths)
        messages.append(dict(
            type='text',
            value=f'{line["question"]}\n\n{VIALS_USER_INSTRUCTION_TAIL}',
        ))
        return messages

    @staticmethod
    def _judge_line(line):
        return {
            'Question': line['question'],
            'GTFA': line['answer'],
            'LeadBoundLower': line.get('answer_lower_bound'),
            'LeadBoundUpper': line.get('answer_upper_bound'),
        }

    @staticmethod
    def _judge_cache_key(line, multimodal_fallback):
        payload = {
            'stage': VIALS_JUDGE_CACHE_STAGE,
            'index': str(line['index']),
            'question': str(line['question']),
            'reference': str(line['answer']),
            'candidate': extract_answer_block(str(line.get('prediction', ''))) or '',
            'lower': _numeric_bound(line.get('answer_lower_bound')),
            'upper': _numeric_bound(line.get('answer_upper_bound')),
            'multimodal_fallback': bool(multimodal_fallback),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _judge_cache_entry_failed(entry):
        required = {
            'extracted_answer', 'judge_reasoning', 'judge_response',
            'judge_multimodal', 'correct', 'judge_error',
        }
        return (
            not isinstance(entry, dict)
            or not required.issubset(entry)
            or bool(entry.get('judge_error'))
        )

    @staticmethod
    def _call_judge(judge, messages, max_retries=VIALS_JUDGE_MAX_RETRIES):
        last_response = ''
        for attempt in range(max_retries):
            last_response = judge.generate(messages)
            parsed = parse_judge_response(last_response)
            if parsed is not None:
                return parsed, last_response, attempt + 1
        raise ValueError(
            f'Judge returned invalid JSON after {max_retries} attempts: '
            f'{last_response[:200]!r}'
        )

    @classmethod
    def _evaluate_single(cls, judge, line, multimodal_fallback=True,
                         max_retries=VIALS_JUDGE_MAX_RETRIES):
        extracted_answer = extract_answer_block(str(line['prediction']))
        judged = {
            'extracted_answer': extracted_answer,
            'judge_reasoning': None,
            'judge_response': None,
            'judge_multimodal': False,
            'correct': False,
            'judge_error': None,
        }
        if not extracted_answer:
            judged['judge_reasoning'] = 'No final answer was extracted.'
            return judged

        item = cls._judge_line(line)
        prompt = build_judge_user_message(item, extracted_answer)
        messages = [
            dict(role='system', type='text', value=VIALS_JUDGE_SYSTEM),
            dict(type='text', value=prompt),
        ]
        try:
            result, response, _ = cls._call_judge(judge, messages, max_retries)
        except ValueError as exc:
            judged['judge_error'] = str(exc)
            judged['judge_reasoning'] = f'[JudgeError after {max_retries} retries] {exc}'
            return judged

        judged.update({
            'correct': result['correct'],
            'judge_reasoning': result['judge_reasoning'],
            'judge_response': response,
        })
        if not (multimodal_fallback and result['judge_unverifiable']):
            return judged

        image_paths = [Path(path) for path in line['_judge_image_paths']]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if not image_paths or missing:
            judged['correct'] = False
            judged['judge_error'] = f'image not found: {", ".join(missing) or "no image supplied"}'
            judged['judge_reasoning'] = f'[MultimodalJudgeError] {judged["judge_error"]}'
            judged['judge_multimodal'] = True
            return judged

        mm_prompt = build_multimodal_judge_user_message(item, extracted_answer)
        mm_messages = [dict(role='system', type='text', value=VIALS_MULTIMODAL_JUDGE_SYSTEM)]
        mm_messages.extend(dict(type='image', value=str(path)) for path in image_paths)
        mm_messages.append(dict(type='text', value=mm_prompt))
        try:
            mm_result, mm_response, _ = cls._call_judge(judge, mm_messages, max_retries)
        except ValueError as exc:
            judged['correct'] = False
            judged['judge_error'] = str(exc)
            judged['judge_reasoning'] = f'[MultimodalJudgeError after {max_retries} retries] {exc}'
            judged['judge_multimodal'] = True
            return judged

        judged.update({
            'correct': mm_result['correct'],
            'judge_reasoning': mm_result['judge_reasoning'],
            'judge_response': mm_response,
            'judge_multimodal': True,
        })
        return judged

    @staticmethod
    def _compute_metrics(data):
        total = len(data)
        hit = data['correct'].map(
            lambda value: value.lower() in {'1', 'true', 'yes'}
            if isinstance(value, str) else bool(value)
        )
        correct = int(hit.sum()) if total else 0
        metrics = {'Overall': round(correct / total * 100, 4) if total else 0.0}
        for domain in sorted(data['domain'].dropna().unique()):
            subset = data[data['domain'] == domain]
            metrics[f'Domain/{domain}'] = round(
                hit[subset.index].mean() * 100, 4
            )
        return metrics

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(value) if pd.notna(value) else '' for value in data['prediction']]
        data['answer'] = [str(value) if pd.notna(value) else '' for value in data['answer']]

        judge_model = judge_kwargs.pop('model', self.DEFAULT_JUDGE_MODEL)
        warn_if_judge_model_changed(
            judge_model,
            (self.DEFAULT_JUDGE_MODEL, 'openai/gpt-5-mini'),
            VIALS_DATASET,
        )
        detail_file = get_intermediate_file_path(
            eval_file, self.PASS_K_DETAIL_SUFFIX, self.PASS_K_DETAIL_FORMAT)
        nproc = judge_kwargs.pop('nproc', 4)
        multimodal_fallback = judge_kwargs.pop('multimodal_fallback', True)
        max_retries = judge_kwargs.pop('retry', VIALS_JUDGE_MAX_RETRIES)
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
            raise ValueError(f'VIALS judge retry must be a positive integer, got {max_retries!r}')
        max_tokens = judge_kwargs.pop('max_tokens', VIALS_JUDGE_MAX_TOKENS)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError(f'VIALS judge max_tokens must be a positive integer, got {max_tokens!r}')
        cache_file = get_judge_cache_file(eval_file, VIALS_JUDGE_CACHE_STAGE, judge_model)
        answers = load_judge_cache(cache_file)

        records = data.to_dict('records')
        for record in records:
            record['_judge_image_paths'] = self._image_paths(record)
        row_keys = [
            self._judge_cache_key(record, multimodal_fallback)
            for record in records
        ]
        pending = [
            (record, key)
            for record, key in zip(records, row_keys)
            if key not in answers or self._judge_cache_entry_failed(answers[key])
        ]

        if pending:
            judge_kwargs = dict(judge_kwargs)
            judge_kwargs.pop('response_format', None)
            # VIALS owns the retry loop. Keep the underlying judge transport at
            # one attempt so retries are not multiplied inside BaseAPI.generate.
            judge_kwargs['retry'] = 1
            judge = build_judge(
                model=judge_model,
                max_tokens=max_tokens,
                response_format=VIALS_JUDGE_RESPONSE_FORMAT,
                img_detail='high',
                **judge_kwargs,
            )

            answers = run_cached_tasks(
                self._evaluate_single,
                [(judge, record, multimodal_fallback, max_retries) for record, _ in pending],
                [key for _, key in pending],
                cache_file,
                nproc=nproc,
                chunksize=nproc,
            )

        judged = data.copy()
        for column in ('extracted_answer', 'judge_reasoning', 'judge_response'):
            judged[column] = [answers[key][column] for key in row_keys]
        judged['judge_multimodal'] = [answers[key]['judge_multimodal'] for key in row_keys]
        judged['correct'] = [answers[key]['correct'] for key in row_keys]
        judged['judge_error'] = [answers[key]['judge_error'] for key in row_keys]
        storage = get_judge_detail_file(
            eval_file, VIALS_JUDGE_CACHE_STAGE, judge_model, 'tsv')
        dump(judged, storage)

        dump(judged, detail_file)
        metrics = self._compute_metrics(judged)
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(metrics, score_file)
        return metrics

    @staticmethod
    def metrics_to_frame(metrics):
        return d2df(metrics)
