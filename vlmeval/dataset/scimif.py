import io
import json
import os.path as osp
import re
from typing import Any

import pandas as pd
from PIL import Image

from vlmeval.smp import dump, encode_image_to_base64, get_intermediate_file_path, get_logger, load
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge
from .utils.scimif_eval import evaluate_record, summarize_results

logger = get_logger(__name__)


class _JudgeClient:

    def __init__(self, judge):
        self.judge = judge

    def __call__(self, prompt: str) -> str:
        result = self.judge.generate(prompt)
        fail_message = getattr(self.judge, 'fail_msg', '')
        if not result or (fail_message and fail_message in result):
            raise RuntimeError('The judge model failed to return a response.')
        return str(result)


def _evaluate_scimif_row(item, llm_client, judge_model):
    return evaluate_record(item, llm_client=llm_client, judge_model=judge_model)


class SciMIF(ImageBaseDataset):
    """SciMIF benchmark loaded from its Hugging Face dataset repository."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE_MODEL = 'gpt-4.1'

    HF_REPO_ID = 'Sheryle7436/SciMIF'
    HF_CONFIG = 'default'
    HF_SPLIT = 'test'

    @classmethod
    def supported_datasets(cls):
        return ['SciMIF']

    def __init__(self, dataset='SciMIF', skip_noimg=False):
        # SciMIF contains both multimodal and text-only samples. Text-only
        # samples must remain in the benchmark.
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    @staticmethod
    def _to_pil_image(value: Any) -> Image.Image:
        """Convert a decoded or non-decoded Hugging Face image to PIL."""

        if isinstance(value, Image.Image):
            return value

        if isinstance(value, dict):
            image_bytes = value.get('bytes')
            image_path = value.get('path')

            if image_bytes is not None:
                if isinstance(image_bytes, memoryview):
                    image_bytes = image_bytes.tobytes()
                with Image.open(io.BytesIO(image_bytes)) as image:
                    return image.copy()

            if image_path:
                with Image.open(image_path) as image:
                    return image.copy()

        if isinstance(value, str):
            with Image.open(value) as image:
                return image.copy()

        raise TypeError(f'Unsupported SciMIF image value: {type(value)!r}')

    @staticmethod
    def _as_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [item for item in value if item is not None]
        return [value]

    @classmethod
    def _convert_record(cls, record: dict, index: int) -> dict:
        sample_id = str(record.get('sample_id') or f'SciMIF_{index}')

        images = cls._as_list(record.get('image'))
        encoded_images = [encode_image_to_base64(cls._to_pil_image(image)) for image in images]

        image_paths = [
            str(path).removeprefix('images/') for path in cls._as_list(record.get('image_path'))
            if str(path).strip() and str(path).strip() != '[]'
        ]
        if len(image_paths) != len(encoded_images):
            image_paths = [f'{sample_id}_{image_index}.jpg' for image_index in range(len(encoded_images))]

        answer = record.get('answer')
        has_answer = answer is not None and str(answer).strip() != ''
        if answer is None:
            answer = ''
        elif not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)

        return {
            'index': index,
            'id': record.get('id'),
            'sample_id': sample_id,
            'split': cls.HF_SPLIT,
            'category': record.get('subject', ''),
            'subject': record.get('subject', ''),
            'task': record.get('task', ''),
            'question': record.get('edit_question', ''),
            'edit_question': record.get('edit_question', ''),
            'original_question': record.get('original_question', ''),
            'answer': answer,
            'has_answer': has_answer,
            'choose_instruction': json.dumps(record.get('choose_instruction') or [], ensure_ascii=False),
            'instruction_list': json.dumps(record.get('instruction_list') or [], ensure_ascii=False),
            # ImageBaseDataset parses JSON lists and writes decoded images to
            # $LMUData/images/SciMIF when build_prompt() is called.
            # Keep text-only samples truly empty. ImageBaseDataset interprets
            # short non-empty strings (such as "[]") as references to another
            # sample's image, which is not the meaning here.
            'image': json.dumps(encoded_images) if encoded_images else None,
            'image_path': json.dumps(image_paths, ensure_ascii=False),
        }

    def load_data(self, dataset):
        if dataset != 'SciMIF':
            raise ValueError(f'Unsupported dataset name: {dataset!r}')

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError('Loading SciMIF requires the `datasets` package. '
                              'Install it with `pip install datasets`.') from exc

        hf_dataset = load_dataset(
            self.HF_REPO_ID,
            self.HF_CONFIG,
            split=self.HF_SPLIT,
        )

        rows = [self._convert_record(record, index) for index, record in enumerate(hf_dataset)]
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_value = line.get('image')
        has_image = (bool(image_value)
                     if isinstance(image_value, str) else isinstance(image_value, list) and len(image_value) > 0)
        image_paths = self.dump_image(line) if has_image else []
        messages = [dict(type='image', value=image_path) for image_path in image_paths]
        messages.append(dict(type='text', value=line['question']))
        return messages

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if 'prediction' not in data:
            raise ValueError('SciMIF evaluation requires a `prediction` column.')

        judge_options = dict(judge_kwargs)
        nproc = judge_options.pop('nproc', 4)
        judge_name = judge_options.pop('model', cls.DEFAULT_JUDGE_MODEL)
        judge_options.pop('use_verifier', None)
        judge_options.pop('use_vllm', None)
        safe_judge_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(judge_name))

        detail_file = get_intermediate_file_path(eval_file, f'_{safe_judge_name}_details', 'xlsx')
        score_file = get_intermediate_file_path(eval_file, f'_{safe_judge_name}_score', 'csv')
        tmp_file = get_intermediate_file_path(eval_file, f'_{safe_judge_name}_tmp', 'pkl')

        records = data.to_dict(orient='records')
        keys = [str(record.get('index', position)) for position, record in enumerate(records)]
        cached = load(tmp_file) if osp.exists(tmp_file) else {}
        if not isinstance(cached, dict):
            cached = {}

        pending_records = []
        pending_keys = []
        for key, record in zip(keys, records):
            if key not in cached:
                pending_keys.append(key)
                pending_records.append(record)

        if pending_records:
            judge_options.setdefault('temperature', 0)
            judge_options.setdefault('timeout', 300)
            judge_options.setdefault('max_tokens', 1024)
            judge = build_judge(model=judge_name, **judge_options)
            assert judge.working(), ('SciMIF instruction evaluation requires a working judge API.\n' + DEBUG_MESSAGE)
            llm_client = _JudgeClient(judge)
            tasks = [dict(item=record, llm_client=llm_client, judge_model=judge_name) for record in pending_records]
            new_results = track_progress_rich(
                _evaluate_scimif_row,
                tasks,
                nproc=nproc,
                chunksize=nproc,
                keys=pending_keys,
                save=tmp_file,
            )
            cached.update(dict(zip(pending_keys, new_results)))
        else:
            logger.info(f'Reused all {len(cached)} cached SciMIF evaluation results.')

        evaluated_records = []
        for key, record in zip(keys, records):
            evaluation = cached[key]
            evaluated_records.append({**record, **evaluation})

        details = pd.DataFrame(evaluated_records)
        details['instruction_results'] = details['instruction_results'].map(
            lambda value: json.dumps(value, ensure_ascii=False))
        dump(details, detail_file)

        summary = pd.DataFrame(summarize_results(evaluated_records))
        dump(summary, score_file)
        logger.info(f'SciMIF detailed results saved to {detail_file}.')
        logger.info(f'SciMIF scores saved to {score_file}.')
        return summary
