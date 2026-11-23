import hashlib
import json
from pathlib import Path

import pandas as pd

from vlmeval.smp import get_logger, load
from .image_base import ImageBaseDataset

logger = get_logger(__name__)


class VisualOverload(ImageBaseDataset):
    """VisualOverload inference with submission export for privately held answers."""

    TYPE = 'VQA'
    force_use_dataset_prompt = True
    EVALUATOR_URL = 'https://huggingface.co/spaces/paulgavrikov/visualoverload-submit'

    @classmethod
    def supported_datasets(cls):
        return ['VisualOverload']

    def load_data(self, dataset):
        from datasets import Image, load_dataset

        # Keep the original image bytes: resolution and small details matter here.
        source = load_dataset('paulgavrikov/visualoverload', split='test')
        source = source.cast_column('image', Image(decode=False))
        image_root = Path(self.img_root)
        image_root.mkdir(parents=True, exist_ok=True)
        records = []
        for item in source:
            image = item['image']
            image_bytes = image['bytes']
            if image_bytes is None:
                image_bytes = Path(image['path']).read_bytes()
            image_path = image_root / f'{hashlib.sha256(image_bytes).hexdigest()}.jpg'
            if not image_path.exists():
                image_path.write_bytes(image_bytes)
            records.append({
                'index': len(records),
                'question_id': item['question_id'],
                'question': item['default_prompt'],
                'image_path': str(image_path),
                'question_type': item['question_type'],
                'category': item['category'],
                'difficulty': item['difficulty'],
            })
        return pd.DataFrame(records)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [
            dict(type='image', value=line['image_path']),
            dict(type='text', value=line['question']),
        ]

    @classmethod
    def export_submission(cls, eval_file):
        """Export raw predictions using the official question_id/response contract."""
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        if data['question_id'].isna().any() or data['question_id'].duplicated().any():
            raise ValueError('VisualOverload submissions require unique, non-null question_id values.')
        records = [
            {'question_id': str(row['question_id']),
             'response': '' if pd.isna(row['prediction']) else str(row['prediction'])}
            for _, row in data.iterrows()
        ]
        submission_file = Path(eval_file).with_name(Path(eval_file).stem + '_submission.json')
        with submission_file.open('w', encoding='utf-8') as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2)
        logger.info(
            'VisualOverload answers are private. Submit %s to %s', submission_file, cls.EVALUATOR_URL)
        return str(submission_file)

    def evaluate(self, eval_file, **judge_kwargs):
        self.export_submission(eval_file)
        return {}
