import json
from io import BytesIO
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset, Image
from PIL import Image as PILImage

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE, build_dataset
from vlmeval.dataset.visualoverload import VisualOverload
from vlmeval.inference import infer_data_api
from vlmeval.smp import dump


def test_load_and_prompt_preserve_original_image(tmp_path, monkeypatch):
    monkeypatch.setenv('LMUData', str(tmp_path))
    stream = BytesIO()
    PILImage.new('RGB', (3840, 2160)).save(stream, format='JPEG')
    image_bytes = stream.getvalue()
    records = [{
        'question_id': question_id,
        'image': {'bytes': image_bytes, 'path': None},
        'default_prompt': prompt,
        'question_type': question_type,
        'category': question_type,
        'difficulty': 'hard',
    } for question_id, prompt, question_type in [
        ('001abc', 'Choose A or B.', 'choice'),
        ('count', 'How many animals?', 'counting'),
        ('ocr', 'Read the ninth word.', 'ocr'),
    ]]
    source = Dataset.from_list(records).cast_column('image', Image(decode=False))
    with mock.patch('datasets.load_dataset', return_value=source) as loader:
        dataset = build_dataset('VisualOverload')
    loader.assert_called_once_with('paulgavrikov/visualoverload', split='test')
    assert DATASET_TYPE('VisualOverload') == 'VQA'
    assert DATASET_MODALITY('VisualOverload') == 'IMAGE'
    assert 'answer' not in dataset.data
    assert dataset.data['question_id'].tolist() == ['001abc', 'count', 'ocr']
    assert len(set(dataset.data['image_path'])) == 1
    assert Path(dataset.data.iloc[0]['image_path']).read_bytes() == image_bytes
    for index, record in enumerate(records):
        prompt = dataset.build_prompt(index)
        assert prompt[1] == {'type': 'text', 'value': record['default_prompt']}
        with PILImage.open(prompt[0]['value']) as image:
            assert image.size == (3840, 2160)


def test_inference_uses_official_prompt_even_with_model_custom_prompts(tmp_path):
    dataset = object.__new__(VisualOverload)
    dataset.dataset_name = 'VisualOverload'
    dataset.data = pd.DataFrame([{
        'index': 0, 'image_path': 'image.jpg', 'question': 'Official default_prompt.',
    }])

    class CustomPromptModel:
        is_api = True

        def use_custom_prompt(self, dataset_name):
            return True

        def build_prompt(self, line, dataset):
            raise AssertionError('The model must not replace the official prompt.')

        def generate(self, message, dataset):
            assert message == [
                {'type': 'image', 'value': 'image.jpg'},
                {'type': 'text', 'value': 'Official default_prompt.'},
            ]
            return 'B'

    result = infer_data_api(
        model=CustomPromptModel(), work_dir=str(tmp_path), model_name='test-model', dataset=dataset, api_nproc=1
    )
    assert result == {0: 'B'}


@pytest.mark.parametrize('suffix', ['.xlsx', '.tsv'])
def test_submission_preserves_ids_and_freeform_responses(tmp_path, suffix):
    prediction_file = tmp_path / f'model_VisualOverload{suffix}'
    dump(pd.DataFrame({
        'question_id': ['001abc', 'count', 'ocr', 'empty'],
        'prediction': ['B', 'There are 17 animals.', 'café', None],
    }), str(prediction_file))
    submission = VisualOverload.export_submission(str(prediction_file))
    assert json.loads(Path(submission).read_text()) == [
        {'question_id': '001abc', 'response': 'B'},
        {'question_id': 'count', 'response': 'There are 17 animals.'},
        {'question_id': 'ocr', 'response': 'café'},
        {'question_id': 'empty', 'response': ''},
    ]


@pytest.mark.parametrize('ids', [['same', 'same'], ['valid', None]])
def test_submission_rejects_invalid_ids(tmp_path, ids):
    prediction_file = tmp_path / 'predictions.xlsx'
    dump(pd.DataFrame({'question_id': ids, 'prediction': ['A', 'B']}), str(prediction_file))
    with pytest.raises(ValueError, match='question_id'):
        VisualOverload.export_submission(str(prediction_file))
