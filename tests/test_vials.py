import importlib
import json
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, Image
from PIL import Image as PILImage

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE, build_dataset
from vlmeval.dataset.utils.judge_cache import get_judge_cache_file
from vlmeval.inference import infer_data_api
from vlmeval.smp import dump, load, toliststr

vials = importlib.import_module('vlmeval.dataset.vials')


def _png_bytes(color):
    stream = BytesIO()
    PILImage.new('RGB', (32, 24), color=color).save(stream, format='PNG')
    return stream.getvalue()


def test_vials_registration_and_official_defaults():
    assert DATASET_TYPE('VIALS') == 'VQA'
    assert DATASET_MODALITY('VIALS') == 'IMAGE'
    assert vials.VIALS.DEFAULT_JUDGE_MODEL == 'gpt-5-mini'
    assert vials.VIALS.DEFAULT_N_RUNS == 3
    assert vials.VIALS.PASS_K_HIT_COL == 'correct'
    assert vials.VIALS.PASS_K_DETAIL_FORMAT == 'tsv'


@pytest.mark.parametrize(
    'prediction, expected',
    [
        ('REASONING: x\nANSWER: 42', '42'),
        ('- **ANSWER:** $5.0$', '5.0'),
        ('1. ANSWER: lane 3\nANSWER: lane 4', 'lane 4'),
        ('The answer: 5', '5'),
        ('The value is 5', None),
        ('', None),
    ],
)
def test_extract_answer_block_follows_official_protocol(prediction, expected):
    assert vials.extract_answer_block(prediction) == expected


def test_load_data_writes_pinned_revision_and_ordered_images(tmp_path, monkeypatch):
    monkeypatch.setenv('LMUData', str(tmp_path))
    monkeypatch.setattr(vials, 'VIALS_EXPECTED_TASKS', 2)

    single_bytes = _png_bytes('red')
    first_panel = _png_bytes('green')
    second_panel = _png_bytes('blue')
    records = [
        {
            'id': 'single',
            'question': 'Count the red cells.',
            'images': [{'bytes': single_bytes, 'path': None}],
            'answer': '3',
            'domain': 'Cell counting/quantification',
            'metadata': {
                'secondary_tags': ['counting'],
                'image_filenames': ['single.png'],
                'answer_lower_bound': 2.0,
                'answer_upper_bound': 4.0,
            },
        },
        {
            'id': 'multi',
            'question': 'Compare the two panels.',
            'images': [
                {'bytes': first_panel, 'path': None},
                {'bytes': second_panel, 'path': None},
            ],
            'answer': 'second',
            'domain': 'Blotting',
            'metadata': {
                'secondary_tags': ['blot'],
                'image_filenames': ['first.png', 'second.png'],
                'answer_lower_bound': None,
                'answer_upper_bound': None,
            },
        },
    ]
    source = Dataset.from_list(records).cast_column('images', [Image(decode=False)])
    with mock.patch('datasets.load_dataset', return_value=source) as loader:
        dataset = build_dataset('VIALS')

    loader.assert_called_once_with(
        vials.VIALS_HF_REPO,
        vials.VIALS_HF_CONFIG,
        split=vials.VIALS_HF_SPLIT,
        revision=vials.VIALS_HF_REVISION,
    )
    assert len(dataset) == 2
    assert dataset.data['index'].tolist() == ['single', 'multi']
    assert dataset.data.iloc[0]['answer_lower_bound'] == 2.0
    assert np.isnan(dataset.data.iloc[1]['answer_lower_bound'])

    single_prompt = dataset.build_prompt(0)
    assert single_prompt[0] == {
        'role': 'system', 'type': 'text', 'value': vials.VIALS_SYSTEM_PROMPT,
    }
    assert [item['type'] for item in single_prompt] == ['text', 'image', 'text']
    assert Path(single_prompt[1]['value']).read_bytes() == single_bytes
    assert single_prompt[-1]['value'] == (
        'Count the red cells.\n\n' + vials.VIALS_USER_INSTRUCTION_TAIL
    )

    multi_prompt = dataset.build_prompt(1)
    assert [item['type'] for item in multi_prompt] == ['text', 'image', 'image', 'text']
    assert [Path(item['value']).read_bytes() for item in multi_prompt[1:3]] == [
        first_panel, second_panel,
    ]
    assert len(toliststr(dataset.data.iloc[1]['image_path'])) == 2
    metadata = json.loads((tmp_path / 'VIALS' / vials.VIALS_INFO_FILE).read_text())
    assert metadata['revision'] == vials.VIALS_HF_REVISION
    assert metadata['num_tasks'] == 2

    # A second construction must reuse the materialized LMUData cache.
    with mock.patch('datasets.load_dataset', side_effect=AssertionError('should reuse cache')):
        cached = build_dataset('VIALS')
    assert cached.data['index'].tolist() == ['single', 'multi']


def test_inference_keeps_official_prompt_when_model_has_custom_prompt(tmp_path):
    image = tmp_path / 'image.png'
    image.write_bytes(_png_bytes('white'))
    dataset = object.__new__(vials.VIALS)
    dataset.dataset_name = 'VIALS'
    dataset.img_root = str(tmp_path)
    dataset.data = pd.DataFrame([{
        'index': 'task',
        'question': 'What is shown?',
        'image_path': 'image.png',
    }])

    class CustomPromptModel:
        is_api = True

        def use_custom_prompt(self, dataset_name):
            return True

        def build_prompt(self, line, dataset):
            raise AssertionError('The model must not replace the VIALS prompt.')

        def generate(self, message, dataset):
            assert message[0] == {
                'role': 'system', 'type': 'text', 'value': vials.VIALS_SYSTEM_PROMPT,
            }
            assert message[1] == {'type': 'image', 'value': str(image)}
            assert message[2]['value'] == (
                'What is shown?\n\n' + vials.VIALS_USER_INSTRUCTION_TAIL
            )
            return 'REASONING: x\nANSWER: smoke'

    result = infer_data_api(
        model=CustomPromptModel(),
        work_dir=str(tmp_path),
        model_name='test-model',
        dataset=dataset,
        api_nproc=1,
    )
    assert result == {'task': 'REASONING: x\nANSWER: smoke'}


class _QueueJudge:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_judge_regrades_unverifiable_answers_with_all_images(tmp_path):
    image_a = tmp_path / 'a.png'
    image_b = tmp_path / 'b.png'
    image_a.write_bytes(_png_bytes('red'))
    image_b.write_bytes(_png_bytes('blue'))
    judge = _QueueJudge([
        json.dumps({'reasoning': 'UNVERIFIABLE: lane labels need the figure.', 'equivalent': False}),
        json.dumps({'reasoning': 'The labels correspond to lane 3.', 'equivalent': True}),
    ])
    line = {
        'question': 'Which lane contains the protein?',
        'answer': 'condition B',
        'answer_lower_bound': np.nan,
        'answer_upper_bound': np.nan,
        'prediction': 'REASONING: I inspected the blot.\nANSWER: lane 3',
        '_judge_image_paths': [image_a, image_b],
    }

    result = vials.VIALS._evaluate_single(judge, line, multimodal_fallback=True)

    assert result['extracted_answer'] == 'lane 3'
    assert result['correct'] is True
    assert result['judge_multimodal'] is True
    assert judge.calls[0][0]['value'] == vials.VIALS_JUDGE_SYSTEM
    assert judge.calls[1][0]['value'] == vials.VIALS_MULTIMODAL_JUDGE_SYSTEM
    assert [item['value'] for item in judge.calls[1][1:3]] == [str(image_a), str(image_b)]
    assert 'Lead-assigned' not in judge.calls[0][1]['value']


def test_judge_uses_numeric_bounds_and_treats_missing_answer_as_wrong(tmp_path):
    image = tmp_path / 'image.png'
    image.write_bytes(_png_bytes('white'))
    judge = _QueueJudge([
        json.dumps({'reasoning': 'The count is within the LeadBounds.', 'equivalent': True}),
    ])
    line = {
        'question': 'How many colonies?',
        'answer': '10',
        'answer_lower_bound': 9.0,
        'answer_upper_bound': 12.0,
        'prediction': 'REASONING: counted\nANSWER: 11',
        '_judge_image_paths': [image],
    }
    result = vials.VIALS._evaluate_single(judge, line)
    assert result['correct'] is True
    assert 'lower=9.0, upper=12.0' in judge.calls[0][1]['value']

    no_answer = dict(line, prediction='I cannot determine the answer.')
    result = vials.VIALS._evaluate_single(judge, no_answer)
    assert result['correct'] is False
    assert result['judge_reasoning'] == 'No final answer was extracted.'
    assert len(judge.calls) == 1


class _StaticJudge:
    def generate(self, messages):
        prompt = messages[-1]['value']
        equivalent = 'Candidate final answer:\nblue' in prompt
        return json.dumps({
            'reasoning': 'The candidate matches the reference answer.',
            'equivalent': equivalent,
        })


def test_evaluate_writes_pass_k_detail_and_domain_metrics(tmp_path, monkeypatch):
    image_root = tmp_path / 'images' / 'VIALS'
    image_root.mkdir(parents=True)
    image = image_root / 'image.png'
    image.write_bytes(_png_bytes('white'))
    prediction_file = tmp_path / 'model_VIALS.tsv'
    dump(pd.DataFrame([
        {
            'index': 'a', 'question': 'Color?', 'answer': 'blue', 'domain': 'Other',
            'image_path': 'image.png', 'prediction': 'REASONING: x\nANSWER: blue',
            'answer_lower_bound': np.nan, 'answer_upper_bound': np.nan,
        },
        {
            'index': 'b', 'question': 'Color?', 'answer': 'blue', 'domain': 'Other',
            'image_path': 'image.png', 'prediction': 'REASONING: x\nANSWER: red',
            'answer_lower_bound': np.nan, 'answer_upper_bound': np.nan,
        },
    ]), str(prediction_file))

    dataset = object.__new__(vials.VIALS)
    dataset.img_root = str(image_root)

    monkeypatch.setattr(vials, 'build_judge', lambda **kwargs: _StaticJudge())

    metrics = dataset.evaluate(str(prediction_file), nproc=1)

    detail = load(str(tmp_path / 'model_VIALS_results.tsv'))
    assert detail['index'].tolist() == ['a', 'b']
    assert detail['extracted_answer'].tolist() == ['blue', 'red']
    assert detail['correct'].tolist() == [True, False]
    assert metrics['Overall'] == 50.0
    assert metrics['Domain/Other'] == 50.0

    cache_file = get_judge_cache_file(
        str(prediction_file), vials.VIALS_JUDGE_CACHE_STAGE, 'gpt-5-mini')
    assert Path(cache_file).is_file()

    # A complete cache must be reused without constructing another judge.
    def fail_build_judge(**kwargs):
        raise AssertionError('complete VIALS cache should not rebuild the judge')

    monkeypatch.setattr(vials, 'build_judge', fail_build_judge)
    assert dataset.evaluate(str(prediction_file), nproc=1)['Overall'] == 50.0


def test_evaluate_invalidates_cache_when_prediction_changes(tmp_path, monkeypatch):
    image_root = tmp_path / 'images' / 'VIALS'
    image_root.mkdir(parents=True)
    image = image_root / 'image.png'
    image.write_bytes(_png_bytes('white'))
    prediction_file = tmp_path / 'model_VIALS.tsv'

    def write_predictions(answer):
        dump(pd.DataFrame([{
            'index': 'a', 'question': 'Color?', 'answer': 'blue', 'domain': 'Other',
            'image_path': 'image.png', 'prediction': f'REASONING: x\nANSWER: {answer}',
            'answer_lower_bound': np.nan, 'answer_upper_bound': np.nan,
        }]), str(prediction_file))

    dataset = object.__new__(vials.VIALS)
    dataset.img_root = str(image_root)
    build_calls = []

    def build_judge(**kwargs):
        build_calls.append(kwargs)
        return _StaticJudge()

    monkeypatch.setattr(vials, 'build_judge', build_judge)
    write_predictions('red')
    assert dataset.evaluate(str(prediction_file), nproc=1)['Overall'] == 0.0
    assert len(build_calls) == 1
    assert build_calls[0]['retry'] == 1
    assert build_calls[0]['max_tokens'] == vials.VIALS_JUDGE_MAX_TOKENS

    write_predictions('blue')
    assert dataset.evaluate(str(prediction_file), nproc=1)['Overall'] == 100.0
    assert len(build_calls) == 2


def test_failed_judge_cache_entries_are_pending_again():
    assert vials.VIALS._judge_cache_entry_failed({'judge_error': 'API failed'})
    assert vials.VIALS._judge_cache_entry_failed({'correct': False})
    valid = {
        'extracted_answer': 'blue',
        'judge_reasoning': 'Equivalent.',
        'judge_response': '{}',
        'judge_multimodal': False,
        'correct': True,
        'judge_error': None,
    }
    assert not vials.VIALS._judge_cache_entry_failed(valid)


def test_judge_kwargs_retry_controls_explicit_retry_loop(tmp_path, monkeypatch):
    image_root = tmp_path / 'images' / 'VIALS'
    image_root.mkdir(parents=True)
    image = image_root / 'image.png'
    image.write_bytes(_png_bytes('white'))
    prediction_file = tmp_path / 'model_VIALS.tsv'
    dump(pd.DataFrame([{
        'index': 'a', 'question': 'Color?', 'answer': 'blue', 'domain': 'Other',
        'image_path': 'image.png', 'prediction': 'REASONING: x\nANSWER: blue',
        'answer_lower_bound': np.nan, 'answer_upper_bound': np.nan,
    }]), str(prediction_file))

    judge = _QueueJudge([
        'not json',
        json.dumps({'reasoning': 'Equivalent.', 'equivalent': True}),
    ])
    build_kwargs = []

    def build_judge(**kwargs):
        build_kwargs.append(kwargs)
        return judge

    dataset = object.__new__(vials.VIALS)
    dataset.img_root = str(image_root)
    monkeypatch.setattr(vials, 'build_judge', build_judge)

    metrics = dataset.evaluate(str(prediction_file), nproc=1, retry=2, max_tokens=77)

    assert build_kwargs[0]['retry'] == 1
    assert build_kwargs[0]['max_tokens'] == 77
    assert len(judge.calls) == 2
    assert metrics['Overall'] == 100.0
