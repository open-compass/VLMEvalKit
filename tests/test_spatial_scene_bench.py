import pandas as pd
import pytest
from PIL import Image

from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.dataset.spatial_scene_bench import (SpatialSceneBench, mean_relative_accuracy,
                                                 parse_answer)
from vlmeval.inference import infer_data_api
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin


def test_spatial_scene_bench_is_registered():
    assert 'SpatialSceneBench' in SUPPORTED_DATASETS
    assert 'SpatialSceneBench_mini' in SUPPORTED_DATASETS


@pytest.mark.parametrize('answer_type,answer_format', [
    ('count', 'Answer with a single integer.'),
    ('yesno', 'Answer with yes or no.'),
    ('choice', 'Answer with the letter of the correct option.'),
    ('numeric', 'Answer with a single number.'),
])
@pytest.mark.parametrize('frame_count', [1, 16])
@pytest.mark.parametrize('use_custom_prompt', [True, False])
def test_inference_uses_dataset_prompt_and_preserves_all_images(
    answer_type, answer_format, frame_count, use_custom_prompt, tmp_path, monkeypatch
):
    image_paths = []
    for frame_index in range(frame_count):
        image_path = tmp_path / f'frame_{frame_index}.png'
        Image.new('RGB', (2, 2)).save(image_path)
        image_paths.append(str(image_path))
    data = pd.DataFrame([{
        'index': 0,
        'question': 'Benchmark question.',
        'answer_type': answer_type,
        'image_path': image_paths,
    }])
    monkeypatch.setattr(SpatialSceneBench, 'load_data', lambda self, dataset: data)
    dataset = SpatialSceneBench('SpatialSceneBench_mini')
    dataset.img_root = str(tmp_path)
    model = Qwen2VLPromptMixin(use_custom_prompt=use_custom_prompt)
    monkeypatch.setattr(model, 'is_api', True, raising=False)
    received = []

    def generate(message, dataset):
        received.append(message)
        return 'test prediction'

    monkeypatch.setattr(model, 'generate', generate, raising=False)
    result = infer_data_api(
        model=model, work_dir=str(tmp_path), model_name='test-model', dataset=dataset, api_nproc=1
    )
    expected = [{'type': 'image', 'value': path} for path in image_paths]
    expected.append({'type': 'text', 'value': f'Benchmark question.\n{answer_format}'})
    assert received == [expected]
    assert result == {0: 'test prediction'}


@pytest.mark.parametrize(
    'text,answer_type,options,expected',
    [
        ('3', 'count', None, '3'),
        ('There are three cubes.', 'count', None, '3'),
        ('<answer>2.0</answer>', 'count', None, '2'),
        ('1.25 meters', 'numeric', None, '1.25'),
        ('Yes, it is.', 'yesno', None, 'yes'),
        ('(C)', 'choice', ['a', 'b', 'c', 'd'], 'C'),
        ('The answer is B.', 'choice', ['a', 'b', 'c', 'd'], 'B'),
        ('blue cube', 'choice', ['red sphere', 'blue cube', 'gray cylinder', 'metal cube'], 'B'),
        ('I am not sure', 'count', None, None),
    ],
)
def test_parse_answer(text, answer_type, options, expected):
    assert parse_answer(text, answer_type, options) == expected


def test_mean_relative_accuracy():
    assert mean_relative_accuracy(2.0, 2.0) == 1.0
    assert mean_relative_accuracy(2.2, 2.0) == pytest.approx(0.8)
    assert mean_relative_accuracy(10.0, 2.0) == 0.0


def test_compute_metrics():
    data = pd.DataFrame({
        'question_type': ['count', 'count', 'relation', 'video_distance', 'closest'],
        'answer_type': ['count', 'count', 'yesno', 'numeric', 'choice'],
        'answer': ['3', '4', 'no', '2.0', 'B'],
        'options': ['[]', '[]', '[]', '[]', '["red sphere", "blue cube", "gray cube", "ball"]'],
        'prediction': ['3', '3', 'yes', '2.0', 'blue cube'],
    })
    metrics = SpatialSceneBench.compute_metrics(data)

    assert metrics['count'] == 50.0
    assert metrics['relation'] == 0.0
    assert metrics['video_distance'] == 100.0
    assert metrics['closest'] == 100.0
    assert metrics['Image'] == pytest.approx((50.0 + 0.0 + 100.0) / 3, abs=0.01)
    assert metrics['Video'] == 100.0
    assert metrics['Overall'] == 62.5
    assert metrics['Count_MRA'] == 75.0
