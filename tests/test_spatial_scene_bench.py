import pandas as pd
import pytest
from PIL import Image

from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.dataset.spatial_scene_bench import (SpatialSceneBench, mean_relative_accuracy,
                                                 parse_answer)
from vlmeval.inference import infer_data_api
from vlmeval.smp.dataset_alias import resolve_dataset_spec
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin


def test_spatial_scene_bench_is_registered():
    assert 'SpatialSceneBench' in SUPPORTED_DATASETS
    assert 'SpatialSceneBench_mini' in SUPPORTED_DATASETS


@pytest.mark.parametrize('dataset_name', ['SpatialSceneBench', 'SpatialSceneBench_mini'])
@pytest.mark.parametrize('frame_count', [1, 16])
@pytest.mark.parametrize('max_frames,sampled_indices', [
    (None, [0, 2, 4, 6, 9, 11, 13, 15]),
    (1, [0]),
    (4, [0, 5, 10, 15]),
    (16, list(range(16))),
])
def test_frame_sampling_uses_official_sampling(
    dataset_name, frame_count, max_frames, sampled_indices, tmp_path, monkeypatch
):
    from run import build_dataset_from_spec

    paths = []
    for frame_index in range(frame_count):
        path = tmp_path / f'frame_{frame_index}.png'
        Image.new('RGB', (2, 2), color=(frame_index, 0, 0)).save(path)
        paths.append(str(path))
    data = pd.DataFrame([{
        'index': 0, 'question': 'Benchmark question.', 'answer_type': 'count', 'image_path': paths,
    }])
    monkeypatch.setattr(SpatialSceneBench, 'load_data', lambda self, dataset: data)
    data_config = None if max_frames is None else {
        dataset_name: {'class': 'SpatialSceneBench', 'dataset': dataset_name, 'max_frames': max_frames},
    }
    spec = resolve_dataset_spec(dataset_name, data_config)
    dataset = build_dataset_from_spec(spec)
    dataset.img_root = str(tmp_path)
    assert dataset.max_frames == (8 if max_frames is None else max_frames)
    assert dataset.dataset_name == dataset_name
    expected_indices = [0] if frame_count == 1 else sampled_indices
    expected = [{'type': 'image', 'value': paths[i]} for i in expected_indices]
    expected.append({'type': 'text', 'value': 'Benchmark question.\nAnswer with a single integer.'})
    assert dataset.build_prompt(0) == expected


@pytest.mark.parametrize('max_frames', [0, -1, 1.5, None, True])
def test_invalid_frame_count(max_frames):
    with pytest.raises(ValueError, match='max_frames must be a positive integer'):
        SpatialSceneBench(max_frames=max_frames)


def test_short_frame_list_is_preserved(tmp_path, monkeypatch):
    paths = []
    for frame_index in range(7):
        path = tmp_path / f'frame_{frame_index}.png'
        Image.new('RGB', (2, 2)).save(path)
        paths.append(str(path))
    data = pd.DataFrame([{
        'index': 0, 'question': 'Benchmark question.', 'answer_type': 'count', 'image_path': paths,
    }])
    monkeypatch.setattr(SpatialSceneBench, 'load_data', lambda self, dataset: data)
    dataset = SpatialSceneBench(max_frames=8)
    dataset.img_root = str(tmp_path)
    assert [item['value'] for item in dataset.build_prompt(0) if item['type'] == 'image'] == paths


@pytest.mark.parametrize('answer_type,answer_format', [
    ('count', 'Answer with a single integer.'),
    ('yesno', 'Answer with yes or no.'),
    ('choice', 'Answer with the letter of the correct option.'),
    ('numeric', 'Answer with a single number.'),
])
@pytest.mark.parametrize('frame_count', [1, 16])
@pytest.mark.parametrize('use_custom_prompt', [True, False])
def test_inference_preserves_answer_format(
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
    expected_indices = [0] if frame_count == 1 else [0, 2, 4, 6, 9, 11, 13, 15]
    expected = [{'type': 'image', 'value': image_paths[i]} for i in expected_indices]
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
