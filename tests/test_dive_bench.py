import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE, SUPPORTED_DATASETS, build_dataset
from vlmeval.dataset.dive_bench import (EDUCATIONAL, EDUCATIONAL_POST_PROMPT, HIGH_MOTION, PREVIEW,
                                        DIVEBench, ordered_annotation_sha256, sample_indices)
from vlmeval.dataset.utils.dive_bench import (_compute_cer, _compute_exact_match,
                                              _compute_token_f1, _compute_wer,
                                              _grid_sequence_metrics, parse_grid_sequence)


@pytest.fixture
def release(tmp_path, monkeypatch):
    (tmp_path / 'cache').mkdir()
    monkeypatch.setenv('LMUData', str(tmp_path / 'cache'))
    root = tmp_path / 'videos'
    (root / 'egodex' / 'action_a').mkdir(parents=True)
    (root / 'egodex' / 'action_b').mkdir(parents=True)
    (root / 'egodex' / 'action_a' / '0.mp4').touch()
    (root / 'egodex' / 'action_b' / '0.mp4').touch()
    (root / 'lecture.mp4').touch()

    def make(high_motion=False):
        count = 3243 if high_motion else 634
        rows = []
        for index in range(count):
            rows.append({
                'video_path': ('egodex/action_a/0.mp4' if index % 2 else 'egodex/action_b/0.mp4')
                if high_motion else 'lecture.mp4',
                'qid': str(index), 'question': 'Track the hand.\n\nWe consider all 10 frames',
                'answer': json.dumps(['topleft', 'middle', 'bottomright'] * 3 + ['middle'])
                if high_motion else 'reference words',
                'frame_count': 10,
            })
        path = tmp_path / ('motion.parquet' if high_motion else 'educational.parquet')
        pd.DataFrame(rows).to_parquet(path)
        key = HIGH_MOTION if high_motion else EDUCATIONAL
        spec = dict(DIVEBench.SPECS[key], sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if high_motion:
            spec['ordered_content_sha256'] = ordered_annotation_sha256(pd.DataFrame(rows))
        monkeypatch.setitem(DIVEBench.SPECS, key, spec)
        return path, root
    return make


def test_registration():
    for task in DIVEBench.supported_datasets():
        assert task in SUPPORTED_DATASETS
        assert DATASET_MODALITY(task) == 'VIDEO'
        assert DATASET_TYPE(task) == 'VQA'


def test_educational_suffix_matches_published_protocol():
    assert EDUCATIONAL_POST_PROMPT == (
        ' Provide the most complete answer possible. For subtitle or OCR questions, '
        'reproduce the relevant text from the video instead of answering with only the video id.'
    )


def test_owner_verified_high_motion_source_metadata():
    spec = DIVEBench.SPECS[HIGH_MOTION]
    assert spec['revision'] == 'd44407f607fdf020c59b816884f06ed6d453cf26'
    assert spec['sha256'] == '518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd'
    assert spec['compatible_sha256'] == (
        '39f9da7aca9020d79f383953646a5893f09c6f8e5f60433560011280ee987b2d',
    )
    assert spec['ordered_content_sha256'] == (
        '90ee915016105f6a709f391e8a03a6d0e99bc5c908f945cdf7b80d0cb289e789'
    )


def test_motion_accepts_only_two_audited_serializations(release, monkeypatch, tmp_path):
    annotation, root = release(True)
    alternate = tmp_path / 'compatible.parquet'
    unexpected = tmp_path / 'unreviewed.parquet'
    raw = pd.read_parquet(annotation)
    raw.to_parquet(alternate, compression='gzip')
    raw.to_parquet(unexpected, compression=None)
    assert len({hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (annotation, alternate, unexpected)}) == 3
    monkeypatch.setitem(DIVEBench.SPECS[HIGH_MOTION], 'compatible_sha256', (
        hashlib.sha256(alternate.read_bytes()).hexdigest(),
    ))
    first = DIVEBench(dataset=PREVIEW, annotation_file=annotation, data_root=root)
    second = DIVEBench(dataset=PREVIEW, annotation_file=alternate, data_root=root)
    pd.testing.assert_frame_equal(first.data, second.data)
    with pytest.raises(ValueError, match='SHA-256'):
        DIVEBench(dataset=PREVIEW, annotation_file=unexpected, data_root=root)


@pytest.mark.parametrize('change', ['reorder', 'tail'])
def test_motion_content_guard_checks_full_release_before_preview(release, monkeypatch, change):
    annotation, root = release(True)
    raw = pd.read_parquet(annotation)
    if change == 'reorder':
        raw = raw.iloc[::-1]
    else:
        raw.loc[2000, 'question'] = 'A changed question outside the preview'
    raw.to_parquet(annotation)
    # Even a byte-allowlisted file must independently match ordered content.
    monkeypatch.setitem(DIVEBench.SPECS[HIGH_MOTION], 'sha256', hashlib.sha256(annotation.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='content/order'):
        DIVEBench(dataset=PREVIEW, annotation_file=annotation, data_root=root)


def test_cached_annotations_are_published_atomically(release, monkeypatch, tmp_path):
    from vlmeval.dataset import dive_bench as module
    annotation, root = release()
    DIVEBench(annotation_file=annotation, data_root=root)
    target = next((tmp_path / 'cache' / 'DIVE-Bench').glob('*.tsv'))
    original = target.read_bytes()
    dump = module.dump
    writes = []

    def checked_dump(data, path):
        assert Path(path) != target
        assert target.read_bytes() == original
        dump(data, path)
        assert target.read_bytes() == original
        writes.append(path)

    monkeypatch.setattr(module, 'dump', checked_dump)
    DIVEBench(annotation_file=annotation, data_root=root)
    assert len(writes) == 1
    assert target.read_bytes() == original
    assert list(target.parent.glob('*.tsv')) == [target]


def test_registered_initialization_and_authoritative_answers(release, tmp_path):
    annotation, root = release()
    dataset = build_dataset(EDUCATIONAL, annotation_file=annotation, data_root=root)
    assert len(dataset) == 634
    assert dataset.nframe == 8
    assert dataset.data['index'].is_unique
    assert len(dataset.videos) == 1
    messages = dataset.build_prompt(0, video_llm=True)
    assert messages == [{'type': 'video', 'value': str(root / 'lecture.mp4')},
                        {'type': 'text', 'value': dataset.data.iloc[0]['question'] + EDUCATIONAL_POST_PROMPT}]
    assert dataset.build_grt_prompt(0) == messages
    predictions = pd.DataFrame({'index': range(634), 'prediction': ['reference words'] * 634,
                                'answer': ['forged reference'] * 634})
    path = tmp_path / 'results.xlsx'
    predictions.to_excel(path, index=False)
    result = dataset.evaluate(str(path))
    assert result['token_f1'] == result['exact_match'] == 1.0
    assert result['cer'] == result['wer'] == 0.0
    assert result['samples'] == 634
    assert not any('mos' in metric or 'gpt' in metric for metric in result)


@pytest.mark.parametrize('task,count', [
    (HIGH_MOTION, 3243), (PREVIEW, 1000), ('densevideo_highmotion', 1000),
])
def test_motion_split_independent_of_ambient_limit(release, monkeypatch, task, count):
    annotation, root = release(True)
    monkeypatch.setenv('DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES', '7')
    dataset = build_dataset(task, annotation_file=annotation, data_root=root)
    assert len(dataset) == count
    assert len(dataset.videos) == 2
    assert dataset.data.iloc[0]['video'] != dataset.data.iloc[1]['video']


def test_motion_exact_frame_prompt_and_metrics(release, monkeypatch, tmp_path):
    annotation, root = release(True)
    dataset = build_dataset(PREVIEW, annotation_file=annotation, data_root=root)
    monkeypatch.setattr(dataset, '_sample_frames', lambda row: [f'frame-{i}.png' for i in range(8)])
    with pytest.raises(ValueError, match='VIDEO_LLM'):
        dataset.build_prompt(0, video_llm=True)
    prompt = dataset.build_prompt(0, video_llm=False)
    assert [item['type'] for item in prompt] == ['image'] * 8 + ['text']
    assert 'exactly eight frames' in prompt[-1]['value']
    assert 'exactly seven commas' in prompt[-1]['value']
    assert 'We consider all 10 frames' not in prompt[-1]['value']
    target = ['topleft', 'middle', 'bottomright'] * 3 + ['middle']
    sampled = ','.join(target[index] for index in sample_indices(10, 8))
    path = tmp_path / 'motion.xlsx'
    pd.DataFrame({'index': range(1000), 'prediction': [sampled] * 1000}).to_excel(path, index=False)
    result = dataset.evaluate(str(path))
    assert result['grid_acc'] == result['grid_transition_acc'] == result['token_f1'] == 1
    assert result['grid_ade'] == result['grid_fde'] == 0


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'unknown'])
def test_evaluation_rejects_wrong_coverage(release, tmp_path, kind):
    annotation, root = release()
    dataset = DIVEBench(annotation_file=annotation, data_root=root)
    indices = list(range(634))
    if kind == 'duplicate':
        indices[-1] = 0
    elif kind == 'missing':
        indices.pop()
    else:
        indices[-1] = 9999
    path = tmp_path / 'bad.xlsx'
    pd.DataFrame({'index': indices, 'prediction': ['x'] * len(indices)}).to_excel(path, index=False)
    with pytest.raises(ValueError, match='complete selected split'):
        dataset.evaluate(str(path))


def test_checksum_and_missing_video_fail_closed(release):
    annotation, root = release()
    with annotation.open('ab') as output:
        output.write(b'changed')
    with pytest.raises(ValueError, match='SHA-256'):
        DIVEBench(annotation_file=annotation, data_root=root)
    annotation, root = release()
    (root / 'lecture.mp4').unlink()
    with pytest.raises(FileNotFoundError, match='video missing'):
        DIVEBench(annotation_file=annotation, data_root=root)


@pytest.mark.parametrize('high_motion', [False, True])
def test_hub_download_selects_only_unique_parquet(release, monkeypatch, high_motion):
    annotation, root = release(high_motion)
    import huggingface_hub
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(annotation)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    task = HIGH_MOTION if high_motion else EDUCATIONAL
    DIVEBench(dataset=task, data_root=root)
    assert len(calls) == 1
    assert calls[0]['filename'] == ('Egodex_traj.parquet' if high_motion else 'LPM_videos.parquet')
    assert calls[0]['revision'] == DIVEBench.SPECS[task]['revision']


def test_hub_error_is_not_hidden(release, monkeypatch):
    _, root = release()
    import huggingface_hub

    def denied(**kwargs):
        raise PermissionError('gated')

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', denied)
    with pytest.raises(RuntimeError, match='access terms'):
        DIVEBench(data_root=root)


@pytest.mark.parametrize('kwargs', [
    {'nframe': 0}, {'nframe': True}, {'nframe': 1.5}, {'fps': 1}, {'pack': True},
])
def test_invalid_frame_policy(kwargs):
    with pytest.raises(ValueError, match='positive integer'):
        DIVEBench(**kwargs)


def test_endpoint_indices():
    assert sample_indices(10, 8) == np.linspace(0, 9, 8, dtype=int).tolist()
    assert sample_indices(3, 8) == [0, 1, 2]
    assert sample_indices(1, 8) == [0]


def test_real_cpu_video_decoder_preserves_endpoint_frames(tmp_path):
    cv2 = pytest.importorskip('cv2')
    decord = pytest.importorskip('decord')
    from PIL import Image
    video = tmp_path / 'synthetic.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
    if not writer.isOpened():
        pytest.skip('The local OpenCV build has no MP4 encoder')
    try:
        for index in range(10):
            writer.write(np.full((32, 32, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()
    dataset = DIVEBench.__new__(DIVEBench)
    dataset.is_high_motion = True
    dataset.nframe = 8
    dataset.frame_root = str(tmp_path / 'frames')
    row = {'video': str(video), 'frame_count': 10}
    paths = dataset._sample_frames(row)
    indices = sample_indices(10, 8)
    expected = decord.VideoReader(str(video)).get_batch(indices).asnumpy()
    assert len(paths) == 8
    assert [Path(path).name for path in paths] == [f'endpoint-{index}.png' for index in indices]
    for path, frame in zip(paths, expected):
        with Image.open(path) as image:
            np.testing.assert_array_equal(np.asarray(image), frame)
    with pytest.raises(ValueError, match='frame count differs'):
        dataset._sample_frames(dict(row, frame_count=9))


def test_objective_metric_contract():
    assert _compute_exact_match(' A  b! ', 'a b!') == 1
    assert _compute_exact_match('b!', 'b') == 0
    assert _compute_cer('abc', 'a') == 2
    assert _compute_wer('a b c', 'a') == 2
    assert _compute_token_f1('a a b', 'a b b') == pytest.approx(2 / 3)
    assert _compute_token_f1('', '') == 1
    labels = parse_grid_sequence('{"trajectory": ["top-left", "middle", "r3c3"]}')
    assert labels == ['r1c1', 'r2c2', 'r3c3']
    assert _grid_sequence_metrics([], labels)['grid_fde'] == math.sqrt(2)
    assert _grid_sequence_metrics(labels + ['r1c1'], labels)['grid_acc'] == 1


def test_motion_path_does_not_fall_back_to_basename(tmp_path):
    dataset = DIVEBench.__new__(DIVEBench)
    dataset.is_high_motion = True
    (tmp_path / '0.mp4').touch()
    with pytest.raises(FileNotFoundError):
        dataset._resolve_video(tmp_path, 'egodex/action/0.mp4')
    with pytest.raises(ValueError, match='Unsafe'):
        dataset._resolve_video(tmp_path, '../0.mp4')
