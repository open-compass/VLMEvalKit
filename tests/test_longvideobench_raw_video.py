"""Native local-video routing checks; FFmpeg supplies the reader protocol seam."""
import ast
import inspect
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import sys
import types
import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import validators
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), 'exec'), namespace)


@pytest.mark.parametrize('split_frame', [False, True])
@pytest.mark.parametrize('video_llm', [False, True])
def test_prompt_checks_frame_sampling(split_frame, video_llm, tmp_path):
    namespace = dict(VideoBaseDataset=object, osp=os.path)
    definitions('vlmeval/dataset/longvideobench.py', {'LongVideoBench'}, namespace)
    dataset = namespace['LongVideoBench'].__new__(namespace['LongVideoBench'])
    dataset.data_root = str(tmp_path)
    dataset.split_frame = split_frame
    dataset.use_subtitle = False
    dataset.save_video_frames = Mock(return_value=(['frame.jpg'], [0], {'fps': 1}))
    line = dict(video_path='sample.mp4', question='Which color?', candidates="['red', 'blue']")

    if not split_frame and not video_llm:
        with pytest.raises(ValueError, match='fps and nframe cannot both be <= 0'):
            dataset.build_prompt(line, video_llm=video_llm)
        dataset.save_video_frames.assert_not_called()
        assert line['question'] == 'Which color?'
        return

    message = dataset.build_prompt(line, video_llm=video_llm)
    assert [item['type'] for item in message] == ['text', 'video' if video_llm else 'image', 'text']
    if video_llm:
        dataset.save_video_frames.assert_not_called()
    else:
        dataset.save_video_frames.assert_called_once_with('sample.mp4', False)


@pytest.fixture
def local_video(tmp_path, monkeypatch):
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('native video checks require FFmpeg and FFprobe')
    video = tmp_path / 'sample.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=red:s=64x64:r=1:d=4', '-c:v', 'libx264', '-threads', '2',
                    '-pix_fmt', 'yuv420p', '-y', str(video)], check=True, timeout=10)
    counts = dict(readers=0, frames=0)

    class Reader:
        def __init__(self, path):
            counts['readers'] += 1
            self.path = path
            self.metadata = json.loads(subprocess.check_output(
                [ffprobe, '-v', 'error', '-show_entries',
                 'stream=nb_frames,avg_frame_rate,width,height', '-of', 'json', path],
                timeout=10))['streams'][0]

        def __len__(self):
            return int(self.metadata['nb_frames'])

        def get_avg_fps(self):
            numerator, denominator = map(int, self.metadata['avg_frame_rate'].split('/'))
            return numerator / denominator

        def __getitem__(self, index):
            counts['frames'] += 1
            raw = subprocess.check_output(
                [ffmpeg, '-v', 'error', '-i', self.path, '-vf', f'select=eq(n\\,{index})',
                 '-frames:v', '1', '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 'pipe:1'], timeout=10)
            return np.frombuffer(raw, dtype=np.uint8).reshape(
                self.metadata['height'], self.metadata['width'], 3)

    decord = types.ModuleType('decord')
    decord.VideoReader = Reader
    monkeypatch.setitem(sys.modules, 'decord', decord)
    namespace = dict(os=os, osp=os.path, np=np, pd=pd, Image=Image, inspect=inspect,
                     warnings=warnings, ABCMeta=ABCMeta, abstractmethod=abstractmethod,
                     Any=Any, Path=Path, logger=logging.getLogger('local-video'),
                     INFER_FAIL_MSG='Failed to obtain answer', mimetypes=mimetypes, validators=validators)
    definitions('vlmeval/smp/file.py', {'load', 'parse_file'}, namespace)
    namespace['LMUDataRoot'] = lambda: str(tmp_path)
    definitions('vlmeval/dataset/video_base.py', {'VideoBaseDataset'}, namespace)
    definitions('vlmeval/dataset/longvideobench.py', {'LongVideoBench'}, namespace)
    definitions('vlmeval/vlm/base.py', {'BaseModel'}, namespace)
    pd.DataFrame([dict(index=0, video='sample', video_path='sample.mp4',
                       question='Which color?', candidates="['red', 'blue']")]).to_csv(
                           tmp_path / 'local.tsv', sep='\t', index=False)

    class LocalVideoBench(namespace['LongVideoBench']):
        def prepare_dataset(self, dataset_name):
            return dict(root=str(tmp_path), data_file=str(tmp_path / 'local.tsv'))

    return LocalVideoBench, namespace['BaseModel'], Reader, counts, video


@pytest.mark.parametrize('nframe', [0, 3])
def test_raw_video_bypasses_frame_reader(local_video, nframe):
    dataset_class, model_class, reader_class, counts, video = local_video
    dataset = dataset_class(dataset='local', nframe=nframe, fps=-1)
    if nframe:
        reader = reader_class(str(video))
        for index, path in zip((1, 2, 3), dataset.frame_paths('sample')):
            Image.fromarray(reader[index]).save(path)
        counts.update(readers=0, frames=0)
    message = dataset.build_prompt(0, video_llm=True)
    model = model_class()
    model.VIDEO_LLM = True
    # Run the shipping model's real media validation and prompt consumer.
    prompt, path = model.message_to_promptvideo(model.preproc_content(message))
    assert path == str(video)
    assert 'A. red' in prompt and 'B. blue' in prompt
    assert counts == dict(readers=0, frames=0)


def test_sampled_frame_cache_still_uses_video_metadata(local_video):
    dataset_class, _, reader_class, counts, video = local_video
    dataset = dataset_class(dataset='local', nframe=3, fps=-1)
    reader = reader_class(str(video))
    assert len(reader) == 4 and reader.get_avg_fps() == 1
    paths = dataset.frame_paths('sample')
    # Populate the ordinary frame cache with physically decoded RGB frames.
    for index, path in zip((1, 2, 3), paths):
        Image.fromarray(reader[index]).save(path)
    counts.update(readers=0, frames=0)
    message = dataset.build_prompt(0, video_llm=False)
    assert [item['value'] for item in message if item['type'] == 'image'] == paths
    assert counts == dict(readers=1, frames=0)
    for path in paths:
        pixel = np.array(Image.open(path))[32, 32]
        assert pixel[0] > 240 and pixel[1] < 10 and pixel[2] < 10
