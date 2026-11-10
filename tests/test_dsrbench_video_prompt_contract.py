import ast
import inspect
import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_function(path, function_name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(
        item for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == function_name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function_name]


def test_shared_prompt_dispatch_forwards_video_mode_only_when_supported():
    build_dataset_prompt = _load_function(
        REPO_ROOT / 'vlmeval' / 'inference.py',
        '_build_dataset_prompt',
        {'inspect': inspect},
    )

    class VideoPromptDataset:
        def build_prompt(self, item, video_llm=False):
            return item, video_llm

    class LegacyPromptDataset:
        def build_prompt(self, item):
            return item

    class Model:
        VIDEO_LLM = True

    assert build_dataset_prompt(VideoPromptDataset(), 'sample', Model()) == ('sample', True)
    assert build_dataset_prompt(LegacyPromptDataset(), 'sample', Model()) == 'sample'


def _load_dsrbench_prompt():
    source = REPO_ROOT / 'vlmeval' / 'dataset' / 'dsrbench.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    dataset_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'DSRBench'
    )
    method = next(
        node for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == 'build_prompt'
    )
    namespace = {'ast': ast, 'json': json, 'os': os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['build_prompt']


class _PromptDataset:
    FRAMES_TMPL_NOSUB = 'video frames'
    build_prompt = _load_dsrbench_prompt()

    def __init__(self, nframe=0, fps=-1):
        self.data_root = '/videos'
        self.nframe = nframe
        self.fps = fps

    def save_video_frames(self, video_path, video_llm=False):
        if self.nframe <= 0 and self.fps <= 0:
            raise AssertionError('frame extraction must not start without a sampling setting')
        return ['/frames/1.jpg', '/frames/2.jpg'], [], {}


def test_dsrbench_native_video_mode_uses_video_input_without_sampling():
    message = _PromptDataset().build_prompt(
        {'question': 'What happens?', 'options': ['A', 'B'], 'video': 'clip.mp4'},
        video_llm=True,
    )

    assert {'type': 'video', 'value': os.path.join('/videos', 'clip.mp4')} in message
    assert not any(part['type'] == 'image' for part in message)


def test_dsrbench_requires_sampling_for_frame_based_models():
    with pytest.raises(ValueError, match='DSRBench requires nframe > 0 or fps > 0'):
        _PromptDataset().build_prompt(
            {'question': 'What happens?', 'options': ['A', 'B'], 'video': 'clip.mp4'}
        )


def test_dsrbench_uses_configured_frames_for_frame_based_models():
    message = _PromptDataset(nframe=2).build_prompt(
        {'question': 'What happens?', 'options': ['A', 'B'], 'video': 'clip.mp4'}
    )

    assert [part['value'] for part in message if part['type'] == 'image'] == [
        '/frames/1.jpg', '/frames/2.jpg'
    ]
