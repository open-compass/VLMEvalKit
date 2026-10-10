import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image


def load_v2pbench():
    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']
    smp.__getattr__ = lambda name: mock.Mock()
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.__getattr__ = lambda name: mock.Mock()
    log = types.ModuleType('vlmeval.smp.log')
    log.get_logger = lambda _name: mock.Mock()
    status = types.ModuleType('vlmeval.smp.status_report')
    status.is_number = mock.Mock()
    status.to_number = mock.Mock()
    image_base = types.ModuleType('vlmeval.dataset.image_base')
    image_base.__getattr__ = lambda name: mock.Mock()
    hub = types.ModuleType('huggingface_hub')
    hub.snapshot_download = mock.Mock()
    lock = types.ModuleType('portalocker')
    lock.Lock = lambda *args, **kwargs: nullcontext()
    modules = {'vlmeval': package, 'vlmeval.dataset': dataset, 'vlmeval.smp': smp,
               'vlmeval.smp.file': smp_file, 'vlmeval.smp.log': log, 'vlmeval.smp.status_report': status,
               'vlmeval.dataset.image_base': image_base, 'huggingface_hub': hub, 'portalocker': lock}
    with mock.patch.dict(sys.modules, modules):
        for name in ['video_base', 'v2pbench']:
            spec = importlib.util.spec_from_file_location(
                f'vlmeval.dataset.{name}', f'vlmeval/dataset/{name}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return module


class TestV2PBenchFramePath(unittest.TestCase):

    def test_frame_and_video_prompts_read_the_same_dataset_video(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for suffix in ['mp4', 'avi']:
                with self.subTest(suffix=suffix):
                    paths.append(self.check_prompt(suffix, directory))
            self.assertTrue(set(paths[0]).isdisjoint(paths[1]))

    def check_prompt(self, suffix, directory):
        module = load_v2pbench()
        with nullcontext(directory):
            root = Path(directory)
            video = root / f'videos/EgoSchema/sample.{suffix}'
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()
            dataset = module.V2PBench.__new__(module.V2PBench)
            dataset.data_root = dataset.video_path = directory
            dataset.frame_root = str(root / 'cached_frames')
            dataset.frame_tmpl = 'frame-{}-of-{}.jpg'
            dataset.nframe = 2
            dataset.fps = -1
            line = pd.Series({'video_path': f'EgoSchema/sample.{suffix}', 'frame_path': 'region.png',
                              'question': 'What is happening?'})
            decord = types.ModuleType('decord')
            reader = mock.MagicMock()
            reader.__len__.return_value = 6
            reader.__getitem__.return_value.asnumpy.return_value = np.zeros((2, 2, 3), dtype=np.uint8)

            def open_video(path):
                self.assertEqual(Path(path), video)
                self.assertTrue(Path(path).is_file())
                return reader

            decord.VideoReader = mock.Mock(side_effect=open_video)
            with mock.patch.dict(sys.modules, {'decord': decord}):
                frame_prompt = dataset.build_prompt(line, video_llm=False)
            video_prompt = dataset.build_prompt(line, video_llm=True)
            self.assertEqual(video_prompt[1], {'type': 'video', 'value': str(video)})
            decord.VideoReader.assert_called_once_with(str(video))
            frame_paths = [item['value'] for item in frame_prompt if item['type'] == 'image'][:-1]
            self.assertEqual(len(frame_paths), 2)
            self.assertTrue(all(Path(path).is_file() for path in frame_paths))
            for path in frame_paths:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (2, 2))
            return frame_paths

    def test_complete_cache_uses_videos_subdirectory(self):
        module = load_v2pbench()
        dataset = module.V2PBench.__new__(module.V2PBench)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / 'V2P-Bench.tsv'
            pd.DataFrame({'video_path': ['EgoSchema/sample.mp4']}).to_csv(metadata, sep='\t', index=False)
            video = root / 'videos/EgoSchema/sample.mp4'
            video.parent.mkdir(parents=True)
            video.touch()
            with mock.patch.object(module, 'get_cache_path', return_value=directory), \
                    mock.patch.object(module, 'load', side_effect=lambda path: pd.read_csv(path, sep='\t')), \
                    mock.patch.object(module, 'modelscope_flag_set', return_value=False), \
                    mock.patch.object(module, 'snapshot_download', side_effect=AssertionError) as downloader:
                result = dataset.prepare_dataset()
            downloader.assert_not_called()
            self.assertEqual(result, {'data_file': str(metadata), 'root': directory})


if __name__ == '__main__':
    unittest.main()
