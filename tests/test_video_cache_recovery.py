import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


def load_dataset_module(name):
    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    utils = types.ModuleType('vlmeval.utils')
    utils.track_progress_rich = mock.Mock()
    dataset_utils = types.ModuleType('vlmeval.dataset.utils')
    dataset_utils.__path__ = ['vlmeval/dataset/utils']
    dataset_utils.__getattr__ = lambda name: mock.Mock()
    smp = types.ModuleType('vlmeval.smp')
    smp.__getattr__ = lambda name: mock.Mock()
    hub = types.ModuleType('huggingface_hub')
    hub.snapshot_download = mock.Mock()
    base = types.ModuleType('vlmeval.dataset.video_base')
    base.VideoBaseDataset = type('VideoBaseDataset', (), {})
    modules = {'vlmeval': package, 'vlmeval.dataset': dataset, 'vlmeval.utils': utils,
               'vlmeval.dataset.utils': dataset_utils, 'vlmeval.smp': smp,
               'huggingface_hub': hub, 'vlmeval.dataset.video_base': base}
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(f'vlmeval.dataset.{name}', f'vlmeval/dataset/{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


CASES = [('v2pbench', 'V2PBench', 'V2P-Bench'), ('vcrbench', 'VCRBench', 'VCR-Bench'),
         ('mmbench_video', 'MMBenchVideo', 'MMBench-Video'), ('vdc', 'VDC', 'VDC'),
         ('moviechat1k', 'MovieChat1k', 'MovieChat1k'), ('video_mmlu', 'Video_MMLU_CAP', 'Video_MMLU_CAP'),
         ('video_mmlu', 'Video_MMLU_QA', 'Video_MMLU_QA')]


class DownloadRequested(Exception):
    pass


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


class TestVideoCacheRecovery(unittest.TestCase):

    def test_missing_metadata_allows_snapshot_download_to_resume(self):
        for module_name, class_name, dataset_name in CASES:
            with self.subTest(dataset=dataset_name), tempfile.TemporaryDirectory() as directory:
                module = load_dataset_module(module_name)
                cls = getattr(module, class_name)
                dataset = cls.__new__(cls)
                with mock.patch.object(module, 'get_cache_path', return_value=directory), \
                        mock.patch.object(module, 'load', side_effect=lambda path: pd.read_csv(path, sep='\t')), \
                        mock.patch.object(module, 'md5', side_effect=file_md5, create=True), \
                        mock.patch.object(module, 'modelscope_flag_set', return_value=False, create=True), \
                        mock.patch.object(module, 'snapshot_download', side_effect=DownloadRequested) as downloader:
                    with self.assertRaises(DownloadRequested):
                        dataset.prepare_dataset(dataset_name=dataset_name)
                downloader.assert_called_once()

    def test_complete_metadata_and_videos_reuse_the_cache(self):
        for module_name, class_name, dataset_name in CASES[1:]:
            with self.subTest(dataset=dataset_name), tempfile.TemporaryDirectory() as directory:
                module = load_dataset_module(module_name)
                cls = getattr(module, class_name)
                dataset = cls.__new__(cls)
                root = Path(directory)
                table = pd.DataFrame({'video': ['sample.mp4'], 'video_path': ['sample.mp4']})
                metadata = root / (dataset_name + '.tsv')
                table.to_csv(metadata, sep='\t', index=False)
                dataset.MD5 = file_md5(metadata)
                for path in ['sample.mp4', 'videos/sample.mp4', 'youtube_videos/sample.mp4']:
                    video = root / path
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.touch()
                with mock.patch.object(module, 'get_cache_path', return_value=directory), \
                        mock.patch.object(module, 'load', side_effect=lambda path: pd.read_csv(path, sep='\t')), \
                        mock.patch.object(module, 'md5', side_effect=file_md5, create=True), \
                        mock.patch.object(module, 'modelscope_flag_set', return_value=False, create=True), \
                        mock.patch.object(module, 'snapshot_download', side_effect=AssertionError) \
                        as downloader:
                    result = dataset.prepare_dataset(dataset_name=dataset_name)
                downloader.assert_not_called()
                self.assertEqual(result['data_file'], str(metadata))


if __name__ == '__main__':
    unittest.main()
