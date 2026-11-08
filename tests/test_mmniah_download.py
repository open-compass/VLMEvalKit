import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


def load_image_vqa():
    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    base = types.ModuleType('vlmeval.dataset.image_base')
    base.ImageBaseDataset = type('ImageBaseDataset', (), {})
    smp = types.ModuleType('vlmeval.smp')
    smp.__getattr__ = lambda name: mock.Mock()
    utils = types.ModuleType('vlmeval.utils')
    utils.track_progress_rich = mock.Mock()
    dataset_utils = types.ModuleType('vlmeval.dataset.utils')
    dataset_utils.__path__ = ['vlmeval/dataset/utils']
    dataset_utils.DEBUG_MESSAGE = ''
    dataset_utils.build_judge = mock.Mock()
    cache = types.ModuleType('vlmeval.dataset.utils.judge_cache')
    cache.__getattr__ = lambda name: mock.Mock()
    vqa = types.ModuleType('vlmeval.dataset.utils.vqa_eval')
    vqa.istype = mock.Mock()
    modules = {'vlmeval': package, 'vlmeval.dataset': dataset, 'vlmeval.dataset.image_base': base,
               'vlmeval.smp': smp, 'vlmeval.utils': utils, 'vlmeval.dataset.utils': dataset_utils,
               'vlmeval.dataset.utils.judge_cache': cache, 'vlmeval.dataset.utils.vqa_eval': vqa}
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('vlmeval.dataset.image_vqa', 'vlmeval/dataset/image_vqa.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestMMNIAHShardDownload(unittest.TestCase):

    def check_download(self, resume, interrupt=False):
        module = load_image_vqa()
        dataset = module.MMNIAH.__new__(module.MMNIAH)
        urls = module.MMNIAH.DATASET_URL['MM_NIAH_TEST']
        content = b'index\tquestion\tanswer\n0\tWhat?\tblue\n1\tWhich?\tred\n'
        size = len(content) // len(urls)
        pieces = [content[i * size:(i + 1) * size] for i in range(len(urls) - 1)]
        pieces.append(content[(len(urls) - 1) * size:])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if resume:
                (root / 'part-aa').write_bytes(pieces[0])
            (root / 'part-unrelated').write_bytes(b'This is not an MM-NIAH shard')

            def download(url, path):
                Path(path).write_bytes(pieces[urls.index(url)])

            with mock.patch.object(module, 'LMUDataRoot', return_value=directory), \
                    mock.patch.object(module, 'download_file', side_effect=download) as downloader, \
                    mock.patch.object(module, 'file_size', return_value=0), \
                    mock.patch.object(module, 'load', side_effect=lambda path: pd.read_csv(path, sep='\t')):
                if interrupt:
                    def interrupted_download(url, path):
                        if url == urls[1]:
                            Path(path).write_bytes(b'Incomplete shard')
                            raise ConnectionError('Download interrupted')
                        download(url, path)

                    downloader.side_effect = interrupted_download
                    with self.assertRaises(ConnectionError):
                        dataset.prepare_tsv(urls)
                    self.assertEqual((root / 'part-aa').read_bytes(), pieces[0])
                    self.assertFalse((root / 'part-ab').exists())
                    self.assertFalse(list(root.glob('*.tmp')))
                    downloader.reset_mock()
                    downloader.side_effect = download
                result = dataset.prepare_tsv(urls)
            self.assertEqual((root / 'MM_NIAH_TEST.tsv').read_bytes(), content)
            self.assertEqual(result['answer'].tolist(), ['blue', 'red'])
            expected_urls = urls[1:] if resume or interrupt else urls
            self.assertEqual([call.args[0] for call in downloader.call_args_list], expected_urls)
            for call in downloader.call_args_list:
                self.assertTrue(Path(call.args[1]).name.startswith(call.args[0].rsplit('/', 1)[1] + '.'))
                self.assertFalse(Path(call.args[1]).exists())

    def test_fresh_download_assembles_all_five_shards(self):
        self.check_download(resume=False)

    def test_partial_write_is_removed_and_downloaded_again(self):
        self.check_download(resume=False, interrupt=True)

    def test_existing_shard_files_are_reused(self):
        self.check_download(resume=True)


if __name__ == '__main__':
    unittest.main()
