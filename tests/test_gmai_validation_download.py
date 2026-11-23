import ast
import hashlib
import os
import warnings
from pathlib import Path

import pandas as pd
import pytest


def load_dataset_class(namespace):
    base_tree = ast.parse(Path('vlmeval/dataset/image_base.py').read_text())
    base = next(n for n in base_tree.body
                if isinstance(n, ast.ClassDef) and n.name == 'ImageBaseDataset')
    base.name = 'ImageMCQDataset'
    base.bases = []
    base.keywords = []
    base.body = [n for n in base.body if isinstance(n, ast.FunctionDef)
                 and n.name in ['load_data', 'prepare_tsv']]
    tree = ast.parse(Path('vlmeval/dataset/image_mcq.py').read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == 'GMAIMMBenchDataset')
    cls.body = [n for n in cls.body if isinstance(n, ast.Assign)
                or isinstance(n, ast.FunctionDef) and n.name == 'load_data']
    exec(compile(ast.Module(body=[base, cls], type_ignores=[]), 'datasets.py', 'exec'), namespace)
    return namespace['GMAIMMBenchDataset']


@pytest.mark.parametrize('cache_state', ['missing', 'stale', 'valid'])
def test_validation_data_is_prepared_with_checksum_before_loading(tmp_path, cache_state):
    filename = tmp_path / 'GMAI-MMBench_VAL.tsv'
    fresh = 'index\tquestion\tanswer\n0\tWhich?\tA\n'
    if cache_state != 'missing':
        stale = 'index\tquestion\tanswer\n0\tOld?\tB\n'
        filename.write_text(fresh if cache_state == 'valid' else stale)
    downloads = []

    def download(url, target):
        downloads.append(url)
        Path(target).write_text(fresh)

    namespace = {
        'os': os, 'osp': os.path, 'warnings': warnings,
        'LMUDataRoot': lambda: str(tmp_path),
        'file_size': lambda filename, unit: 0,
        'md5': lambda filename: hashlib.md5(Path(filename).read_bytes()).hexdigest(),
        'download_file': download,
        'load': lambda filename: pd.read_csv(filename, sep='\t'),
    }
    cls = load_dataset_class(namespace)
    dataset = cls()
    dataset.dataset_name = 'GMAI-MMBench_VAL'
    dataset.DATASET_MD5 = {dataset.dataset_name: hashlib.md5(fresh.encode()).hexdigest()}
    result = dataset.load_data(dataset.dataset_name)
    assert result['question'].tolist() == ['Which?']
    assert result['answer'].tolist() == ['A']
    expected = [] if cache_state == 'valid' else [dataset.DATASET_URL[dataset.dataset_name]]
    assert downloads == expected
