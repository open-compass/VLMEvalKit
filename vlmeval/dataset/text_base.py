import os
import os.path as osp
import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, download_file, file_size, istype, load, md5, mmqa_display
from .utils.repeat import default_pass_k_detail_paths, merge_numeric_metrics


class TextBaseDataset:
    MODALITY = 'TEXT'
    DATASET_URL = {}
    DATASET_MD5 = {}
    DEFAULT_JUDGE_MODEL: str | None = None
    # Independent runs per question (avg@n / pass@n); ``--n-runs`` overrides it, see utils/repeat.py.
    DEFAULT_N_RUNS: int = 1
    # Name of the 0/1 (bool) column in the per-sample detail file written by ``evaluate``.
    # ``None`` -> only the merged metrics are reported, pass@n / pass^n are skipped.
    PASS_K_HIT_COL: str | None = None
    # Suffix of that detail file, as passed to ``get_intermediate_file_path``.
    PASS_K_DETAIL_SUFFIX = '_results'
    # Format of that detail file. The dataset's ``evaluate`` writes it and decides that format
    # itself -- most call ``get_intermediate_file_path(eval_file, '_results')`` without a format,
    # which resolves through ``PRED_FORMAT`` (default xlsx), not ``EVAL_FORMAT``. So leave this
    # ``None`` to read whichever format actually exists; set it only when the dataset hardcodes a
    # format that the probe could get wrong, and keep it in sync with what ``evaluate`` writes.
    PASS_K_DETAIL_FORMAT: str | None = None
    # Formats tried in order when ``PASS_K_DETAIL_FORMAT`` is ``None``.
    PASS_K_DETAIL_PROBE = ('csv', 'json', 'xlsx')

    def __init__(self, dataset='MMBench', **kwargs):
        self.dataset_name = dataset

        data = self.load_data(dataset)

        data['index'] = [str(x) for x in data['index']]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def get_pass_k_detail_paths(self, sub_file):
        """Where ``evaluate`` wrote its per-sample detail for the run on ``sub_file``; override when
        the file is named differently (e.g. after the judge)."""
        return default_pass_k_detail_paths(self, sub_file)

    def get_default_n_runs(self):
        return getattr(self, 'DEFAULT_N_RUNS', 1)

    def merge_metrics(self, results):
        """Merge what ``evaluate`` returned for each of several runs; override for metrics that
        cannot be averaged."""
        return merge_numeric_metrics(results)

    def get_default_judge_model(self, judge_kwargs=None):
        default_model = getattr(self, 'DEFAULT_JUDGE_MODEL', None)
        if default_model is not None:
            return default_model

        dataset_type = getattr(self, 'TYPE', None)
        if dataset_type in ('MCQ', 'Y/N', 'MCQ_MMMU_Pro'):
            return 'gpt-4o-mini'
        return None

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
            update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def dump_image(self, line):
        return []

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        assert isinstance(line, pd.Series) or isinstance(line, dict)
        mmqa_display(line)

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        url = self.DATASET_URL[dataset]
        file_md5 = self.DATASET_MD5[dataset]
        return self.prepare_tsv(url, file_md5)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']

        msgs = []
        msgs.append(dict(type='text', value=question))
        return msgs

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass
