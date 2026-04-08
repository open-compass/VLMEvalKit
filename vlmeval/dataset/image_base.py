import os
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vlmeval.smp import (LMUDataRoot, decode_base64_to_image_file, download_file, file_size,
                         istype, load, md5, mmqa_display, read_ok, toliststr)
from vlmeval.smp.file import INFER_FAIL_MSG, RUN_STATUS_NAME, _prediction_table, fetch_aux_files
from vlmeval.smp.status_report import is_number, to_number


def _choose_primary_metric_key(metrics: dict[str, Any]) -> str | None:
    if not metrics:
        return None

    scored = []
    for key in metrics:
        lower_key = str(key).lower()
        score = 0
        if lower_key == 'overall':
            score = 100
        elif lower_key.endswith('|overall'):
            score = 95
        elif 'overall' in lower_key:
            score = 90
        elif lower_key == 'acc' or lower_key.endswith('|acc'):
            score = 80
        elif 'acc' in lower_key:
            score = 70
        elif lower_key == 'score' or lower_key.endswith('|score'):
            score = 60
        elif 'score' in lower_key:
            score = 50

        if 'split=test' in lower_key:
            score += 4
        elif 'split=validation' in lower_key:
            score += 3
        elif 'split=val' in lower_key:
            score += 2
        elif 'split=dev' in lower_key:
            score += 1
        scored.append((score, str(key)))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _count_markers_in_obj(obj: Any, markers: tuple[str, ...]) -> int:
    if obj is None:
        return 0

    if isinstance(obj, pd.DataFrame):
        candidate_cols = [
            col for col in obj.columns if any(
                tag in str(col).lower()
                for tag in ('judge', 'log', 'res', 'rating', 'comment', 'response'))
        ]
        if not candidate_cols:
            candidate_cols = [col for col in obj.columns if obj[col].dtype == object]

        return sum(
            any(marker in str(value) for marker in markers)
            for col in candidate_cols
            for value in obj[col]
        )

    if isinstance(obj, dict):
        return sum(_count_markers_in_obj(value, markers) for value in obj.values())

    if isinstance(obj, (list, tuple, set)):
        return sum(_count_markers_in_obj(value, markers) for value in obj)

    return int(any(marker in str(obj) for marker in markers))


def _score_judge_file_candidate(candidate: Path, judge_model: str | None = None) -> tuple[int, str]:
    name = candidate.name.lower()
    stem = candidate.stem.lower()
    score = 0

    if judge_model and judge_model.lower() in name:
        score += 100

    if 'judge' in stem:
        score += 40
    if 'result' in stem:
        score += 30
    if any(tag in stem for tag in ('response', 'rating', 'review')):
        score += 20

    if candidate.suffix.lower() == '.pkl':
        score += 12
    elif candidate.suffix.lower() in {'.xlsx', '.xls', '.json', '.jsonl', '.tsv'}:
        score += 8
    elif candidate.suffix.lower() == '.csv':
        score += 2

    if any(tag in stem for tag in ('_score', '_acc', 'metrics')):
        score -= 100

    return score, candidate.name


def img_root_map(dataset):
    if 'MM_NIAH' in dataset:
        return 'MMNIAH'
    if 'CRPE' in dataset:
        return 'CRPE'
    if 'OCRVQA' in dataset:
        return 'OCRVQA'
    if 'COCO_VAL' == dataset:
        return 'COCO'
    if "QSpatial" in dataset:
        return "QSpatial"

    mmbench_root_map = {
        'MMBench_DEV_EN': 'MMBench', 'MMBench_TEST_EN': 'MMBench',
        'MMBench_DEV_CN': 'MMBench', 'MMBench_TEST_CN': 'MMBench',
        'MMBench_DEV_KO': 'MMBench',
        'MMBench': 'MMBench', 'MMBench_CN': 'MMBench',
        'MMBench_DEV_EN_V11': 'MMBench_V11', 'MMBench_TEST_EN_V11': 'MMBench_V11',
        'MMBench_DEV_CN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_V11',
        'MMBench_V11': 'MMBench', 'MMBench_CN_V11': 'MMBench',
    }
    if dataset in mmbench_root_map:
        return mmbench_root_map[dataset]
    return dataset


class ImageBaseDataset(metaclass=ABCMeta):

    MODALITY = 'IMAGE'
    DATASET_URL = {}
    DATASET_MD5 = {}
    DEFAULT_JUDGE: str | list = 'gpt-4o-mini'

    INFER_FAIL_MARKERS = (INFER_FAIL_MSG, )
    JUDGE_FAIL_MARKERS = (INFER_FAIL_MSG, )

    def __init__(self, dataset='MMBench', skip_noimg=True):
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, 'images', img_root_map(dataset))

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]

        data['index'] = [str(x) for x in data['index']]

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    @classmethod
    def get_judge_file(cls, eval_file: str | Path, judge_model: str | None = None) -> Path | None:
        eval_path = Path(eval_file)
        aux_files = fetch_aux_files(str(eval_path))
        if not aux_files:
            return None

        prediction_path = eval_path.resolve() if eval_path.exists() else None
        candidates = []
        for aux_file in aux_files:
            aux_path = Path(aux_file)
            if not aux_path.exists():
                continue
            if prediction_path is not None and aux_path.resolve() == prediction_path:
                continue
            if aux_path.name == RUN_STATUS_NAME or aux_path.suffix == '.lock':
                continue
            if aux_path.name.endswith(('_checkpoint.pkl', '_PREV.pkl', '_structs.pkl')):
                continue
            candidates.append(aux_path)

        if not candidates:
            return None

        candidates.sort(key=lambda path: _score_judge_file_candidate(path, judge_model), reverse=True)
        return candidates[0]

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name_legacy = url.split('/')[-1]
        file_name = f"{self.dataset_name}.tsv"
        data_path_legacy = osp.join(data_root, file_name_legacy)
        data_path = osp.join(data_root, file_name)

        self.data_path = data_path
        if osp.exists(data_path):
            if file_md5 is None or md5(data_path) == file_md5:
                pass
            else:
                warnings.warn(f'The tsv file is in {data_root}, but the md5 does not match, will re-download')
                download_file(url, data_path)
                update_flag = True
        else:
            if osp.exists(data_path_legacy) and (file_md5 is None or md5(data_path_legacy) == file_md5):
                warnings.warn(
                    'Due to a modification in #1055, the local target file name has changed. '
                    f'We detected the tsv file with legacy name {data_path_legacy} exists and will do the rename. '
                )
                import shutil
                shutil.move(data_path_legacy, data_path)
            else:
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
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                if 'image_path' in line:
                    image_path = line['image_path']
                else:
                    index = line['index']
                    image_path = [f'{index}_{i}.png' for i in range(len(line['image']))]
                for img, im_name in zip(line['image'], image_path):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)

            elif isinstance(line['image'], str) and 'image_path' in line:
                assert isinstance(line['image_path'], str)
                tgt_path = osp.join(self.img_root, line['image_path'])
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.png")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])
            read_ok_flag = [read_ok(x) for x in tgt_path]
            # Might be the Relative Path
            if not all(read_ok_flag):
                tgt_path_abs = [osp.join(self.img_root, x) for x in tgt_path]
                read_ok_flag = [read_ok(x) for x in tgt_path_abs]
                assert read_ok_flag, f"Field `image` is missing and we could not find {tgt_path} both as absolute or relative paths. "  # noqa
                tgt_path = tgt_path_abs

        return tgt_path

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
        url = self.DATASET_URL.get(dataset, None)
        if url is None or url == '':
            url = dataset + '.tsv'
        file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        return self.prepare_tsv(url, file_md5)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @classmethod
    def report_infer_err(cls, prediction_file: str | Path | None) -> dict[str, int]:
        if prediction_file is None or not Path(prediction_file).exists():
            return dict(failed=0, total=0)

        frame = _prediction_table(str(prediction_file))
        if frame is None:
            return dict(failed=0, total=0)

        predictions = frame['prediction'] if 'prediction' in frame else []
        total = len(predictions)
        failed = sum(
            any(marker in str(prediction) for marker in cls.INFER_FAIL_MARKERS)
            for prediction in predictions
        )
        return dict(failed=failed, total=total)

    @classmethod
    def report_judge_err(
        cls,
        prediction_file: str | Path | None,
        *,
        total_samples: int | None,
        judge_model: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, int | None]:
        if total_samples is None or total_samples <= 0:
            return dict(failed=0, total=total_samples)

        if prediction_file is None or not Path(prediction_file).exists():
            failed = total_samples if error_message else 0
            return dict(failed=failed, total=total_samples)

        judge_file = cls.get_judge_file(prediction_file, judge_model=judge_model)
        if judge_file is None:
            failed = total_samples if error_message else 0
            return dict(failed=failed, total=total_samples)

        try:
            data = load(str(judge_file))
        except Exception:
            data = None

        failed = min(_count_markers_in_obj(data, cls.JUDGE_FAIL_MARKERS), total_samples)
        if error_message and failed == 0:
            failed = total_samples
        return dict(failed=failed, total=total_samples)

    @classmethod
    def report_primary_metric(cls, metrics: dict[str, Any] | None) -> dict[str, float | int]:
        if not isinstance(metrics, dict) or not metrics:
            return {}

        matched_keys = []
        fallback = _choose_primary_metric_key(metrics)
        if fallback is not None:
            matched_keys = [fallback]

        primary_metrics = {}
        for key in matched_keys:
            value = metrics.get(key)
            if is_number(value):
                primary_metrics[str(key)] = to_number(value)
        return primary_metrics
