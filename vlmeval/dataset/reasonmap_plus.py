# ReasonMap-Plus is an extension of ReasonMap with denser supervision (counting and true/false
# questions on the same transit maps). Reference papers:
#   1. ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps (CVPR 2026):
#      https://arxiv.org/abs/2505.18675
#   2. RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via
#      Multi-Stage Reinforcement Learning (ICLR 2026): https://arxiv.org/abs/2510.02240
# Data: https://huggingface.co/datasets/FSCCS/ReasonMap-Plus
#
# If any problem occurs, please open an issue on GitHub
# (https://github.com/fscdc/RewardMap or https://github.com/fscdc/ReasonMap).

import os.path as osp

import pandas as pd

from vlmeval.smp import d2df, dump, get_intermediate_file_path, get_logger, load
from .image_base import ImageBaseDataset
from .reasonmap import (atomic_dump, ensure_map_images, hf_dataset_file, map_image_name,
                        reasonmap_data_dir, reasonmap_maps_dir, warn_truncated_predictions)
from .utils.reasonmap import extract_plus_answer, plus_groups, plus_is_correct, plus_weight

logger = get_logger(__name__)

HF_REPO = 'FSCCS/ReasonMap-Plus'
HF_TEST_FILE = 'data/test.parquet'


class ReasonMap_Plus(ImageBaseDataset):
    """ReasonMap-Plus test set: 1448 questions (Counting1/2/3, TorF1/2) on the 11 ReasonMap test maps."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    # The questions carry their own answer instructions (e.g. "put your answer in \boxed{}").
    force_use_dataset_prompt = True
    DATASET_URL = {'ReasonMap-Plus': ''}  # built from the official HF dataset in ``load_data``
    DATASET_MD5 = {'ReasonMap-Plus': None}
    PASS_K_HIT_COL = 'hit'

    _COLUMNS = ('index', 'question', 'answer', 'type', 'image_path', 'country', 'city', 'station_1',
                'station_2', 'difficulty_city', 'city_line_count', 'city_transfer_count')

    @classmethod
    def supported_datasets(cls):
        return ['ReasonMap-Plus']

    # ------------------------------------------------------------------ data

    def load_data(self, dataset):
        self.img_root = reasonmap_maps_dir()  # the maps are shared with ReasonMap
        data_file = osp.join(reasonmap_data_dir(), 'ReasonMap-Plus.tsv')
        data = load(data_file) if osp.exists(data_file) else None
        if data is None or not set(self._COLUMNS) <= set(data.columns):
            data = self._build_from_hf()
            atomic_dump(data, data_file)
        ensure_map_images(set(zip(data['country'], data['city'])), self.img_root)
        return data

    @classmethod
    def _build_from_hf(cls):
        import pyarrow.parquet as pq

        logger.info(f'Converting {HF_REPO} ({HF_TEST_FILE}) for VLMEvalKit')
        records = pq.read_table(hf_dataset_file(HF_REPO, HF_TEST_FILE)).to_pylist()
        rows = []
        for index, r in enumerate(records):
            # ``figure`` is a path such as ./maps/portugal/lisboa.png into the ReasonMap repository.
            rows.append(dict(
                index=index,
                question=r['question'],
                answer=int(r['answer']),
                type=r['type'],
                image_path=map_image_name(r['country'], r['city']),
                country=r['country'],
                city=r['city'],
                station_1=r['station_1'],
                station_2=r['station_2'],
                difficulty_city=r['difficulty_city'],
                city_line_count=r['city_line_count'],
                city_transfer_count=r['city_transfer_count'],
            ))
        return pd.DataFrame(rows)[list(cls._COLUMNS)]

    def dump_image(self, line):
        return [osp.join(self.img_root, line['image_path'])]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        msgs = [dict(type='image', value=p) for p in self.dump_image(line)]
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, eval_file, **judge_kwargs):
        preds = load(eval_file)
        warn_truncated_predictions(eval_file, preds['prediction'])
        meta = self.data.set_index(self.data['index'].astype(str))

        records = []
        for _, item in preds.iterrows():
            row = meta.loc[str(item['index'])]
            extracted = extract_plus_answer(item['prediction'])
            records.append(dict(
                index=item['index'],
                type=row['type'],
                country=row['country'],
                city=row['city'],
                difficulty_city=row['difficulty_city'],
                weight=plus_weight(row['difficulty_city']),
                prediction=item['prediction'],
                extracted_answer=extracted,
                answer=row['answer'],
                hit=int(plus_is_correct(extracted, row['answer'], row['type'])),
            ))
        detail = pd.DataFrame(records)
        dump(detail, get_intermediate_file_path(eval_file, '_results'))

        metrics = self.summarize(detail)
        dump(d2df(metrics), get_intermediate_file_path(eval_file, '_acc'))
        return metrics

    @staticmethod
    def summarize(detail):
        """Accuracy and difficulty-weighted accuracy (%), overall and per question group."""
        def acc_pair(sub):
            weight = sub['weight'].astype(float)
            return (100 * float((weight * sub['hit']).sum() / weight.sum()) if len(sub) else 0.0,
                    100 * float(sub['hit'].mean()) if len(sub) else 0.0)

        metrics = {}
        metrics['Weighted Acc'], metrics['Acc'] = acc_pair(detail)
        groups = [plus_groups(t) for t in detail['type']]
        for group in ('Counting', 'TorF', 'TorF1', 'TorF2'):
            sub = detail[[group in g for g in groups]]
            metrics[f'Weighted Acc ({group})'], metrics[f'Acc ({group})'] = acc_pair(sub)
        metrics['Samples'] = int(len(detail))
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if not isinstance(metrics, dict) or 'Weighted Acc' not in metrics:
            return {}
        return {'Weighted Acc': float(metrics['Weighted Acc'])}
