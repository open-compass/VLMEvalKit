import os
import pandas as pd
from huggingface_hub import snapshot_download
from .image_mcq import ImageMCQDataset
from ..smp import *


class MedQDEGBenchDataset(ImageMCQDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        'MedQDEGBench_simulate_dev': 'MedQ-Robust-simulate-dev.tsv',
        'MedQDEGBench_simulate_test': 'MedQ-Robust-simulate-test.tsv',
        'MedQDEGBench_good_dev': 'MedQ-Robust-good-dev.tsv',
        'MedQDEGBench_good_test': 'MedQ-Robust-good-test.tsv',
    }

    DATASET_MD5 = {
        'MedQDEGBench_simulate_dev': None,
        'MedQDEGBench_simulate_test': None,
        'MedQDEGBench_good_dev': None,
        'MedQDEGBench_good_test': None,
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def __init__(self, dataset='MedQDEGBench_simulate_test', **kwargs):
        super().__init__(dataset, **kwargs)

    def prepare_dataset(self, dataset_name, repo_id='jiyaoliufd/MedQ-DEG-Bench'):
        def check_integrity(pth):
            data_file = osp.join(pth, self.DATASET_URL[dataset_name])
            return os.path.exists(data_file)

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

        data_file = osp.join(dataset_path, self.DATASET_URL[dataset_name])
        return dict(root=dataset_path, data_file=data_file)

    def load_data(self, dataset):
        dataset_info = self.prepare_dataset(dataset)
        data_path = dataset_info['data_file']
        data_root = dataset_info['root']

        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Data file not found: {data_path}')

        data = load(data_path)
        self.data_root = data_root

        if 'index' in data.columns:
            data['index'] = data['index'].astype(str)

        if 'answer' in data.columns:
            data['answer'] = data['answer'].apply(
                lambda x: str(x).strip().upper() if pd.notna(x) else x
            )

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = line['question']
        options = {}

        for opt in ['A', 'B', 'C', 'D', 'E']:
            if opt in line and pd.notna(line[opt]) and str(line[opt]).strip():
                options[opt] = str(line[opt]).strip()

        option_text = '\n'.join([f'{k}. {v}' for k, v in options.items()])

        prompt = (
            f'{question}\n\n'
            f'{option_text}\n\n'
            f'Answer with the option letter from the given choices directly.'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs.append(dict(type='image', value=tgt_path))

        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)

        assert 'answer' in data.columns and 'prediction' in data.columns

        lt = len(data)
        data['prediction'] = data['prediction'].apply(self._extract_option_letter)
        data['answer'] = data['answer'].apply(lambda x: str(x).strip().upper())
        data['hit'] = data.apply(
            lambda row: str(row['answer']).upper() == str(row['prediction']).upper(),
            axis=1,
        )

        overall_acc = data['hit'].mean()
        result = {'Overall': overall_acc, 'Total': lt}

        if 'modality' in data.columns:
            for mod in sorted(data['modality'].dropna().unique()):
                sub = data[data['modality'] == mod]
                result[f'Modality_{mod}'] = sub['hit'].mean()

        if 'degradation_type' in data.columns:
            for dt in sorted(data['degradation_type'].dropna().unique()):
                sub = data[data['degradation_type'] == dt]
                result[f'DegType_{dt}'] = sub['hit'].mean()

        if 'degradation_severity' in data.columns:
            for sev in sorted(data['degradation_severity'].dropna().unique()):
                sub = data[data['degradation_severity'] == sev]
                result[f'Severity_{sev}'] = sub['hit'].mean()

        if 'source' in data.columns:
            for src in sorted(data['source'].dropna().unique()):
                sub = data[data['source'] == src]
                result[f'Source_{src}'] = sub['hit'].mean()

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(result, score_file)

        return result

    def _extract_option_letter(self, prediction):
        if pd.isna(prediction):
            return prediction

        prediction = str(prediction).strip()

        if prediction.upper() in ['A', 'B', 'C', 'D', 'E']:
            return prediction.upper()

        import re

        match = re.match(r'^([A-E])[.\)\s]', prediction.upper())
        if match:
            return match.group(1)

        match = re.search(
            r'(?:answer\s+is\s+|option\s+)([A-E])(?:\s|$|[^A-Z])',
            prediction.upper(),
        )
        if match:
            return match.group(1)

        match = re.search(r'^([A-E])\s*[-:]', prediction.upper())
        if match:
            return match.group(1)

        matches = re.findall(
            r'(?:^|[^A-Z])([A-E])(?:[^A-Z]|$)', prediction.upper()
        )
        if matches:
            return matches[0]

        return prediction
