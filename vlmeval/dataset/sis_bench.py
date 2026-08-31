import os.path as osp
from collections import OrderedDict

import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import dump, get_cache_path, get_file_extension, get_intermediate_file_path, load
from .utils.multiple_choice import extract_characters_regex
from .video_base import VideoBaseDataset


class SISBench(VideoBaseDataset):

    TYPE = 'Video-MCQ'
    REPO_ID = 'choucsan/SIS-Bench'
    CHOICES = ('A', 'B', 'C', 'D')

    SPATIAL_COGNITION_TASKS = (
        'object_existence',
        'object_attribute',
        'relative_direction',
        'landmark_appearance_order',
        'landmark_recall',
        'positional_relationship',
        'spatial_consistency',
        'spatio-temporal_consistency',
    )
    SELF_AWARENESS_TASKS = (
        'action_recognition',
        'action_sequence',
        'action_recall',
        'action_prediction',
        'path_planning',
    )
    TASKS = SPATIAL_COGNITION_TASKS + SELF_AWARENESS_TASKS

    @classmethod
    def supported_datasets(cls):
        return ['SIS-Bench', 'SIS-Bench_8frame', 'SIS-Bench_32frame', 'SIS-Bench_1fps']

    @classmethod
    def _generate_tsv(cls, dataset_path, dataset_name):
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        source_file = osp.join(dataset_path, 'SIS-Bench.jsonl')
        data = pd.read_json(source_file, lines=True)
        required = {
            'question_id', 'video_name', 'concat_num', 'task_type', 'question', 'options', 'answer'
        }
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f'SIS-Bench annotations are missing fields: {sorted(missing)}')

        unknown_tasks = set(data['task_type']).difference(cls.TASKS)
        if unknown_tasks:
            raise ValueError(f'SIS-Bench contains unknown task types: {sorted(unknown_tasks)}')

        data = data.assign(index=range(len(data)))
        data['video'] = data['video_name'].map(lambda value: osp.splitext(value)[0])
        data['video_path'] = data['video_name'].map(lambda value: osp.join('video', value))
        for choice in cls.CHOICES:
            data[choice] = data['options'].map(lambda options: options[choice])

        columns = [
            'index', 'question_id', 'video', 'video_path', 'concat_num', 'task_type', 'question',
            *cls.CHOICES, 'answer'
        ]
        data[columns].to_csv(data_file, sep='\t', index=False)
        return data_file

    @classmethod
    def _check_integrity(cls, dataset_path, dataset_name):
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        if not osp.isfile(data_file):
            return False

        data = load(data_file)
        required = {'question', 'video', 'video_path', 'task_type', 'answer', *cls.CHOICES}
        if not required.issubset(data.columns):
            return False
        return all(
            osp.isfile(osp.join(dataset_path, path)) for path in data['video_path'].unique())

    def prepare_dataset(self, dataset_name='SIS-Bench', repo_id=REPO_ID):
        dataset_path = get_cache_path(repo_id)
        if dataset_path is None or not self._check_integrity(dataset_path, dataset_name):
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            self._generate_tsv(dataset_path, dataset_name)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=osp.join(dataset_path, 'video'), data_file=data_file)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            if line >= len(self):
                raise IndexError(f'SIS-Bench index out of range: {line}')
            line = self.data.iloc[line]

        options = '\n'.join(f'({choice}) {line[choice]}' for choice in self.CHOICES)
        prompt = (f"{line['question']}\nOptions:\n{options}\n"
                  'Answer with only the letter of the correct option.')

        if video_llm:
            video_path = osp.join(self.data_root, line['video'] + '.mp4')
            return [dict(type='video', value=video_path), dict(type='text', value=prompt)]

        frame_paths = self.save_video_frames(line['video'])
        message = [dict(type='image', value=path) for path in frame_paths]
        message.append(dict(type='text', value=prompt))
        return message

    @staticmethod
    def _accuracy(data):
        return float(data['score'].mean() * 100) if len(data) else 0.0

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        del judge_kwargs
        if get_file_extension(eval_file) not in ['xlsx', 'json', 'tsv']:
            raise ValueError('SIS-Bench predictions must be an xlsx, json, or tsv file')

        data = load(eval_file)
        required = {'prediction', 'answer', 'task_type'}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f'SIS-Bench prediction file is missing fields: {sorted(missing)}')

        unknown_tasks = set(data['task_type']).difference(cls.TASKS)
        if unknown_tasks:
            raise ValueError(
                f'SIS-Bench prediction file contains unknown task types: {sorted(unknown_tasks)}')

        def extract_prediction(value):
            if pd.isna(value):
                return ''
            return extract_characters_regex(str(value), choices=['(A)', '(B)', '(C)', '(D)'])

        data['predicted_answer'] = data['prediction'].map(extract_prediction)
        data['score'] = (data['predicted_answer'].str.upper() == data['answer'].astype(
            str).str.strip().str.upper()).astype(int)

        metrics = OrderedDict()
        metrics['Overall'] = cls._accuracy(data)
        metrics['Spatial Avg'] = cls._accuracy(data[data['task_type'].isin(
            cls.SPATIAL_COGNITION_TASKS)])
        metrics['Self Avg'] = cls._accuracy(data[data['task_type'].isin(cls.SELF_AWARENESS_TASKS)])
        for task_type in cls.TASKS:
            metrics[task_type] = cls._accuracy(data[data['task_type'] == task_type])

        score_file = get_intermediate_file_path(eval_file, '_score')
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(data, score_file)
        dump(dict(metrics), rating_file)
        return dict(metrics)

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'Overall' in metrics:
            return {'Overall': metrics['Overall']}
        return super().report_primary_metric(metrics)
