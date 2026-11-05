import os.path as osp
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from ..smp import dump, get_cache_path, get_file_extension, get_intermediate_file_path, load
from .video_base import VideoBaseDataset


class AoTBench(VideoBaseDataset):
    """AoTBench from "Seeing the Arrow of Time in Large Multimodal Models" (NeurIPS 2025).

    Paper: https://arxiv.org/abs/2506.03340
    Code: https://github.com/zihuixue/seeAoT
    Data: https://huggingface.co/datasets/sherryxzh/AoTBench

    Each subset is a separate dataset, because the benchmark defines no overall score. Scoring follows
    the official `eval/read_qa.py`: the first character of the reply is compared with the answer letter.

    A few released files have no video track. The official script returns 'Error' for them, which counts as
    wrong, so they stay in the denominator here too: `has_video` marks them and `evaluate` scores them 0.
    """

    TYPE = 'Video-MCQ'
    REPO_ID = 'sherryxzh/AoTBench'
    DATA_FILES = {
        'AoTBench_ReverseFilm': 'ReverseFilm.json',
        'AoTBench_UCF101': 'UCF101.json',
        'AoTBench_Rtime_t2v': 'Rtime_t2v.json',
        'AoTBench_Rtime_v2t': 'Rtime_v2t.json',
        'AoTBench_QA': 'AoTBench_QA.json',
    }

    def __init__(self, dataset='AoTBench_QA', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATA_FILES)

    def prepare_dataset(self, dataset_name='AoTBench_QA', repo_id=None):
        repo_id = repo_id or self.REPO_ID

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            data = load(data_file)
            return 'has_video' in data and all(osp.exists(osp.join(pth, video)) for video in data['video'])

        def has_video(path):
            import cv2

            capture = cv2.VideoCapture(path)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            capture.release()
            return frame_count > 0

        def generate_tsv(pth):
            records = load(osp.join(pth, 'data_files', self.DATA_FILES[dataset_name]))
            data = pd.DataFrame(records).rename(columns={'video_name': 'video', 'ans': 'answer'})
            columns = ['qa_idx', 'video', 'question', 'answer']
            if 'question_type' in data:
                columns.append('question_type')
            data = data[columns]
            data.insert(0, 'index', range(len(data)))
            data['has_video'] = data['video'].map({
                video: has_video(osp.join(pth, video)) for video in data['video'].unique()
            })
            data.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

        dataset_path = get_cache_path(repo_id)
        if dataset_path is None or not check_integrity(dataset_path):
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            if not osp.exists(osp.join(dataset_path, 'videos')):
                with zipfile.ZipFile(osp.join(dataset_path, 'videos.zip')) as zf:
                    zf.extractall(dataset_path)
            generate_tsv(dataset_path)

        return dict(root=dataset_path, data_file=osp.join(dataset_path, f'{dataset_name}.tsv'))

    def save_video_frames(self, video):
        # `video` keeps its extension, since some files use `.MP4`. Frames are spaced as in qwen-vl-utils,
        # which the official evaluation uses. OpenCV reads them in order: decord cannot open files whose
        # video track starts late, and seeking is unreliable there.
        import cv2

        capture = cv2.VideoCapture(osp.join(self.data_root, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps > 0:
            step_size = capture.get(cv2.CAP_PROP_FPS) / self.fps
            indices = [int(i * step_size) for i in range(max(int(frame_count / step_size), 1))]
            frame_paths = self.frame_paths_fps(video, len(indices))
        else:
            indices = np.linspace(0, frame_count - 1, self.nframe).round().astype(int).tolist()
            frame_paths = self.frame_paths(video)

        if not all(osp.exists(pth) for pth in frame_paths):
            targets = defaultdict(list)
            for idx, pth in zip(indices, frame_paths):
                targets[idx].append(pth)
            with portalocker.Lock(osp.join(self.frame_root, video + '.lock'), 'w', timeout=30):
                for idx in range(max(indices) + 1):
                    if not capture.grab():
                        break
                    if idx in targets:
                        frame = Image.fromarray(capture.retrieve()[1][..., ::-1])
                        for pth in targets[idx]:
                            frame.save(pth)
        capture.release()
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        message = []
        if line['has_video'] and video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video'])))
        elif line['has_video']:
            message.extend(dict(type='image', value=pth) for pth in self.save_video_frames(line['video']))
        message.append(dict(type='text', value=line['question']))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be a supported format (xlsx/json/tsv) file'

        data = load(eval_file)
        predictions = data['prediction'].fillna('').astype(str)
        data['score'] = [int(pred[:1] == str(ans)) for pred, ans in zip(predictions, data['answer'])]
        if 'has_video' in data:
            data['score'] *= data['has_video'].astype(int)
        dump(data, get_intermediate_file_path(eval_file, '_score'))

        rating = {'overall': data['score'].mean() * 100, 'correct': int(data['score'].sum()), 'total': len(data)}
        if 'question_type' in data:
            for name, group in data.groupby('question_type'):
                rating[name] = group['score'].mean() * 100
        dump(rating, get_intermediate_file_path(eval_file, '_rating', 'json'))
        return rating
