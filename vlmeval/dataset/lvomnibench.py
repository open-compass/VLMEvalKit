import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         get_logger, load, md5, modelscope_flag_set)
from .video_base import VideoBaseDataset

logger = get_logger(__name__)


class LVOmniBench(VideoBaseDataset):

    MD5 = None

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='LVOmniBench', nframe=0, fps=-1):
        self.dataset_name = dataset
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['LVOmniBench']

    def prepare_dataset(self, dataset_name='LVOmniBench', repo_id='KD-TAO/LVOmniBench'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not os.path.exists(data_file):
                return False
            if self.MD5 is not None and md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file):
                    if self.MD5 is None or md5(data_file) == self.MD5:
                        return

                with open(osp.join(pth, 'data.json'), 'r') as f:
                    json_data = json.load(f)

                data_list = []
                for item in json_data:
                    video_id = item['video_id']
                    data_list.append({
                        'index': int(item['question_id']),
                        'video': video_id,
                        'video_path': f'./videos/{video_id}.mp4',
                        'question': item['question'],
                        'candidates': str(item['options']),
                        'answer': item['correct_option'],
                        'question_type': item['question_type'],
                        'audio_type': item['audio_type'],
                        'difficulty': item['difficulty'],
                        'video_category': item['video_category'],
                    })

                df = pd.DataFrame(data_list)
                df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):
        vid_path = osp.join(self.data_root, 'videos', video + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        message = []
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'videos', line['video'] + '.mp4')))
        else:
            frames, indices, video_info = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        # Official LVOmniBench prompt format
        options = eval(line['candidates'])
        options_str = '\n'.join(options)
        prompt = (
            f"Question: {line['question']}\n"
            f"Options:\n{options_str}\n\n"
            "Select the best answer from the options above. "
            "Directly provide the letter representing your choice (A/B/C/D) and nothing else. "
            "Do not include the full text of the option, do not provide any explanation."
        )
        message.append(dict(type='text', value=prompt))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from .utils.lvomnibench import extract_characters_regex, get_dimension_rating

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be a supported format (xlsx/json/tsv) file'

        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
                extracted = extract_characters_regex(pred)
                data.loc[data['index'] == idx, 'score'] = int(extracted == ans) if extracted else -1

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
