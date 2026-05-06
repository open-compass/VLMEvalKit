import os
import os.path as osp
import pickle
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         load, md5, modelscope_flag_set)
from .utils import DEBUG_MESSAGE, build_judge
from .utils.judge_cache import (get_judge_cache_file, get_judge_detail_file, get_judge_score_file,
                                has_judge_failure, load_judge_cache)
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


class VideoTT(VideoBaseDataset):

    MD5 = 'a7ea23e35339f630b80d9160bb587049'
    SYS = ''

    FRAMES_EVAL_PROMPT = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def __init__(self, dataset='Video-TT', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['Video-TT']

    def prepare_dataset(self, dataset_name='Video-TT', repo_id='lmms-lab/video-tt'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
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

            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                target_dir = os.path.join(pth, 'video/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file.endswith('.zip') and file.startswith('Benchmark-AllVideos-LQ')
                ]
                zip_files.sort()

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith('/'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video file has been restored and stored from the zip file.')
                else:
                    print('The video file already exists.')

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_parquet(os.path.join(pth, 'data/test-00000-of-00001.parquet'))
                data_file = data_file.assign(index=range(len(data_file)))
                data_file['video'] = data_file['video_id']
                data_file['video_path'] = data_file['video_id'].apply(lambda x: f'./video/{x}.mp4')

                data_file = data_file[['index', 'video', 'video_path', 'duration', 'question', 'question_prompt',
                                       'capability', 'answer']]

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):

        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
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
            # not constrained by num_frames, get frames by fps
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

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'] + '.mp4')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        text_prompt = self.FRAMES_EVAL_PROMPT
        message.append(dict(type='text', value=text_prompt))
        question_prompt = deepcopy(line['question_prompt'])
        line['question'] += '\n' + question_prompt
        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videott import extract_characters_regex, extract_option, get_dimension_rating

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501

        judge_name = judge_kwargs.get('model', 'exact_matching')
        tmp_file = get_judge_cache_file(eval_file, 'extract', judge_name)
        legacy_tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        detail_file = get_judge_detail_file(eval_file, 'extract', judge_name)
        score_file = get_judge_score_file(eval_file, judge_name, 'json')

        if not osp.exists(detail_file):
            res = load_judge_cache(tmp_file, legacy_files=[legacy_tmp_file])

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]
            model = None
            model_built = False

            def get_model():
                nonlocal model, model_built
                if judge_name == 'exact_matching':
                    return None
                if not model_built:
                    model = build_judge(**judge_kwargs)
                    if not model.working():
                        warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                        warnings.warn(DEBUG_MESSAGE)
                        model = None
                    model_built = True
                return model

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                if extract_characters_regex(pred) == '':
                    extract_pred = res.get(idx)
                    if has_judge_failure(extract_pred):
                        extract_pred = extract_option(
                            get_model(),
                            data.loc[data['index'] == idx].to_dict(orient='records')[0],
                            'Video-MME'
                        )
                        res[idx] = extract_pred
                        dump(res, tmp_file)
                    data.loc[data['index'] == idx, 'judge_pred'] = extract_pred
                    data.loc[data['index'] == idx, 'score'] = (
                        -1 if extract_pred in ['Fail', ''] else int(extract_pred == ans)
                    )
                else:
                    extract_pred = extract_characters_regex(pred)
                    data.loc[data['index'] == idx, 'judge_pred'] = extract_pred
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, detail_file)

        rating = load(score_file) if osp.exists(score_file) else get_dimension_rating(detail_file)
        if not osp.exists(score_file):
            dump(rating, score_file)
        return rating
