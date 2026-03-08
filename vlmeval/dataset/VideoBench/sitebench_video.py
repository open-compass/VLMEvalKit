import os
import ast
import string
import decord
import numpy as np

from PIL import Image

from vlmeval.smp import *

from .video_base import VideoBaseDataset
from ..sitebench import SiteBenchBase


class SiteBenchVideo(SiteBenchBase, VideoBaseDataset):

    MD5 = ''

    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    LMUData_root = LMUDataRoot()
    DATASET_URL = {
        'SiteBenchVideo': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SiteBenchVideo.tsv'  # noqa: E501
    }
    DATASET_MD5 = {
        'SiteBenchVideo': 'bb2ac531fa83cf8280b23c25d738922d'
    }

    def __init__(self, dataset='SiteBenchVideo', pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['SiteBenchVideo']

    def prepare_dataset(self, dataset_name='SiteBenchVideo'):
        data = super().prepare_tsv(
            self.DATASET_URL[self.dataset_name],
            self.DATASET_MD5[self.dataset_name]
        )

        dataset_path = self.download_sitebench(self.repo_id)
        self.dataset_path = dataset_path

        # === Transfer rel path to abs path ===
        if 'video_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\/')))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data['video'] = data['video_path'].map(to_abs)

        new_data_path = os.path.join(self.LMUData_root, 'SiteBenchVideo_abs_path.tsv')
        if not os.path.exists(new_data_path):
            dump(data, new_data_path)

        return dict(data_file=new_data_path, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):
        vid_path = video
        rel_video_path = os.path.relpath(video, self.dataset_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        if self.nframe > 0 and self.fps < 0:
            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
            # Use os.path.relpath for robust relative path extraction
            frame_paths = self.frame_paths(rel_video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps

            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(rel_video_path, len(indices))

        missing = [
            (idx, pth) for idx, pth in zip(indices, frame_paths)
            if not os.path.exists(pth)
        ]

        if missing and not video_llm:
            for frame_idx, pth in missing:
                try:
                    frame_data = vid[frame_idx].asnumpy()
                    Image.fromarray(frame_data).save(pth)
                except Exception as e:
                    error_msg = f"Error saving frame {frame_idx} from {vid_path}: {str(e)}"
                    print(error_msg)

                    raise ValueError(error_msg) from e

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question']
        raw_options = ast.literal_eval(line['candidates'])

        option_labels = list(string.ascii_uppercase)
        assert len(raw_options) <= len(option_labels), 'Too many options, extend option_labels if needed'

        options = [f'{label}: {opt}' for label, opt in zip(option_labels, raw_options)]
        formatted_options = '\n'.join(options)

        # video prompt from SITE paper
        pre_prompt = (
            'Select the best answer to the following multiple-choice question based on the video. '
            'Respond with only the letter of the correct option.'
        )
        post_prompt = 'Give me the answer letter directly. The best answer is:'

        question_text = 'Question: ' + question + '\n'
        option_text = 'Options:\n' + formatted_options + '\n'

        prompt = pre_prompt + '\n' + question_text + option_text + post_prompt

        message = []
        if video_llm:
            message.append(dict(type='video', value=line['video']))
        else:
            frames, _, _ = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))
        return message
