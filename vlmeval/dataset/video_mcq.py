import os
import string

# uuid is for generate random video file name to save the video
import uuid
from .utils import build_judge, DEBUG_MESSAGE
import pandas as pd

from ..smp import *
from .video_base import VideoBaseDataset


def combine_images(self):
    pass


class VideoMCQDataset(VideoBaseDataset):
    
    TYPE = 'MCQ'

    DATASET_URL = {
        # TaskMeAnything_v1_videoqa
        'TaskMeAnything_v1_videoqa_random': 'https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-videoqa-random/resolve/main/TaskMeAnything-v1-videoqa-random.tsv'
        # Other Benchmarks
    }

    DATASET_MD5 = {
        # TaskMeAnything_v1_videoqa
        'TaskMeAnything_v1_videoqa_random': "627cb1409a98d3cc4f28928c2e0efdde"
        # Other Benchmarks
    }

    def base64_to_mp4(self, base64_string):
        video_name = str(uuid.uuid4())
        video_path = os.path.join(self.data_root, video_name + '.mp4')
        with open(video_path, 'wb') as f:
            f.write(base64.b64decode(base64_string))
        return video_name, video_path

    
    def build_prompt(self, line, num_frames: int, video_llm: bool, is_combine_images: bool=False):
        # if line is an index, get the line from the data
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        # the video stored in data should be a binary stream format
        video_name, video_path = self.base64_to_mp4(line['video'])
        message = []
        # setup default video or frames for ImageQA model or VideoQA model
        if video_llm:
            message.append(dict(type='video', value=video_path))
        elif is_combine_images:
            # combine images means that combine all the frames into one image, instead of provide a sequences of image.
            # This is useful for some models that only accept one image as input. 
            # And I was surprised to find that most of the time, ImageQA models perform better with a combined image instead of a sequence of frames.
            frame_paths = self.save_video_frames(video_name, num_frames)
            combined_image = combine_images(frame_paths)
            message.append(dict(type='image', value=combined_image))
        else: 
            frame_paths = self.save_video_frames(video_name, num_frames)
            for im in frame_paths:
                message.append(dict(type='image', value=im))
        
        # setup default prompt for MCQ
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        message.append(dict(type='text', value=prompt))
        return message
    
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import (
            mcq_circular_eval,
            mcq_vanilla_eval,
            report_acc,
            report_acc_MMT,
        )

        dataset = self.dataset_name

        nproc = judge_kwargs.pop('nproc', 4)

        circular = False
        

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        else:
            acc = report_acc(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc