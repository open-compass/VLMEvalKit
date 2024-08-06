import torch
import os
import warnings
import copy as cp
import numpy as np
import sys
from ..base import BaseModel
from ...smp import isimg, listinstr
from ...dataset import DATASET_TYPE


class VideoChatGPT(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path='LanguageBind/Video-LLaVA-7B', **kwargs):
        assert model_path is not None
        try:
            from video_chatgpt.eval.model_utils import initialize_model
        except:
            warnings.warn(
                'Please export video_chatgpt to python path.'
            )
            sys.exit(-1)
        model_name = 'Path to LLaVA-7B-Lightening Model'
        if not os.path.exists(model_path):
            warnings.warn(
                f'LLaVA-Lightening Model {model_path} does not exist. Please download from https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1 .'
            )
            sys.exit(-1)
        model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(
            model_name, model_path
        )
        self.tokenizer = tokenizer
        self.model = model
        self.processor = image_processor
        self.context_len = video_token_len
        self.kwargs = kwargs
        self.vision_tower = vision_tower
        self.nframe = 8

    def get_model_output(self, model, video_processor, tokenizer, video, qs):
        from video_chatgpt.eval.model_utils import load_video
        from video_chatgpt.inference import video_chatgpt_infer
        conv_mode = 'video-chatgpt_v1'

        video_formats = ['.mp4', '.avi', '.mov', '.mkv']

        video_frames = load_video(video)

        # Run inference on the video and add the output to the list
        output = video_chatgpt_infer(
            video_frames,
            qs,
            conv_mode,
            model,
            self.vision_tower,
            tokenizer,
            video_processor,
            self.context_len,
        )
        return output

    def generate_inner(self, message, dataset=None):
        question, video = self.message_to_promptvideo(message)
        response = self.get_model_output(self.model, self.processor, self.tokenizer, video, question)
        return response
