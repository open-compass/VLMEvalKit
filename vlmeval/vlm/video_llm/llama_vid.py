import torch
import warnings
import copy as cp
import numpy as np
import sys
import os
from ..base import BaseModel
from ...smp import isimg, listinstr
from ...dataset import DATASET_TYPE
from decord import VideoReader, cpu

def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


class LLaMAVID(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path=None, **kwargs):
        assert model_path is not None
        try:
            from llamavid.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install LLaMA-VID from https://github.com/FangXinyu-0913/Video-LLaVA.')
            sys.exit(-1)

        model_base = None
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, None
        )
        self.tokenizer = tokenizer
        self.model = model
        self.processor = image_processor
        self.context_len = context_len
        self.kwargs = kwargs
        self.nframe = 8

    def get_model_output(self, model, video_processor, tokenizer, video, qs):
        from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llamavid.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        original_qs = cp.deepcopy(qs)
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv_mode = 'vicuna_v1'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        video_formats = ['.mp4', '.avi', '.mov', '.mkv']
        # Load the video file
        for fmt in video_formats:  # Added this line
            # temp_path = os.path.join(args.video_dir, f"{video}{fmt}")
            temp_path = f"{video}{fmt}"
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video):
            video = load_video(video)
            video = video_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        cur_prompt = original_qs
        with torch.inference_mode():
            model.update_prompt([[cur_prompt]])
            output_ids = model.generate(
                input_ids,
                images=video,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs

    def generate_inner(self, message, dataset=None):
        question, video = self.message_to_promptvideo(message)
        response = self.get_model_output(self.model, self.processor, self.tokenizer, video, question)
        return response
