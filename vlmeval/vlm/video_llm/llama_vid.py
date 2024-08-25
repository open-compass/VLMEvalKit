import torch
import warnings
import copy as cp
import numpy as np
import sys
import os
from ..base import BaseModel
from ...smp import isimg, listinstr, load, dump, download_file
from ...dataset import DATASET_TYPE
from decord import VideoReader, cpu
from huggingface_hub import snapshot_download


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, total_frame_num, fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def change_file(file_path, mm_vision_tower):
    org_data = load(file_path)
    org_data['image_processor'] = './vlmeval/vlm/video_llm/configs/llama_vid/processor/clip-patch14-224'
    org_data['mm_vision_tower'] = mm_vision_tower
    dump(org_data, file_path)


class LLaMAVID(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path='YanweiLi/llama-vid-7b-full-224-video-fps-1', **kwargs):
        assert model_path is not None
        try:
            from llamavid.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install LLaMA-VID from https://github.com/dvlab-research/LLaMA-VID.')
            sys.exit(-1)

        model_base = None
        model_name = get_model_name_from_path(model_path)

        eva_vit_g_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth'
        true_model_path = snapshot_download(model_path)
        eva_vit_path = os.path.join(true_model_path, 'eva_vit_g.pth')
        if not os.path.exists(eva_vit_path):
            download_file(eva_vit_g_url, eva_vit_path)
        config_path = os.path.join(true_model_path, 'config.json')
        change_file(config_path, eva_vit_path)

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            true_model_path, model_base, model_name, None, device_map='cpu', device='cpu'
        )
        model.cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.processor = image_processor
        self.context_len = context_len
        self.kwargs = kwargs
        self.nframe = 8

    def get_model_output(self, model, video_processor, tokenizer, video, qs):
        from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llamavid.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
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
