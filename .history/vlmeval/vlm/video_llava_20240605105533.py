import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...utils import DATASET_TYPE

class Video_LLaVA(BaseModel):
    def __init__(self, model_pth = 'checkpoints/Video-LLaVA-7B', model_base = 'None', **kwargs):
        try:
            from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
            from videollava.model.builder import load_pretrained_model
        except:
            warnings.warn('Please install videollava before using Video-LLaVA')
            sys.exit(-1)
        model_name = get_model_name_from_path(model_pth)
        self.tokenizer, model, self.processor, context_len = load_pretrained_model(model_pth, model_base, model_name)
        self.model = model.cuda()
        self.conv_mode = "llava_v1"

        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=1024, use_cache=True
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        from videollava.mm_utils import (
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from videollava.conversation import conv_templates, SeparatorStyle
        from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN

        # Support interleave text and image
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], 'PLACEHOLDER')
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            elif msg['type'] == 'image':
                if self.model.config.mm_use_im_start_end:
                    content += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                else:
                    content += DEFAULT_IMAGE_TOKEN + '\n'
                images.append(msg['value'])

        images = [Image.open(s).convert('RGB') for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)
        prompt = prompt.replace('PLACEHOLDER', content)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # return output


        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN] * 8) + DEFAULT_VID_END_TOKEN + '\n' + qs
        else:
            qs = ''.join([DEFAULT_IMAGE_TOKEN] * 8) + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        video_tensor = (
            video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
        )
        video_tensor = video_tensor[:, 3, :, :].unsqueeze(1).repeat(1, 8, 1, 1)
        # print(video_tensor.shape)
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            .unsqueeze(0)
            .to(args.device)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[video_tensor],
                do_sample=False,
                temperature=0.0,
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
        print(outputs)
        return outputs
