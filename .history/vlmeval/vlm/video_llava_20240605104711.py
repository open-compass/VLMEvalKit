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
        tokenizer, model, processor, context_len = load_pretrained_model(model_pth, model_base, model_name)
        model = model.cuda()

        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=1024, use_cache=True
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        from videollava.mm_utils import (
            get_model_name_from_path,
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
        return output
