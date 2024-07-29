import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import copy


class VILA(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='Efficient-Large-Model/Llama-3-VILA1.5-8b',
                 **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN # noqa E501
            from llava.conversation import conv_templates, SeparatorStyle
        except:
            warnings.warn('Please install VILA before using VILA')
            warnings.warn('Please install VILA from https://github.com/NVlabs/VILA')
            warnings.warn('Please install VLMEvalKit after installing VILA')
            warnings.warn('VILA is supported only with transformers==4.36.2')
            sys.exit(-1)

        warnings.warn('Please install the latest version of VILA from GitHub before you evaluate the VILA model.')
        assert osp.exists(model_path) or len(model_path.split('/')) == 2

        model_name = get_model_name_from_path(model_path)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device='cpu',
                device_map='cpu'
            )
        except Exception as e:
            warnings.warn(f'Error loading VILA model: {e}')
            exit(-1)

        self.model = self.model.cuda()
        if '3b' in model_path:
            self.conv_mode = 'vicuna_v1'
        if '8b' in model_path:
            self.conv_mode = 'llama_3'
        elif '13b' in model_path:
            self.conv_mode = 'vicuna_v1'
        elif '40b' in model_path:
            self.conv_mode = 'hermes-2'

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True) # noqa E501

        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Using the following kwargs for generation config: {self.kwargs}')

        self.conv_templates = conv_templates
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self. DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.SeparatorStyle = SeparatorStyle
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        # TODO see if custom prompt needed
        return False

    def generate_inner(self, message, dataset=None):

        content, images = '', []

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            elif msg['type'] == 'image':
                image = Image.open(msg['value']).convert('RGB')
                images.append(image)
                content += (self.DEFAULT_IMAGE_TOKEN + '\n')

        image_tensor = self.process_images(
            images, self.image_processor,
            self.model.config).to(self.model.device, dtype=torch.float16)

        # Support interleave text and image
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX,
                                               return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output
