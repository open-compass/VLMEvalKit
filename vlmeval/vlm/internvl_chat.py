import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, CLIPImageProcessor
import warnings
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..utils import DATASET_TYPE
import pandas as pd
import string


class InternVLChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='OpenGVLab/InternVL-Chat-Chinese-V1-1', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        device = torch.cuda.current_device()
        self.device = device
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).eval().to(device)
        self.image_size = self.model.config.vision_config.image_size

        if 'V1-1' in model_path:
            kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=5)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        response = self.model.chat(self.tokenizer, pixel_values=pixel_values,
                                   question=prompt, generation_config=self.kwargs)
        return response
