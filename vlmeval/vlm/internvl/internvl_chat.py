import math
import pandas as pd
import random
import re
import string
import torch
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import warnings
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor

from .utils import (build_multi_choice_prompt,
                    build_video_prompt,
                    build_mpo_prompt,
                    build_mcq_cot_prompt,
                    build_qa_cot_prompt,
                    mpo_post_processing,
                    reorganize_prompt,
                    split_model, load_image)
from .utils import mpo_prompt_with_final_answer, mpo_prompt_without_final_answer
from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY
from ...smp import *


class InternVLChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_path='OpenGVLab/InternVL-Chat-V1-5',
                 load_in_8bit=False,
                 use_mpo_prompt=False,
                 version='V1.0',
                 **kwargs):

        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.37.2', 'ge')

        self.use_mpo_prompt = use_mpo_prompt
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        if auto_split_flag():
            device_map, visible_devices = split_model(model_path=model_path)
            self.device = visible_devices[0]
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device_map).eval()
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True).eval().cuda()
            self.device = 'cuda'

        self.image_size = self.model.config.vision_config.image_size
        self.version = version
        kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        if self.use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)
        return message

    def set_max_num(self, dataset):
        # The total limit on the number of images processed, set to avoid Out-of-Memory issues.
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            self.max_num = 12
        elif listinstr(res_18_datasets, dataset):
            self.max_num = 18
        elif listinstr(res_24_datasets, dataset):
            self.max_num = 24
        else:
            self.max_num = 6

    def generate_v1_2(self, message, dataset=None):
        self.INTERLEAVE = False
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values=pixel_values,
                                       question=prompt, generation_config=self.kwargs)
        return response

    def generate_v1_5(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])

        if DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            pixel_values_list = []
            for file_name in image_path:
                pixel_values_list.append(load_image(file_name, max_num=max_num).to(self.device).to(torch.bfloat16))
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            pixel_values = load_image(image_path, max_num=max_num).to(self.device).to(torch.bfloat16)
        else:
            pixel_values = None
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=self.kwargs,
                verbose=True)
        return response

    def generate_v2(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = reorganize_prompt(message, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=self.kwargs,
                verbose=True
            )

        if self.use_mpo_prompt:
            response = mpo_post_processing(response, dataset)
        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        print(f'InternVL model version: {self.version}')
        if self.version in ['V1.1', 'V1.2']:
            return self.generate_v1_2(message, dataset)
        elif self.version == 'V1.5':
            return self.generate_v1_5(message, dataset)
        elif self.version == 'V2.0':
            return self.generate_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version: {self.version}')

    def build_history(self, message):
        # Global Variables
        image_path = []
        image_cnt = 0

        def concat_tilist(tilist):
            nonlocal image_cnt  # Declare image_cnt as nonlocal to modify it
            prompt = ''
            for item in tilist:
                # Substitute the pattern in the text
                if item['type'] == 'text':
                    prompt += re.sub(self.pattern, self.replacement, item['value'])
                elif item['type'] == 'image':
                    image_cnt += 1
                    prompt += '<image>\n'
                    image_path.append(item['value'])
            return prompt

        # Only previous messages
        assert len(message) % 2 == 0
        history = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            history.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))

        return history, image_path, image_cnt

    def chat_inner_v2(self, message, dataset=None):

        if len(message) > 1:
            history, image_path, image_cnt = self.build_history(message[:-1])
        else:
            history, image_path, image_cnt = None, [], 1
        current_msg = message[-1]
        question = ''

        # If message is just text in the conversation
        if len(current_msg['content']) == 1 and current_msg['content'][0]['type'] == 'text':
            question = current_msg['content'][0]['value']
            question = re.sub(self.pattern, self.replacement, question)  # Fix pattern as per InternVL
        else:
            for msg in current_msg['content']:
                if msg['type'] == 'text':
                    question += re.sub(self.pattern, self.replacement, msg['value'])
                elif msg['type'] == 'image':
                    image_cnt += 1
                    question += '<image>\n'
                    image_path.append(msg['value'])

        if image_cnt > 1:
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_cnt == 1:
            upscale_flag = listinstr(['MMMU_DEV_VAL'], dataset)
            pixel_values = load_image(
                image_path, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=question,
            generation_config=self.kwargs,
            history=history,
            return_history=True
        )

        response = re.sub(self.reverse_pattern, self.reverse_replacement, response)

        return response

    def chat_inner(self, message, dataset=None):
        self.set_max_num(dataset)

        if self.version in ['V1.1', 'V1.2']:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
        elif self.version == 'V1.5':
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
        elif self.version == 'V2.0':
            kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
            self.kwargs = kwargs_default
            return self.chat_inner_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
