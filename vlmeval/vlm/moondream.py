import torch
import re
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import copy


class Moondream1(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self,
                 model_path='vikhyatk/moondream1',
                 **kwargs):
        try:
            from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
        except:
            warnings.warn('''Please install Transformers version 4.36.2 by running: "pip install transformers==4.36.2",
            please intall torchvision>=0.16.''')
        warnings.warn('''Please install Transformers version 4.36.2 by running: "pip install transformers==4.36.2",
            please intall torchvision>=0.16.''')
        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
            .to(torch.device('cuda'))
        )
        self.tokenizer = Tokenizer.from_pretrained(model_path)

        default_kwargs = dict(
            max_new_tokens=512,
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        images = []
        prompt = ''

        for s in message:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                prompt += s['value']

        images = [Image.open(s) for s in images]
        enc_image = self.model.encode_image(images[0])

        prompt_wtmpl = prompt = f'<image>\n\nQuestion: {prompt}\n\nAnswer: '
        answer = self.model.generate(
            enc_image, prompt_wtmpl, eos_text='<END>', tokenizer=self.tokenizer, **self.kwargs)[0]
        cleaned_answer = re.sub('<$', '', re.sub('END$', '', answer)).strip()
        return cleaned_answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset == 'MMVet':
            prompt = question + '\nAnswer the question directly. '
        elif DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'Hint: {hint}\n' if hint is not None else ''
            prompt += f'{question}\n'
            prompt += (
                f'{options_prompt}\nAnswer with the option’s letter from the given choices directly. '
                if len(options) else 'Answer the question directly. '
            )
        else:
            raise NotImplementedError

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message


class Moondream2(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self,
                 model_path='vikhyatk/moondream2',
                 **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except:
            warnings.warn('''Please install Transformers version 4.44 by running: "pip install transformers==4.44.0",
            please intall torchvision>=0.16.''')
        warnings.warn('''Please install Transformers version 4.44 by running: "pip install transformers==4.44.0",
        please intall torchvision>=0.16.''')
        assert osp.exists(model_path) or splitlen(model_path) == 2

        flash_attn_flag = False
        try:
            import flash_attn
            flash_attn_flag = True
        except ImportError:
            pass
        if flash_attn_flag:
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16,
                                                     attn_implementation='flash_attention_2')
                .to(torch.device('cuda'))
            )
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
                .to(torch.device('cuda'))
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        default_kwargs = dict(
            max_new_tokens=512,
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        images = []
        prompt = ''
        for s in message:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                prompt += s['value']

        images = [Image.open(s) for s in images]
        enc_image = self.model.encode_image(images[0])

        prompt_wtmpl = prompt = f'<image>\n\nQuestion: {prompt}\n\nAnswer: '
        answer = self.model.generate(
            enc_image, prompt_wtmpl, eos_text='<END>', tokenizer=self.tokenizer, **self.kwargs)[0]
        cleaned_answer = re.sub('<$', '', re.sub('END$', '', answer)).strip()
        return cleaned_answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset == 'MMVet':
            prompt = question + '\nAnswer the question directly. '
        elif DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'Hint: {hint}\n' if hint is not None else ''
            prompt += f'{question}\n'
            prompt += (
                f'{options_prompt}\nAnswer with the option’s letter from the given choices directly. '
                if len(options) else 'Answer the question directly. '
            )
        else:
            raise NotImplementedError

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
