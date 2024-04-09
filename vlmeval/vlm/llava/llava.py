import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..utils import DATASET_TYPE


class LLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self,
                 model_pth='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install llava before using LLaVA')
            sys.exit(-1)

        warnings.warn('Please install the latest version of llava from github before you evaluate the LLaVA model. ')
        assert osp.exists(model_pth) or splitlen(model_pth) == 2

        if model_pth == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_pth == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_pth)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_pth,
                model_base=None,
                model_name=model_name,
                device='cpu',
                device_map='cpu'
            )
        except:
            if 'ShareGPT4V' in model_pth:
                import llava
                warnings.warn(
                    'Please manually remove the encoder type check in '
                    f'{llava.__path__[0]}/model/multimodal_encoder/builder.py '
                    'Line 8 to use the ShareGPT4V model. ')
            else:
                warnings.warn('Unknown error when loading LLaVA model.')
            exit(-1)

        self.model = self.model.cuda()
        self.conv_mode = 'llava_v1'

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1)
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
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
        from llava.conversation import conv_templates, SeparatorStyle
        prompt, image_path = self.message_to_promptimg(message)
        image = Image.open(image_path).convert('RGB')
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output


class LLaVA_Next(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_pth='llava-hf/llava-v1.6-vicuna-7b-hf', **kwargs):
        import transformers
        assert version_cmp(transformers.__version__, '4.39.0', 'ge')
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        self.model_pth = model_pth
        if '34b' in model_pth.lower():
            self.processor = LlavaNextProcessor.from_pretrained(self.model_pth, use_fast=False)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_pth)
        flash_attn_flag = False
        try:
            import flash_attn
            flash_attn_flag = True
        except ImportError:
            pass

        if flash_attn_flag:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_pth, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True)
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_pth, torch_dtype=torch.float16, low_cpu_mem_usage=True)

        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def apply_prompt_template(self, prompt):
        model_pth = self.model_pth.lower()
        if 'mistral' in model_pth:
            s = f'[INST] <image>\n {prompt} [/INST]'
        elif 'vicuna' in model_pth:
            s = (
                'A chat between a curious human and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                f'USER: <image>\n{prompt} ASSISTANT:'
            )
        elif '34b' in model_pth:
            s = (
                f'<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|>'
                '<|im_start|>assistant\n'
            )
        else:
            raise NotImplementedError(f'Prompt template for {model_pth} not implemented.')
        return s

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
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        image = Image.open(image_path)
        prompt_wtmpl = self.apply_prompt_template(prompt)
        inputs = self.processor(prompt_wtmpl, image, return_tensors='pt').to('cuda')
        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        if '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()

        return answer
