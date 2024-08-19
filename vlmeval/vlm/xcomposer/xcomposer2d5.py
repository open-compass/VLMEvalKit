import re

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ...dataset import DATASET_TYPE
from ...smp import *
from ..base import BaseModel

pattern = re.compile(r'[A-Z]')


def padding_560(b):
    width, height = b.size
    tar = int(np.ceil(height / 560) * 560)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255])

    return b


def HD_transform(img, im_num=36, id_scale=1.5):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= im_num:
        scale += 1
    scale -= 1

    scale = min(np.ceil(width * id_scale / 560), scale)
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(img, [new_h, new_w],)
    img = padding_560(img)
    width, height = img.size
    assert width * height <= im_num * 560 * 560
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


meta_instruction = """You are an AI assistant whose name is InternLM (书生·浦语).\n" + "- InternLM (书生·浦语) \
is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室).
It is designed to be helpful, honest, and harmless.\n"+"- InternLM (书生·浦语) \
can understand and communicate fluently in the language chosen by the user such as English and 中文."""


def model_gen(model, text, images, need_bos=True, padding=False, beams=3, max_token=500):
    embeds = []
    im_mask = []

    im_idx = 0
    sub_q = text.split('<IM_POS>')
    add_im = len(sub_q) - 1
    for subtext in sub_q:
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(
                subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).to(model.device))
            need_bos = False

        if im_idx < len(images) and add_im:
            try:
                image = Image.open(images[im_idx]).convert('RGB')
            except:
                image = images[im_idx].convert('RGB')
            if len(images) > 1:
                image = HD_transform(image, im_num=model.hd_num // len(images), id_scale=model.id_scale)
            else:
                image = HD_transform(
                    image, im_num=model.hd_num, id_scale=model.id_scale)
            image = model.vis_processor(image).unsqueeze(0).to(model.device)
            image_embeds = model.encode_img(image)
            im_idx += 1
            add_im -= 1
            embeds.append(image_embeds)
            im_mask.append(torch.ones(
                image_embeds.shape[:2], dtype=torch.long).to(model.device))

    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                             temperature=1.0, max_new_tokens=max_token, num_beams=beams,
                             do_sample=False, repetition_penalty=1.0)

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(
        output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
    return output_text


class XComposer2d5(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='internlm/internlm-xcomposer2d5-7b', id_scale=1.5, beam=3, **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.id_scale = id_scale
        self.beam = beam

        model = AutoModel.from_pretrained(
            self.model_path, device_map='cpu', trust_remote_code=True, local_files_only=True).cuda().eval()
        model.half()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        model.tokenizer = tokenizer
        self.model = model
        self.device = self.model.model.tok_embeddings.weight.device
        self.model.hd_num = 36
        self.model.id_scale = self.id_scale

    def message_to_promptimg(self, message, dataset=None):
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value']
                               for x in message if x['type'] == 'text'])
            image = None
        else:
            image = [x['value'] for x in message if x['type'] == 'image']
            if len(image) == 1:
                prompt = ''.join([x['value']
                                 for x in message if x['type'] == 'text'])
                im_prompt = '<IM_POS>'
                prompt = prompt.replace('<image 1>', '')
                prompt = im_prompt + prompt
            else:
                prompt = ''
                im_prompt = [
                    f'Image{im_idx+1}: <IM_POS>;' for im_idx in range(len(image))]
                add_im = len(im_prompt)
                im_idx = 0
                for x in message:
                    if x['type'] == 'text':
                        prompt += x['value']
                        if add_im > im_idx:
                            prompt += f'Image{im_idx + 1}'
                            im_idx += 1
                im_prompt = ' '.join(im_prompt)
                for i in range(len(image)):
                    prompt = prompt.replace(f'<image {i+1}>', f'Image{i+1} ')
                if listinstr(['mmlongbench', 'dude', 'slidevqa'], dataset.lower()):     # fix bug for multi-image prompt
                    prompt = '[UNUSED_TOKEN_146]user\n' + im_prompt + re.sub(
                        re.escape('[UNUSED_TOKEN_146]user\n'), '', prompt
                    )
                    prompt = re.sub('Image1$', '', prompt)
        return prompt, image

    def generate_mme(self, image_path, text):
        text = text.split('Please answer')[0].strip()
        text = f'{text} Answer this question briefly'
        text = f'[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'

        return model_gen(self.model, text, image_path, need_bos=True, padding=True, beams=self.beam)

    def generate_multichoice(self, image_path, text, dataset):
        out = model_gen(self.model, text, image_path,
                        need_bos=True, padding=False, beams=self.beam, max_token=5)
        if 'mmmu' in dataset.lower():
            return out
        res = pattern.findall(out)
        if len(res) == 0:
            print('Error:', out)
            res = 'Z'
        return res[0]

    def generate_vqa(self, image_path, text):
        out = model_gen(self.model, text, image_path, beams=self.beam,
                        need_bos=True, max_token=100)
        return out

    def generate_vanilla(self, image_path, text):
        out = model_gen(self.model, text, image_path, beams=self.beam,
                        need_bos=True, max_token=500)
        return out

    def generate_brief(self, image_path, text):
        text = '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}\
               [UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'.format(text)
        out = model_gen(self.model, text, image_path, beams=self.beam,
                        need_bos=True, max_token=10)
        return out

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        with torch.cuda.amp.autocast():
            if dataset is None:
                return self.generate_vanilla(image_path, prompt)
            assert isinstance(dataset, str)
            if dataset == 'MME':
                return self.generate_mme(image_path, prompt)
            elif listinstr(['hallu', 'pope'], dataset.lower()):
                return self.generate_brief(image_path, prompt)
            elif listinstr(['llava', 'mmvet'], dataset.lower()):
                return self.generate_vanilla(image_path, prompt)
            elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
                return self.generate_multichoice(image_path, prompt, dataset)
            elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
                return self.generate_vqa(image_path, prompt)
            else:
                return self.generate_vanilla(image_path, prompt)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'VQA':
            return True
        return False

    def build_mcqa(self, line):
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        img_prompt = '[UNUSED_TOKEN_146]user\n'
        if len(options):
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item} '
            options_prompt = options_prompt.strip()
            hint = line['hint'] if (
                'hint' in line and not pd.isna(line['hint'])) else None

            context = 'N/A' if hint is None else hint
            mid_prompt = 'Question: ' + question + '\nContext: ' + \
                context + '\nOptions: ' + options_prompt
            ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'
            prompt = img_prompt + mid_prompt + ans_prompt
        else:
            mid_prompt = f'Answer the question using a single word or phrase.{question}'
            ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
            prompt = img_prompt + mid_prompt + ans_prompt

        return prompt

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_mcqa(line)
        elif DATASET_TYPE(dataset) == 'VQA':
            if 'mathvista' in dataset.lower():
                q = line['question']
                prompt = f'[UNUSED_TOKEN_146]user\n{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
            elif listinstr(['llava', 'mmvet'], dataset.lower()):
                q = line['question']
                prompt = '[UNUSED_TOKEN_146]system\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]user\n{}\
                         Answer this question in detail.[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]\
                         assistant\n'.format(meta_instruction, q)
            elif listinstr(['mmlongbench_doc', 'dude', 'slidevqa'], dataset.lower()):
                q = line['question']
                prompt = f'[UNUSED_TOKEN_146]user\n{q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
            else:
                q = line['question']
                prompt = f'[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.\
                          {q}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        ret = [dict(type='text', value=prompt)]
        ret.extend([dict(type='image', value=s) for s in tgt_path])
        return ret
