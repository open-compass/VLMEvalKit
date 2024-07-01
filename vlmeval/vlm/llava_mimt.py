import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import re
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

# for LLaVa
def eval_model(query, image_files, model, tokenizer, image_processor, model_name, history, conv_mode=None, sep = ",", temperature=0, top_p=None, num_beams=1, max_new_tokens=4096):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if '<image-placeholder>' in query:
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "llava_v0"
    meta_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

    messages = history.copy()
    new_conversation = [
        {"from": "Human", "content": qs},
        {"from": "Assistant", "content": ""}
    ]
    messages.extend(new_conversation)
    prompt = ""
    for i, message in enumerate(messages):
        if i >= len(messages) - 2:  # 检查是否是最后一个message
            if message["from"] == "Human":
                prompt += "###Human:"
            elif message["from"] == "Assistant":
                prompt += "###Assistant:"
            prompt += message["content"]
            prompt += "\n"
        else:
            prompt += message["content"]
            prompt += "\n"
    prompt = meta_prompt + prompt


    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # print(images_tensor.shape)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    new_conversation = [
        {"from": "Human", "content": qs},
        {"from": "Assistant", "content": outputs}
    ]
    history.extend(new_conversation)

    return outputs, history

# for LLaVa_Next
def gen_answer(model, processor, query, img_list, history):

    images = [Image.open(img).resize((336,336), Image.Resampling.LANCZOS) for img in img_list]
    new_conversation = "[INST]"+ query + "[/INST]"
    if len(history)!=0:
        prompt = history[0] + new_conversation
    else:
        prompt = new_conversation
    
    inputs = processor(text=prompt, images=images, padding=True, return_tensors="pt").to(model.device)

    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=2048)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    index = response[0].rfind('[/INST]')
    if index != -1:
        response = response[0][index + len('[/INST]'):].strip()
    else:
        response = ""

    
    if len(history)!=0:
        hisroty = [history[0] + new_conversation + response]
    else:
        hisroty = [new_conversation + response]

    return response, hisroty


class LLaVA_mimt:

    INSTALL_REQ = True
    INTERLEAVE = True

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

        self.model_name = model_name

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

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')


    def generate(self, message, dataset=None):

        img_paths = []
        questions = []
        for item in message:
            if item['type'] == 'image':
                img_paths.append(item['value'])
            elif item['type'] == 'text':
                questions.append(item['value'])
        questions = eval(questions[0])

        responses = []
        pics_number = 0
        history = []
        for index, q in enumerate(questions):
            if "<ImageHere>" in q:
                tag_number = q.count('<ImageHere>')
                pics_number += tag_number
                images = img_paths[:pics_number]
            else:
                images = img_paths[:pics_number]
            logging.info(pics_number)
            q = q.replace("<ImageHere>", "<image-placeholder>")
            q = q.lstrip()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    response, history = eval_model(query=q, image_files=images, model=self.model, tokenizer=self.tokenizer, image_processor=self.image_processor, model_name=self.model_name, history=history)
            responses.append(response)

        return responses



class LLaVA_Next_mimt:

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_pth='llava-hf/llava-v1.6-vicuna-7b-hf', **kwargs):
        import transformers
        # assert version_cmp(transformers.__version__, '4.39.0', 'ge')
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

    def generate(self, message, dataset=None):

        img_paths = []
        questions = []
        for item in message:
            if item['type'] == 'image':
                img_paths.append(item['value'])
            elif item['type'] == 'text':
                questions.append(item['value'])
        questions = eval(questions[0])

        responses = []
        pics_number = 0
        history = []
        for index, q in enumerate(questions):
            if "<ImageHere>" in q:
                tag_number = q.count('<ImageHere>')
                pics_number += tag_number
                images = img_paths[:pics_number]
            else:
                images = img_paths[:pics_number]
            logging.info(pics_number)
            q = q.replace("<ImageHere>", "<image>\n")
            q = q.lstrip()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    response, history = gen_answer(model=self.model, processor=self.processor, query=q, img_list=images, history=history)
            responses.append(response)
        return responses



