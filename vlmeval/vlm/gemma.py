from PIL import Image
import torch

from .base import BaseModel
from ..smp import *


class PaliGemma(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='google/paligemma-3b-mix-448', **kwargs):
        try:
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
            revision='bfloat16',
        ).eval()
        self.model = model.cuda()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')

        model_inputs = self.processor(
            text=prompt, images=image, return_tensors='pt'
        ).to('cuda')
        input_len = model_inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=512, do_sample=False
            )
            generation = generation[0][input_len:]
            res = self.processor.decode(generation, skip_special_tokens=True)
        return res


class Gemma3(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='google/gemma-3-4b-it', **kwargs):
        logging.info(
            "Please install transformers via \n"
            "pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"
        )
        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
            import torch
        except Exception as e:
            logging.critical('Please install torch and transformers')
            raise e

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, device_map="cuda", attn_implementation="flash_attention_2"
        ).eval()

        self.device = self.model.device
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.system_prompt = kwargs.pop('system_prompt', 'You are a helpful assistant. ')

        default_kwargs = {
            'do_sample': False,
            'max_new_tokens': 2048
        }
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    def message2pipeline(self, message):
        ret = []
        if hasattr(self, 'system_prompt') and self.system_prompt is not None:
            ret = [
                dict(role='system', content=[dict(type='text', text=self.system_prompt)])
            ]
        content = []
        for m in message:
            if m['type'] == 'text':
                content.append(dict(type='text', text=m['value']))
            elif m['type'] == 'image':
                content.append(dict(type='image', url=m['value']))
        ret.append(dict(role='user', content=content))
        return ret

    def generate_inner(self, message, dataset=None):
        messages = self.message2pipeline(message)
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
