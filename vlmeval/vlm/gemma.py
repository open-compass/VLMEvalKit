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
            from transformers import pipeline 
            import torch
        except Exception as e:
            logging.critical('Please install torch and transformers')
            raise e

        self.model = pipeline('image-text-to-text', model=model_path, device="cuda", torch_dtype=torch.bfloat16)
        self.system_prompt = kwargs.pop('system_prompt', 'You are a helpful assistant. ')
        default_kwargs = {
            'temperature': 0, 
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
        pipeline_input = self.message2pipeline(message)
        output = self.model(text=pipeline_input, **self.kwargs)
        if os.environ.get('VERBOSE', 0):
            logging.debug(output)
        return output[0]['generated_text'][-1]['content']
