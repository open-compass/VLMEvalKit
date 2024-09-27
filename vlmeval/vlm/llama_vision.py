import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class llama_vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct', **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except:
            warnings.warn('Please install transformers>=4.45.0 before using llama_vision.')
            sys.exit(-1)

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        kwargs_default = dict(do_sample=False, max_new_tokens=512, temperature=0.0, top_p=None, num_beams=1)
        kwargs.update(kwargs_default)
        print(f'Following kwargs received: {kwargs}, will use as generation config. ')
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.model.device)
        if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'Y/N':
            self.kwargs['max_new_tokens'] = 128
        else:
            self.kwargs['max_new_tokens'] = 512
        output = self.model.generate(**inputs, **self.kwargs)
        return self.processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '')
