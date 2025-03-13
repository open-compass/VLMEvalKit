from PIL import Image
import torch

from .base import BaseModel
from ..smp import *


class Phi4Multimodal(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='microsoft/Phi-4-multimodal-instruct', **kwargs):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='cuda', trust_remote_code=True, torch_dtype='auto',attn_implementation='flash_attention_2').eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(model_path)
        
        self.model = model
        self.processor = processor
        # self.kwargs = kwargs
        self.generation_config=generation_config

    def generate_inner(self, message, dataset=None):
        user_question = '\n'.join([msg['value'] for msg in message if msg['type'] == 'text'])
        images = [Image.open(msg['value']).convert('RGB') for msg in message if msg['type'] == 'image']


        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        prompt = f'{user_prompt}<|image_1|>{user_question}{prompt_suffix}{assistant_prompt}'
        inputs = self.processor(text=prompt, images=images[0], return_tensors='pt').to('cuda') 

        # Generate response
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
