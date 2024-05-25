from PIL import Image
import torch

from .base import BaseModel
from ..smp import *


class Phi3Vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='microsoft/Phi-3-vision-128k-instruct', **kwargs):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except:
            warnings.warn('Please install the latest version transformers.')
            sys.exit(-1)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='cuda', trust_remote_code=True, torch_dtype='auto').eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = model
        self.processor = processor
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        image = Image.open(image_path).convert('RGB')
        messages = [
            {'role': 'user', 'content': f'<|image_1|>\n{prompt}'}
        ]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors='pt').to('cuda')

        generation_args = {
            'max_new_tokens': 500,
            'temperature': 0.0,
            'do_sample': False,
        }
        generation_args.update(self.kwargs)

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response
