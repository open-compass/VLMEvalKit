import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class molmo(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='allenai/Molmo-7B-D-0924', **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            import einops
        except Exception as e:
            logging.critical('Please install transformer and einops before using molmo.')
            raise e

        if '72b' not in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='cuda')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto')

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.kwargs = kwargs
        self.model_name = model_path

    def generate_inner(self, message, dataset=None):
        from transformers import GenerationConfig
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # process the image and text
        inputs = self.processor.process(
            images=[image],
            text=prompt
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        return generated_text
