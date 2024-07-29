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
        except:
            warnings.warn('Please install the latest version transformers.')
            sys.exit(-1)
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
