from PIL import Image
import torch

from .base import BaseModel
from ..smp import *


class LLaVA_HF(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='llava-hf/llava-1.5-7b-hf', **kwargs):
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except:
            warnings.warn('Please install the latest version transformers.')
            sys.exit(-1)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
            low_cpu_mem_usage=True,
        ).eval()
        self.model = model.cuda()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None, transform=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = Image.open(image_path).convert('RGB')
        if transform:
            augmented_image_np = transform(image=np.array(image))['image']
            image = Image.fromarray(augmented_image_np)
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