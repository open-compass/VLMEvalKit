from PIL import Image
import torch

from .base import BaseModel
from ..smp import *


class FastVLM(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='apple/FastVLM-0.5B', **kwargs):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        messages = [
            {"role": "user", "content": f"<image>\n{prompt}"}
        ]

        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        pre, post = rendered.split("<image>", 1)

        pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids

        # Splice in the image token at placeholder position
        img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)

        # Preprocess image via model's vision processor
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.model.get_vision_tower().image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"]
        pixel_values = pixel_values.to(self.model.device, dtype=self.model.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=512,
                do_sample=False,
                **self.kwargs
            )

        # Decode generated text excluding input tokens
        input_len = input_ids.shape[-1]
        generated_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response
