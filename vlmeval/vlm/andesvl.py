from .base import BaseModel
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor


class AndesVL(BaseModel):
    def __init__(self, model_path, **kwargs):
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            .cuda()
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        kwargs_default = {"max_new_tokens": 64, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def generate_inner(self, message, dataset=None):
        """
        pil_rgb = Image.fromarray(rgb)
        conversations = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": { "url": pil_rgb }},
                {"type": "text", "text": current_prompt},
            ],
        }]
        """

        conversations = []
        for m in message:
            conv = {"role": "user", "content": []}
            if m["type"] == "text":
                conv['content'].append({"type": "text", "text": m["value"]})
            elif m["type"] == "image":
                pil_rgb = Image.open(m['value']).convert("RGB")
                conv['content'].append({"type": "image_url", "image_url": {"url": pil_rgb}})
            conversations.append(conv)
            
        response = self.model.chat(conversations,
                                   self.tokenizer,
                                   self.image_processor,
                                   generation_config=self.model.generation_config,
                                   max_new_tokens=64,
                                   do_sample=True,
                                   temperature=0.6,
        )

        return response
