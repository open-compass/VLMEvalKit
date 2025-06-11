import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from io import BytesIO
import base64
from mimetypes import guess_type

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE, DATASET_MODALITY


class KimiVL(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="moonshotai/Kimi-VL-A3B-Thinking", **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def encode_image(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"
        image = Image.open(image_path)
        # Handle the alpha channel
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)

        encoded_image = self._encode_image(image, image_format)

        return encoded_image

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    @staticmethod
    def _rgba_to_rgb(image):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def message_to_promptimg(self, message, dataset=None):
        processed_message = []
        images = []
        for item in message:
            if item['type'] == 'text':
                processed_message.append({
                    "type": "text",
                    "text": f"{item['value']}"
                })
            elif item['type'] == 'image':
                image_path = item['value']
                encoded_image = self.encode_image(image_path)
                image = Image.open(BytesIO(base64.b64decode(encoded_image)))
                image.load()
                processed_message.append({
                    "type": "image",
                    "image": image_path,
                })
                images.append(image)
        return processed_message, images

    def generate_inner(self, message, dataset=None):
        prompt, images = self.message_to_promptimg(message, dataset=dataset)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = self.processor(
            images=images, text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
