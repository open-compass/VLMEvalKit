import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
from io import BytesIO
import base64
from mimetypes import guess_type
import os


class GLM4_1v(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="THUDM/GLM-4.1V-9B-Thinking", **kwargs):
        try:
            from transformers import AutoProcessor, Glm4vForConditionalGeneration
        except Exception as e:
            logging.critical('Please install transformers>=4.54.0 before using glm4_1v. \n'
                             'Run `pip install git+https://github.com/huggingface/transformers.git`')
            raise e
        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = 24
        if self.use_vllm:
            from vllm import LLM
            # Set tensor_parallel_size [8, 4, 2, 1] based on the number of available GPUs
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for Llama4 inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
            self.llm = LLM(
                model=model_path,
                max_num_seqs=4,
                max_model_len=32768,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )
        else:
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def message_to_promptimg(self, message, dataset=None):
        processed_message = []
        for item in message:
            if item['type'] == 'text':
                processed_message.append({
                    "type": "text",
                    "text": f"{item['value']}"
                })
            elif item['type'] == 'image':
                image_path = item['value']
                img = Image.open(image_path).convert('RGB')
                b64 = encode_image_to_base64(img)
                processed_message.append({
                    "type": "image",
                    "url": f"{b64}",
                })
        return processed_message

    def message_to_promptimg_vllm(self, message, dataset=None):
        processed_message = []
        images = []
        num_images = 0
        for item in message:
            if item['type'] == 'text':
                processed_message.append({
                    "type": "text",
                    "text": item['value']
                })
            elif item['type'] == 'image':
                if num_images < self.limit_mm_per_prompt:
                    image_path = item['value']
                    img = Image.open(image_path).convert('RGB')
                    img.load()
                    processed_message.append({
                        "type": "image",
                        "url": "",
                    })
                    images.append(img)
                    num_images += 1
        if num_images >= self.limit_mm_per_prompt:
            logging.warning(
                f"Number of images exceeds the limit of {self.limit_mm_per_prompt}."
                f"Only the first {self.limit_mm_per_prompt} images will be used."
            )
        return processed_message, images

    def generate_inner_transformers(self, message, dataset=None):
        prompt = self.message_to_promptimg(message, dataset=dataset)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=8192
        )
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
        return output_text

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        prompt, images = self.message_to_promptimg_vllm(message, dataset=dataset)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=8192)
        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": images
                },
            },
            sampling_params=sampling_params
        )

        for o in outputs:
            generated_text = o.outputs[0].text

        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
