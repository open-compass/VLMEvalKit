import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
from io import BytesIO
import base64
from mimetypes import guess_type


class llama4(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct", **kwargs):
        try:
            from transformers import AutoProcessor, Llama4ForConditionalGeneration
        except Exception as e:
            logging.critical('Please install transformers>=4.51.0 before using llama4.')
            raise e

        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = 10  # vLLM support max 10 images per prompt for Llama 4
        if self.use_vllm:
            from vllm import LLM, SamplingParams
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
            import os
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
            # export VLLM_WORKER_MULTIPROC_METHOD=spawn
        else:
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model_name = model_path

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA'], dataset):
            # For Certain dataset we use custom prompt
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        if listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options} Indicate the correct answer at the end.'
            )
            for i in range(len(tgt_path)):
                prompt = prompt.replace(f'<image {i + 1}>', '')
        elif listinstr(['MathVista'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            prompt = f'{question}'
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the question '
                f'with the text as seen exactly in the image. '
                f'For yes/no questions, just respond Yes or No. '
                f'If the answer is numeric, just respond with the number and nothing else. '
                f'If the answer has multiple words, just respond with the words and absolutely nothing else. '
                f'Never respond in a sentence or a phrase.\n Question: {question}'
            )
        else:
            raise NotImplementedError(f'Dataset {dataset}) not supported.')

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

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
        for item in message:
            if item['type'] == 'text':
                processed_message.append({
                    "type": "text",
                    "text": f"{item['value']}"
                })
            elif item['type'] == 'image':
                image_path = item['value']
                encoded_image = self.encode_image(image_path)
                processed_message.append({
                    "type": "image",
                    "url": f"{encoded_image}",
                })
        return processed_message

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
        max_new_tokens = 8192
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:]
        )[0]
        if generated_text.endswith("<|eot|>"):
            generated_text = generated_text[:-7]
        return generated_text

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
                    encoded_image = self.encode_image(image_path)
                    image = Image.open(BytesIO(base64.b64decode(encoded_image)))
                    image.load()
                    processed_message.append({
                        "type": "image",
                        "url": "",
                    })
                    images.append(image)
                    num_images += 1
        if num_images >= self.limit_mm_per_prompt:
            logging.warning(
                f"Number of images exceeds the limit of {self.limit_mm_per_prompt}."
                f"Only the first {self.limit_mm_per_prompt} images will be used."
            )
        return processed_message, images

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import LLM, SamplingParams
        prompt, images = self.message_to_promptimg_vllm(message, dataset=dataset)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=4096)
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

        if generated_text.endswith("<|eot|>"):
            generated_text = generated_text[:-7]  # 删除末尾的<|eot|>

        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
