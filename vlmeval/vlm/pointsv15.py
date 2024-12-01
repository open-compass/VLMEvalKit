from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLImageProcessor
import torch
from .base import BaseModel
from ..dataset import DATASET_TYPE
from ..smp import cn_string, concat_images_vlmeval
import pandas as pd
import string
from typing import List


class POINTSV15(BaseModel):
    """Official implementation of POINTSv1.5

    This implementation is based on the official implementation of POINTSv1.5
    (https://github.com/WePOINTS/WePOINTS)

    Args:
        model_path (str): The path or the name (the unique huggingface id)
            of the model.
    """

    INTERLEAVE = True

    def __init__(self, model_path: str, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          trust_remote_code=True,  # noqa
                                                          device_map='cuda'
                                                          ).to(torch.bfloat16)
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(model_path) # noqa

    def use_custom_prompt(self, dataset: str) -> bool:
        """Whether to use custom prompt for the dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt for the dataset.
        """
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line: str, dataset: str) -> List[dict]:
        """Build prompt for multi-choice dataset.

        Args:
            line (str): one line of the dataset.
            dataset (str): The name of the dataset.

        Returns:
            List[dict]: A list of elements constructed for current line.
        """
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if (
            'hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else  # noqa
                "\nAnswer with the option\'s letter from the given choices directly."  # noqa
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(  # noqa
                prompt) else '\nAnswer the question directly.'
        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def set_image_processor(self, dataset: str) -> None:
        """Set the image processor for the dataset.

        Args:
            dataset (str): The name of the dataset.
        """
        if dataset in ['OCRBench']:
            self.image_processor.min_pixels = 280 * 280
        elif dataset in ['MMMU_DEV_VAL']:
            self.image_processor.min_pixels = 1280 * 28 * 28
            self.image_processor.max_pixels = 16384 * 28 * 28
        elif dataset in ['MathVista_MINI']:
            self.image_processor.min_pixels = 56 * 56
        elif dataset in ['MMVet', 'HallusionBench',
                         'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11']:
            self.image_processor.min_pixels = 1280 * 28 * 28
        else:
            self.image_processor.min_pixels = 840 * 840

    def message_to_promptimg(self, message, dataset=None):
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        else:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            images = [x['value'] for x in message if x['type'] == 'image']
            if 'BLINK' == dataset:
                image = concat_images_vlmeval(images, target_size=512)
                images = [image]
        return prompt, images

    def construct_messages(self, prompt: str,
                           image_paths: List[str]) -> List[dict]:
        """Construct messages for the given prompt and image paths.

        Args:
            prompt (str): The prompt for the generation.
            image_paths (List[str]): A list of image paths.

        Returns:
            List[dict]: A list of elements constructed for current line.
        """
        content = []
        for image_path in image_paths:
            content.append(
                dict(type='image', image=image_path)
            )
        content.append(
            dict(type='text', text=prompt)
        )
        messages = [
            {
                'role': 'user',
                'content': content
            }
        ]
        return messages

    def generate_inner(self, message: List[dict], dataset: str = None) -> str:
        """Generate response for the given message.

        Args:
            message (List[dict]): A list of elements constructed for
                current line.
            dataset (str): The name of the dataset.

        Returns:
            str: The generated response.
        """
        self.set_image_processor(dataset)
        prompt, image_paths = self.message_to_promptimg(message)
        if dataset == 'HallusionBench':
            prompt = prompt + \
                ' Please answer yes or no. Answer the question using a single word or phrase.'  # noqa
        elif dataset == 'MMVet':
            prompt = prompt + ' Answer this question in detail.'
        else:
            # use default setting
            pass
        messages = self.construct_messages(prompt, image_paths)

        generation_config = {
            'max_new_tokens': 1024,
            'temperature': 0.0,
            'top_p': 0.0,
            'num_beams': 1,
        }
        response = self.model.chat(messages,
                                   self.tokenizer,
                                   self.image_processor,
                                   generation_config)
        return response
