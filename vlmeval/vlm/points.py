from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import CLIPImageProcessor
from PIL import Image
import torch
from .base import BaseModel
from ..dataset import DATASET_TYPE
from ..smp import cn_string, listinstr
import pandas as pd
import string
from typing import List


class POINTS(BaseModel):
    """Official implementation of POINTS: Improving Your Vision-language Model with Affordable Strategies # noqa

    Paper link: https://arxiv.org/abs/2409.04828
    POINTS is a vision-language model developed by researchers from WeChat AI. The existing model is the first version
    of our series of multimodal models, called WePOINTS.

    Args:
        model_name_or_path (str): The path or the name (the unique huggingface id) of the model.
    """

    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          trust_remote_code=True,  # noqa
                                                          device_map='cuda'
                                                          ).to(torch.bfloat16)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_name_or_path)

    def use_custom_prompt(self, dataset: str) -> bool:
        """Whether to use custom prompt for the dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt for the dataset.
        """
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
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

    def generate_inner(self, message: List[dict], dataset: str = None) -> str:
        """Generate response for the given message.

        Args:
            message (List[dict]): A list of elements constructed for
                current line.
            dataset (str): The name of the dataset.

        Returns:
            str: The generated response.
        """
        prompt, image_path = self.message_to_promptimg(message)
        catty = True  # whether to use catty
        if dataset == 'HallusionBench':
            prompt = prompt + \
                ' Please answer yes or no. Answer the question using a single word or phrase.'  # noqa
        if dataset == 'MMVet':
            prompt = prompt + ' Answer this question in detail.'
            catty = False

        if listinstr(['MMBench', 'OCRBench'], dataset):
            max_splits = 12
        else:
            max_splits = 8
        image = Image.open(image_path).convert('RGB')
        generation_config = {
            'max_new_tokens': 1024,
            'temperature': 0.0,
            'top_p': 0.0,
            'num_beams': 1,
        }
        response = self.model.chat(image,
                                   prompt,
                                   self.tokenizer,
                                   self.image_processor,
                                   catty,
                                   generation_config,
                                   max_splits)
        return response
