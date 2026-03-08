import os
import os.path as osp
import warnings
from .base import BaseGenModel
from ..smp import splitlen, listinstr
from ..dataset import DATASET_TYPE
import sys

from typing import List, Tuple
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import to_tensor

import logging


class OmniGen2(BaseGenModel):
    INSTALL_REQ = True
    INTERLEAVE = False
    EXPERTISE = ['T2I', 'TI2I']

    def __init__(self, model_path='OmniGen2/OmniGen2', omnigen_code_root=None, **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2
        if omnigen_code_root is not None and omnigen_code_root not in sys.path:
            sys.path.append(omnigen_code_root)
        try:
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

        except Exception as err:
            logging.critical(
                'Please first install OmniGen2 from source codes in: https://github.com/VectorSpaceLab/OmniGen2')
            raise err

        import accelerate
        accelerator = accelerate.Accelerator()

        default_kwargs = dict(
            num_inference_steps=50,
            seed=0,
            height=1024,
            width=1024,
            max_sequence_length=1024,
            dtype='bf16',
            text_guidance_scale=5.0,
            image_guidance_scale=2.0,
            # num_images_per_prompt=1,
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

        weight_dtype = default_kwargs.pop('dtype', 'bf16')
        if weight_dtype == 'fp16':
            weight_dtype = torch.float16
        elif weight_dtype == 'bf16':
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32
        token = default_kwargs.pop('token', None)
        seed = default_kwargs.pop('seed', 0)

        pipeline = OmniGen2Pipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
            token=token,
        )

        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

        self.model = pipeline.to(accelerator.device, dtype=weight_dtype)
        self.generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    def generate_inner(self, message, dataset=None):
        prompt = [msg['value'] if msg['type'] == 'text' else "" for msg in message]
        prompt = '\n'.join(prompt)

        images = [Image.open(msg['value']) for msg in message if msg['type'] == 'image']

        results = self.model(
            prompt=prompt,
            input_images=images,
            generator=self.generator,
            output_type='pil',
            **self.kwargs
        )

        return results.images[0]

    def batch_generate_inner(self, message, dataset, num_generations):
        prompt = [msg['value'] if msg['type'] == 'text' else "" for msg in message]
        prompt = '\n'.join(prompt)

        images = [Image.open(msg['value']) for msg in message if msg['type'] == 'image']

        results = self.model(
            prompt=prompt,
            input_images=images,
            generator=self.generator,
            output_type='pil',
            num_images_per_prompt=num_generations,
            **self.kwargs
        )

        return results.images
