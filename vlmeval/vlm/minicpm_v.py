import os
import os.path as osp
import sys
from abc import abstractproperty

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ..smp import *
from ..utils import DATASET_TYPE, CustomPrompt


class MiniCPM_V:

    INSTALL_REQ = False

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, device_map='cpu')
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()

    def generate(self, image_path, prompt, dataset=None):
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': prompt}]
        if dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
            max_new_tokens = 10
        else:
            max_new_tokens = 1024
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=1
        )
        default_kwargs.update(self.kwargs)
        res, _, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )
        return res
