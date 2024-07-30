import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...utils import DATASET_TYPE

class Video_LLaVA(BaseModel):
    def __init__(self, model_pth = 'checkpoints/Video-LLaVA-7B', model_base = 'None', **kwargs):
        try:
            from videollava.conversation import conv_templates, SeparatorStyle
            from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
            from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
            from videollava.model.builder import load_pretrained_model
        except:
            warnings.warn('Please install videollava before using Video-LLaVA')
            sys.exit(-1)
        model_name = get_model_name_from_path(model_pth)
        tokenizer, model, processor, context_len = load_pretrained_model(model_pth, model_base, model_name)
        model = model.cuda()

        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=1, use_cache=True
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
