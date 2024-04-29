import torch
import sys
import os.path as osp
import warnings
from .base import BaseModel
from transformers import StoppingCriteriaList
from omegaconf import OmegaConf
from PIL import Image


class Vxverse(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, mode='', root=None, temperature=1, max_out_len=512):

        if root is None:
            warnings.warn('Please set root to the directory of vxverse.')

        if mode == 'vxverse':
            cfg = 'vxverse_13b_eval.yaml'
        else:
            raise NotImplementedError

        self.mode = mode
        self.temperature = temperature
        self.max_out_len = max_out_len
        self.root = root
        this_dir = osp.dirname(__file__)

        self.cfg = osp.join(this_dir, 'misc', cfg)
        sys.path.append(self.root)

        from vxverse.common.registry import registry
        from vxverse.conversation.conversation import (
            StoppingCriteriaSub,
            CONV_VISION_XVERSE,
        )

        device = torch.cuda.current_device()
        self.device = device

        cfg_path = self.cfg
        cfg = OmegaConf.load(cfg_path)

        model_cfg = cfg.model
        model_cls = registry.get_model_class(model_cfg.arch)
        model = model_cls.from_config(model_cfg)
        model = model.to(device)
        model.eval()
        vis_processor_cfg = cfg.datasets.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
        self.model = model
        self.vis_processor = vis_processor
        self.vis_processor_cfg = vis_processor_cfg

        self.CONV_VISION = CONV_VISION_XVERSE
        self.CONV_VISION.system = ''
        stop_words_ids = [[835], [2277, 29937]]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)

        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)

        if self.vis_processor_cfg.name == 'hd_image_train':
            patches_per_image = [[image.shape[0]]]
            image = [image]
        else:
            patches_per_image = None
            image = image.unsqueeze(0)

        chat_state = self.CONV_VISION.copy()
        texts = self.prepare_texts([prompt], chat_state)
        texts = [text.lstrip() for text in texts]
        answers = self.model.generate(
            image,
            texts,
            max_new_tokens=self.max_out_len,
            patches_per_images=patches_per_image,
            do_sample=False,
            stop_words_ids=self.stop_words_ids,
        )
        return answers[0]

    def prepare_texts(self, texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        [
            conv.append_message(conv.roles[0], '<ImageHere>\n{}'.format(text))
            for conv, text in zip(convs, texts)
        ]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]
        return texts
