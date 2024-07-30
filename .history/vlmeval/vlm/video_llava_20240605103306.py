import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...utils import DATASET_TYPE

class Video_LLaVA(BaseModel):
    def __init__(self, model_pth = None):
        super(Video_LLaVA, self).__init__(model_pth)
        self.model = VideoLLaVA()
        self.model.eval()
        self.model.to(self.device)