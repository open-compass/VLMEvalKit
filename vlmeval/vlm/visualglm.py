import torch
from PIL import Image
from abc import abstractproperty
import os.path as osp
import os 
from vlmeval.smp import *


class VisualGLM:
    def __init__(self):
        self.model_paths = [
            '/mnt/lustre/duanhaodong/petrel_share/visualglm-6b'
        ]
        model_path = "THUDM/visualglm-6b"
        for m in self.model_paths:
            if osp.exists(m):
                model_path = m
                
        from transformers import AutoModel
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model =  AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model = model

    def generate(self, image_path, prompt):
        
        output, _ = self.model.chat(
            image_path = image_path,
            tokenizer = self.tokenizer,
            query = prompt,
            history = []
        )
        return output