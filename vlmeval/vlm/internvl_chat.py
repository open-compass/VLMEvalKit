import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, CLIPImageProcessor
import warnings
import os.path as osp
from vlmeval.smp import isimg
import re
from PIL import Image
    
class InternVLChat:

    INSTALL_REQ = False

    def __init__(self, model_path='OpenGVLab/InternVL-Chat-Chinese-V1-1', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        device = torch.cuda.current_device()
        self.device = device
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).eval().to(device)
        self.image_size = self.model.config.vision_config.image_size
        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
    
    def generate(self, image_path, prompt, dataset=None):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )
        response = self.model.chat(self.tokenizer, pixel_values=pixel_values,
                                   question=prompt, generation_config=generation_config)
        return response