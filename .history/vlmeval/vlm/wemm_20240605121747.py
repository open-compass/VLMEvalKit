import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..smp import *
from .base import BaseModel
from ..utils import DATASET_TYPE, CustomPrompt
from transformers import AutoModel,AutoModelForCausalLM
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig


class WeMM(BaseModel):
    def __init__(self, model_path="feipengma/WeMM", **kwargs):
        self.wemm = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.wemm.cuda()
        self.wemm.eval()
        torch.cuda.empty_cache()
        
    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
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
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])
        return message
        # return {'image': tgt_path, 'text': prompt}
    
    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        
        if(dataset == 'HallusionBench'):
            prompt = prompt + " Please answer yes or no. Answer the question using a single word or phrase."
        
        gen_config = None
        pred = self.wemm.mm_generate(image_path, prompt, gen_config)
        print(f"prompt is {prompt}\n answer is {pred}")
        
        return pred
