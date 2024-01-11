import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os.path as osp
from vlmeval.smp import isimg
from ..utils import CustomPrompt
import re

class Monkey:

    INSTALL_REQ = False

    def __init__(self, model_path='echo840/Monkey', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', trust_remote_code=True)
        model.eval()
        self.model = model.cuda()
        self.kwargs = kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()
    
    def generate(self, image_path, prompt, dataset=None):
        cur_prompt = f'<img>{image_path}</img>\n{prompt} Answer:'
        input_ids = self.tokenizer(cur_prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        output_ids = self.model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                min_new_tokens=1,
                length_penalty=3,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,
            )
        response = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        
        return response
