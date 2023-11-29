import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os.path as osp

class QwenVL:

    INSTALL_REQ = False

    def __init__(self, name):
        self.name = name
        pths = {
            'qwen_base': [
                'Qwen/Qwen-VL'
            ],
            'qwen_chat': [
                'Qwen/Qwen-VL-Chat'
            ]
        }[self.name]
        pth = None 
        for p in pths:
            if osp.exists(p):
                pth = p
                break
            elif len(p.split('/')) == 2:
                pth = p
                break

        assert pth is not None
        self.tokenizer = AutoTokenizer.from_pretrained(pth, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(pth, device_map='cuda', trust_remote_code=True).eval()
        torch.cuda.empty_cache()
    
    def generate(self, image_path, prompt, dataset=None):
        vl_pair = [{'image': image_path}, {'text': prompt}]
        query = self.tokenizer.from_list_format(vl_pair)
        
        if self.name == 'qwen_base':
            inputs = self.tokenizer(query, return_tensors='pt')
            inputs = inputs.to(self.model.device)
            pred = self.model.generate(**inputs)
            response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
            response = response.split(prompt)[1].split('<|endoftext|>')[0]
        elif self.name == 'qwen_chat':
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        return response