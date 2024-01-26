import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os.path as osp
from vlmeval.smp import isimg, listinstr
import re

class QwenVL:

    INSTALL_REQ = False

    def __init__(self, model_path='Qwen/Qwen-VL', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        default_kwargs = {
            'do_sample':False, 'num_beams':1, 'max_new_tokens': 100, 
            'min_new_tokens': 1, 'length_penalty': 1, 'num_return_sequences': 1,
            'output_hidden_states': True, 'use_cache': True,
            'pad_token_id':self.tokenizer.eod_id, 'eos_token_id': self.tokenizer.eod_id
        }
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()
    
    def generate(self, image_path, prompt, dataset=None):
        vl_pair = [{'image': image_path}, {'text': prompt+ ' Answer: '}]
        query = self.tokenizer.from_list_format(vl_pair)
        if listinstr(['TextVQA'], dataset):
            self.kwargs['max_new_tokens'] = 10
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(prompt)[1].split('<|endoftext|>')[0]
        return response
    
    def multi_generate(self, image_paths, prompt, dataset=None):
        vl_list = [{'image': img} for img in image_paths] + [{'text': prompt+ ' Answer: '}]
        query = self.tokenizer.from_list_format(vl_list)
        if listinstr(['TextVQA'], dataset):
            self.kwargs['max_new_tokens'] = 10
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(prompt)[1].split('<|endoftext|>')[0]
        return response
    
    def interleave_generate(self, ti_list, dataset=None):
        vl_list = [{'image': s} if isimg(s) else {'text': s+ ' Answer: '} for s in ti_list]
        query = self.tokenizer.from_list_format(vl_list)
        if listinstr(['TextVQA'], dataset):
            self.kwargs['max_new_tokens'] = 10
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, **self.kwargs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response.split(query)[1].split('<|endoftext|>')[0]
        return response
    
class QwenVLChat:

    INSTALL_REQ = False

    def __init__(self, model_path='Qwen/Qwen-VL-Chat', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        torch.cuda.empty_cache()
        default_kwargs = {
            'do_sample':False, 'num_beams':1, 'max_new_tokens': 100, 
            'min_new_tokens': 1, 'length_penalty': 1, 'num_return_sequences': 1,
            'output_hidden_states': True, 'use_cache': True,
            'pad_token_id':self.tokenizer.eod_id, 'eos_token_id': self.tokenizer.eod_id
        }
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
    
    def generate(self, image_path, prompt, dataset=None):
        vl_pair = [{'image': image_path}, {'text': prompt + ' Answer: '}]
        query = self.tokenizer.from_list_format(vl_pair)
        if listinstr(['TextVQA'], dataset):
            self.kwargs['max_new_tokens'] = 10
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response
    
    def multi_generate(self, image_paths, prompt, dataset=None):
        vl_list = [{'image': img} for img in image_paths] + [{'text': prompt+ ' Answer: '}]
        query = self.tokenizer.from_list_format(vl_list)    
        if listinstr(['TextVQA'], dataset):
            self.kwargs['max_new_tokens'] = 10
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response
    
    def interleave_generate(self, ti_list, dataset=None):
        vl_list = [{'image': s} if isimg(s) else {'text': s+ ' Answer: '} for s in ti_list]
        query = self.tokenizer.from_list_format(vl_list)
        if listinstr(['TextVQA'], dataset):
            self.kwargs['max_new_tokens'] = 10
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response