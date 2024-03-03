import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
import copy as cp
import os.path as osp
import torch.nn as nn
import torch
from vlmeval.smp import *

def get_gpu_num(model_name):
    model_name = model_name.lower()
    kws = {
        8: ['65b', '70b'],
        4: ['30b', '33b', '35b', '40b'],
        2: ['13b', '14b', '20b'],
        1: ['6b', '7b', 'moss'],
    }
    for k in [8, 4, 2, 1]:
        for keyword in kws[k]:
            if keyword in model_name:
                return k
    return 8

model_map = {
    'chatglm3-6b': 'THUDM/chatglm3-6b',
    'internlm2-7b': 'internlm/internlm2-chat-7b',
    'internlm2-20b': 'internlm/internlm2-chat-20b',
    'qwen-7b-v1.5': 'Qwen/Qwen1.5-7B-Chat',
    'qwen-14b-v1.5': 'Qwen/Qwen1.5-14B-Chat',
    'qwen-72b-v1.5': 'Qwen/Qwen1.5-72B-Chat', 
    'vicuna-13b-v1.5':'lmsys/vicuna-13b-v1.5',
    'vicuna-7b-v1.5':'lmsys/vicuna-7b-v1.5', 
    'yi-6b': '01-ai/Yi-6B-Chat', 
    'yi-34b': '01-ai/Yi-34B-Chat', 
    'mistral-7b': 'mistralai/Mistral-7B-v0.1'
}
Auto_model = [model_map['chatglm3-6b']]

class HFChatModel:

    def _get_context_length(self, model, model_path):
        # By default, we use model.config.seq_length
        model_path = model_path.lower()
        if listinstr(['chatglm'], model_path):
            context_window = model.config.seq_length
        else:
            context_window = model.config.max_position_embeddings
        return context_window
    
    def _get_context_length_robust(self, model, model_path):
        try:
            context_window = self._get_context_length(model, model_path)
            return context_window
        except:
            warnings.warn(
                "Failed to extract context_window information from config / generation_config. "
                "Please read the above code and check if the logic works for you model path"
            )
            raise NotImplementedError
        
    def __init__(self, 
                 model_path, 
                 system_prompt: str=None,
                 **model_kwargs):
        
        if 'vicuna' in model_path.lower():
            try:
                from fastchat.model import get_conversation_template
            except:
                warnings.warn("Please install fastchat first to use vicuna. ")

        self.explicit_device = model_kwargs.pop('device', None)

        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
        from transformers.generation import GenerationConfig
        
        if model_path in model_map:
            model_path = model_map[model_path]
        self.model_path = model_path
        if model_path in Auto_model:
            LoadModel = AutoModel
        else:
            LoadModel = AutoModelForCausalLM
        assert osp.exists(model_path) or len(model_path.split('/')) == 2

        device = self.explicit_device if self.explicit_device else "auto"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LoadModel.from_pretrained(model_path, trust_remote_code=True, device_map='cpu')
        model = model.eval()
        
        if device != 'cpu':
            model = model.to(f'cuda:{device}' if isinstance(device, int) else 'cuda')
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True, device_map=device)
        except:
            pass

        torch.cuda.empty_cache()
        self.model = model
        self.context_length = self._get_context_length_robust(model=model, model_path=model_path)
        self.answer_buffer = 128
        self.system_prompt = system_prompt
        default_kwargs = {
            'do_sample': False
        }
        default_kwargs.update(model_kwargs)

        for k, v in default_kwargs.items():
            warnings.warn(f'Following args will be used for inference, {k}: {v}. ')
        self.kwargs = default_kwargs
        torch.cuda.empty_cache()
        
    def generate(self, input):
        if 'vicuna' in self.model_path.lower():
            from fastchat.model import get_conversation_template
            conv = get_conversation_template('vicuna')
            conv.append_message(conv.roles[0], input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = self.tokenizer([prompt], return_tensors="pt")
            if torch.cuda.is_available():
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
            kwargs = dict(do_sample=True, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512)
            kwargs.update(self.kwargs)
            outputs = self.model.generate(**inputs, **kwargs)
            resp = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True, spaces_between_special_tokens=False)
        elif 'qwen' in self.model_path.lower():
            prompt = input
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").cuda()

            generated_ids = self.model.generate(model_inputs.input_ids, **self.kwargs)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            resp = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            resp, _ = self.model.chat(self.tokenizer, input, history=[], **self.kwargs)
        torch.cuda.empty_cache()
        return resp

    def length_ok(self, inputs):
        tot = len(self.tokenizer.encode(self.system_prompt)) if self.system_prompt is not None else 0
        for s in inputs:
            tot += len(self.tokenizer.encode(s))
        return tot + self.answer_buffer < self.context_length
    