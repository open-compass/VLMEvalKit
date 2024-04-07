import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import copy as cp
from .base import BaseModel
from ..smp import isimg, listinstr
from ..utils import DATASET_TYPE


class QwenVL(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Qwen/Qwen-VL', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        default_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def adjust_kwargs(self, dataset):
        kwargs = cp.deepcopy(self.kwargs)
        if DATASET_TYPE(dataset) in ['multi-choice', 'Y/N']:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 100
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 10
        return kwargs

    def generate_inner(self, message, dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs
        prompt = ''
        for s in message:
            if s['type'] == 'image':
                prompt += f'<img>{s["value"]}</img>'
            elif s['type'] == 'text':
                prompt += s['value']
        if dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            prompt += ' Answer:'
        encoded = self.tokenizer([prompt], return_tensors='pt', padding='longest')
        input_ids = encoded.input_ids.to('cuda')
        attention_mask = encoded.attention_mask.to('cuda')

        pred = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs)
        answer = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        return answer


class QwenVLChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Qwen/Qwen-VL-Chat', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        torch.cuda.empty_cache()
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        vl_list = [{'image': s['value']} if s['type'] == 'image' else {'text': s['value']} for s in message]
        query = self.tokenizer.from_list_format(vl_list)
        response, _ = self.model.chat(self.tokenizer, query=query, history=None, **self.kwargs)
        return response
