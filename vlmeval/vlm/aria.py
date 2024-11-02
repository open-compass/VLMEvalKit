import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import warnings
import copy as cp
from PIL import Image
import pandas as pd
import string
from .base import BaseModel
from ..smp import isimg, listinstr, cn_string
from ..dataset import DATASET_TYPE, DATASET_MODALITY


class Aria(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='rhymes-ai/Aria', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.unk_token_id
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()
        default_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.unk_token_id,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()
    
    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
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

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def adjust_kwargs(self, dataset):
        kwargs = cp.deepcopy(self.kwargs)
        if DATASET_MODALITY(dataset) == "VIDEO":
            kwargs["max_image_size"] = 490
        else:
            kwargs["max_image_size"] = 980
            
        kwargs["split_image"] = False
        
        if listinstr(['MMMU', 'MMStar', 'Math'], dataset):
            # These datasets may lead the model to work as a CoT-alike behaviour.
            # Allow to output longer.
            kwargs['max_new_tokens'] = 1024
            return kwargs
        if DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            kwargs['max_new_tokens'] = 64
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 64
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 128
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 32
                
        if listinstr(['OCR', 'ChartQA', 'DocVQA', 'InfoVQA', 'TextVQA'], dataset):
            # OCR-related datasets that need to split image
            kwargs["split_image"] = True
            
        return kwargs

    def generate_inner(self, message, dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs
            
        max_image_size = kwargs.pop("max_image_size")
        split_image = kwargs.pop("split_image")
        
        prompt = '<|im_start|>user\n'
        images = []
        for s in message:
            if s['type'] == 'image':
                prompt += '<fim_prefix><|img|><fim_suffix>'
                images.append(s['value'])
            elif s['type'] == 'text':
                prompt += s['value']
        prompt += '<|im_end|>\n<|im_start|>assistant\n'
        if images:
            images = [Image.open(s).convert('RGB') for s in images]
            encoded = self.processor(
                text=prompt,
                images=images,
                return_tensors='pt',
                padding='longest',
                max_image_size=max_image_size,
                split_image=split_image,
            )
        else:
            encoded = self.processor(text=prompt, return_tensors='pt', padding='longest')
        encoded["pixel_values"] = encoded["pixel_values"].to(self.model.dtype)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        pred = self.model.generate(
            **encoded,
            **kwargs)
        answer = self.tokenizer.decode(pred[0][encoded['input_ids'].size(1):].cpu(), skip_special_tokens=True).strip()
        answer = answer.replace('<|im_end|>', '')
        return answer