import torch
from transformers import AutoModelForCausalLM

from .base import BaseModel
from ..dataset import DATASET_TYPE
from ..smp import *


class Ovis(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='AIDC-AI/Ovis1.5-Llama3-8B', **kwargs):
        assert model_path is not None
        # Recommend to install `transformers==4.43.2` and `torch==2.1.2`.
        self.model_path = model_path
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            multimodal_max_length=8192,
            trust_remote_code=True
        )
        self.model = self.model.eval().to(device=self.device)
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()
        self.image_placeholder = '<image>'
        self.gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )
        self.gen_kwargs.update(kwargs)

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            message = MMMUDataset.split_MMMU(message)

        return message

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question']
        if listinstr(['HallusionBench'], dataset):
            prompt += ' Please answer yes or no.'
        prompt += '\n请用单个词或短语回答问题。' if cn_string(
            prompt) else '\nAnswer the question using a single word or phrase.'
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None):
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
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def generate_inner(self, message, dataset=None):
        prompt, input_ids, attention_mask, pixel_values = self.prepare_inputs(message)
        output_ids = self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **self.gen_kwargs
        )
        response = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return response

    def prepare_inputs(self, message):
        # build query
        images = [x['value'] for x in message if x['type'] == 'image']
        texts = [x['value'] for x in message if x['type'] == 'text']
        if len(images) == 0:
            query = '\n'.join(texts)
        elif len(images) == 1 and len(texts) == 1:
            query = self.image_placeholder + '\n' + texts[0]
        else:  # interleave sample
            chunks = [x['value'] if x['type'] == 'text' else self.image_placeholder for x in message]
            query = '\n'.join(chunks)

        # format conversation
        prompt, input_ids = self.conversation_formatter.format_query(query)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)

        # preprocess images
        if len(images) == 0:
            pixel_values = [None]
        else:
            preprocessed_images = [self.visual_tokenizer.preprocess_image(Image.open(image)) for image in images]
            pixel_values = [torch.cat(preprocessed_images, dim=0).to(device=self.device, dtype=self.dtype)]

        return prompt, input_ids, attention_mask, pixel_values


class Ovis1_6(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='AIDC-AI/Ovis1.6-Gemma2-9B', **kwargs):
        assert model_path is not None
        # Recommend to install `python=3.10`, `transformers==4.44.2`, `torch==2.2.0`, and `numpy==1.24.3`
        self.model_path = model_path
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            multimodal_max_length=8192,
            trust_remote_code=True
        )
        self.model = self.model.eval().to(device=self.device)
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.max_partition = 9
        self.image_placeholder = '<image>'
        self.gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True
        )
        self.gen_kwargs.update(kwargs)

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) == 'Y/N' or DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line['question'] + '\nAnswer the question using a single word or phrase.'
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None):
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
            prompt += "\nAnswer with the option's letter from the given choices directly."

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            message = MMMUDataset.split_MMMU(message)

        return message

    def generate_inner(self, message, dataset=None):
        prompt, input_ids, attention_mask, pixel_values = self.prepare_inputs(message)
        output_ids = self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **self.gen_kwargs
        )
        response = self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response

    def prepare_inputs(self, message):
        # build query
        images = [x['value'] for x in message if x['type'] == 'image']
        texts = [x['value'] for x in message if x['type'] == 'text']
        if len(images) == 0:
            query = '\n'.join(texts)
        elif len(images) == 1 and len(texts) == 1:
            query = self.image_placeholder + '\n' + texts[0]
        else:  # interleaved sample
            chunks = [x['value'] if x['type'] == 'text' else self.image_placeholder for x in message]
            query = '\n'.join(chunks)

        # preprocess inputs
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [Image.open(image) for image in images], max_partition=self.max_partition
        )

        # move to self.device
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
        pixel_values = [
            pixel_values.to(device=self.device, dtype=self.dtype) if pixel_values is not None else None
        ]

        return prompt, input_ids, attention_mask, pixel_values


class Ovis1_6_Plus(Ovis1_6):
    # Recommend to install `python=3.10`, `transformers==4.46.2`, `torch==2.4.0`, and `numpy==1.25.0`

    def build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        import string
        import pandas as pd

        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above.'
        prompt = prompt.rstrip()
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset.startswith('MMMU_'):
            prompt = self.build_mmmu_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'Y/N':
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f'Invalid dataset type: {DATASET_TYPE(dataset)}')

        message = [dict(type='image', value=s) for s in tgt_path] + [dict(type='text', value=prompt)]

        return message
