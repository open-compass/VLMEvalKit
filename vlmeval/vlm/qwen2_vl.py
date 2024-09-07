import os
import warnings

import torch

from .base import BaseModel


class Qwen2VLChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int,
        max_pixels: int,
        **kwargs,
    ):
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        assert min_pixels is None or isinstance(min_pixels, int)
        assert max_pixels is None or isinstance(max_pixels, int)
        assert model_path is not None

        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
        ).eval()
        torch.cuda.empty_cache()
        self.kwargs = kwargs

        default_eval_kwargs = dict(
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            repetition_penalty=1.0,
        )
        for k, v in default_eval_kwargs.items():
            self.kwargs.setdefault(k, v)
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            warnings.warn("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise

        content = []
        for s in message:
            if s['type'] == 'image':
                image = str(s['value'])
                prefixes = ['http://', 'https://', 'file://', 'data:image']
                if any(image.startswith(prefix) for prefix in prefixes):
                    pass
                elif os.path.exists(image):
                    image = 'file://' + image
                else:
                    raise ValueError(f'Invalid image: {image}, {s}')

                item = {'type': 'image', 'image': image}
                min_pixels = s['min_pixels'] if 'min_pixels' in s else self.min_pixels
                if min_pixels is not None:
                    item['min_pixels'] = min_pixels
                max_pixels = s['max_pixels'] if 'max_pixels' in s else self.max_pixels
                if max_pixels is not None:
                    item['max_pixels'] = max_pixels
                content.append(item)
            elif s['type'] == 'text':
                content.append({'type': 'text', 'text': s['value']})
            else:
                raise ValueError(f'Invalid message type: {s}')

        message = [{'role': 'user', 'content': content}]
        # print(f"\033[31m{message}\033[0m")
        text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([message])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            **self.kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        # print(f"\033[32m{response}\033[0m")
        return response

    def use_custom_prompt(self, dataset: str):
        return dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}

    def build_prompt(self, line, dataset: str):
        assert dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}

        # copy from ImageMCQDataset.build_prompt.
        # i.e. for MMMU dataset, not use `split_MMMU(msgs)`.
        import string
        import pandas as pd
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
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
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
