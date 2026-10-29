import itertools
import json
import os
import re
import string

import pandas as pd
import torch
from PIL import Image

from .base import BaseModel

DIRECT_ANSWER_PROMPT = '\nPlease answer directly with only the final answer, do not give any explanation.'  # noqa: E501
SINGLE_YES_NO_PROMPT = '\nAnswer with a single word: Yes or No.'
ONLY_YES_NO_PROMPT = '\nOnly output Yes or No.'

REFCOCO_OUTPUT_PROMPT = (
    'Return only a valid JSON array. Do not include markdown, code fences, comments, or any text outside the JSON.\n'  # noqa: E501
    'Each array item must be an object with:\n'
    '- image_id: the zero-based image index. For a single-image input, use 0. For multi-image inputs, use 0 for the first image, 1 for the second, and so on.\n'  # noqa: E501
    '- bbox_2d: [xmin, ymin, xmax, ymax] normalized integer coordinates in [0, 1000]\n'
    '- label: a concise label you choose for the predicted object or region\n\n'
    'Return one item per visible matching object or region. Return [] if none are visible.'
)

SCREENSPOT_OUTPUT_PROMPT = (
    'Inspect the screenshot carefully, especially small or unlabeled icons. {disambiguation}'  # noqa: E501
    'Return only one tight bounding box whose center is the exact point you would click. Use this JSON shape: '  # noqa: E501
    '[{{"image_id": 0, "bbox_2d": [xmin, ymin, xmax, ymax], "label": "target"}}]. Use integer coordinates from 0 to 1000. Do not include markdown or any other text.'  # noqa: E501
)

MME_CATEGORY_PROMPTS = {
    'ocr': '{question}',
    'artwork': f'{{question}}{SINGLE_YES_NO_PROMPT}',
    'celebrity': f'This is a face recognition question about the person shown. Decide whether the name in the question matches the image. {{question}}{SINGLE_YES_NO_PROMPT}',  # noqa: E501
    'color': f'Check the visible color of the relevant object in the image. {{question}}{DIRECT_ANSWER_PROMPT}',  # noqa: E501
    'count': f'{{question}}{ONLY_YES_NO_PROMPT}',
    'existence': '{question}',
    'landmark': f'{{question}}{DIRECT_ANSWER_PROMPT}',
    'position': f'{{question}}{ONLY_YES_NO_PROMPT}',
    'posters': f'Look carefully at the image. {{question}}{SINGLE_YES_NO_PROMPT}',
    'scene': f'Look carefully at the image. {{question}}{SINGLE_YES_NO_PROMPT}',
    'code_reasoning': f'{{question}}{SINGLE_YES_NO_PROMPT}',
    'commonsense_reasoning': f'Use the image and common sense. {{question}}{SINGLE_YES_NO_PROMPT}',  # noqa: E501
    'numerical_calculation': f'Calculate carefully from the visible information. {{question}}{SINGLE_YES_NO_PROMPT}',  # noqa: E501
    'text_translation': f'{{question}}{DIRECT_ANSWER_PROMPT}',
}

NO_INSTRUCTION_DATASETS = frozenset({
    'BLINK',
    'MM-IFEval',
    'MME',
    'MMVet',
    'MathVista_MINI',
    'MUIRBench',
    'POPE',
    'RefCOCO',
    'LogicVista',
    'MMMU_DEV_VAL',
    'MMMU_TEST',
    'SimpleVQA',
})

DATASET_INSTRUCTION_PROMPTS = {
    'HallusionBench': '\nPlease answer yes or no.',
    'OCRBench': '\nPlease answer concisely with short words or phrases when possible.',
}


def _options_from_line(line):
    return {
        label: str(line[label])
        for label in string.ascii_uppercase
        if label in line and not pd.isna(line[label])
    }


def _is_screenspot_v2(dataset):
    return dataset == 'ScreenSpot_v2' or (dataset or '').startswith('ScreenSpot_v2_')


def _screen_spot_response(response):
    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return response

    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        return response
    bbox = payload[0].get('bbox_2d')
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(type(value) is not int or not 0 <= value <= 1000 for value in bbox)
    ):
        return response
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        return response

    x = (bbox[0] + bbox[2]) / 2000
    y = (bbox[1] + bbox[3]) / 2000
    return f'pyautogui.click(x={x:.6f}, y={y:.6f})'


class LFM2VL(BaseModel):
    INTERLEAVE = True

    def __init__(self, model_path, use_custom_prompt=True, **kwargs):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._use_custom_prompt = use_custom_prompt
        self._max_new_tokens_explicit = 'max_new_tokens' in kwargs
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16,
            )
            .cuda()
            .eval()
        )

        kwargs_default = {'max_new_tokens': 8192, 'use_cache': True, 'do_sample': False}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def use_custom_prompt(self, dataset):
        return self._use_custom_prompt and (
            dataset in {'MME', 'MathVista_MINI', 'MUIRBench', 'RefCOCO'}
            or _is_screenspot_v2(dataset)
        )

    def build_prompt(self, line, dataset):
        assert self.use_custom_prompt(dataset)
        paths = self.dump_image(line, dataset)
        sub_dataset = dataset
        if dataset == 'ScreenSpot_v2':
            sub_dataset = str(line['SUB_DATASET'])
        if dataset == 'MME':
            return self._build_mme_prompt(line, paths)
        if dataset == 'MathVista_MINI':
            return self._build_mathvista_prompt(line, paths)
        if dataset == 'MUIRBench':
            return self._build_muir_prompt(line, paths)
        if dataset == 'RefCOCO':
            return self._build_refcoco_prompt(line, paths)
        return self._build_screenspot_prompt(line, paths, sub_dataset)

    @staticmethod
    def _build_mme_prompt(line, paths):
        category = str(line['category']).lower()
        prompt = MME_CATEGORY_PROMPTS[category].format(question=line['question'])
        return [dict(type='image', value=path) for path in paths] + [
            dict(type='text', value=prompt)
        ]

    @staticmethod
    def _build_mathvista_prompt(line, paths):
        prompt = ''
        if 'hint' in line and not pd.isna(line['hint']):
            prompt += f'Hint: {line["hint"]}\n'
        prompt += f'Question: {line["question"]}'
        options = _options_from_line(line)
        if options:
            prompt += '\nChoices:'
            prompt += ''.join(f'\n({label}) {text}' for label, text in options.items())
        prompt += '\nPlease reason step by step, and put your final answer within \\boxed{}.'
        return [dict(type='image', value=path) for path in paths] + [
            dict(type='text', value=prompt.strip())
        ]

    @staticmethod
    def _build_muir_prompt(line, paths):
        prompt_items = []
        for index, path in enumerate(paths, start=1):
            prompt_items.extend([
                dict(type='text', value=f'Image-{index}: '),
                dict(type='image', value=path),
                dict(type='text', value='\n'),
            ])

        prompt = ''
        if 'hint' in line and not pd.isna(line['hint']):
            prompt += str(line['hint']) + '\n'
        prompt += str(line['question'])
        prompt += '\n'
        for label, option in _options_from_line(line).items():
            prompt += f'{label}. {option}\n'
        prompt += (
            'Reason carefully about the images and the choices. '
            'End with "Final answer: <option letter>".'
        )
        image_numbers = itertools.count(1)
        prompt = re.sub(r'<image>', lambda _: f'<Image-{next(image_numbers)}>', prompt)
        prompt_items.append(dict(type='text', value=prompt))
        return prompt_items

    @staticmethod
    def _build_refcoco_prompt(line, paths):
        prompt = (
            'Locate the object described by this referring expression:\n'
            f'{str(line["question"]).strip()}\n\n{REFCOCO_OUTPUT_PROMPT}'
        )
        return [dict(type='image', value=path) for path in paths] + [
            dict(type='text', value=prompt)
        ]

    @staticmethod
    def _build_screenspot_prompt(line, paths, dataset):
        disambiguation = ''
        if dataset in {'ScreenSpot_v2_Desktop', 'ScreenSpot_v2_Mobile'}:
            disambiguation = "Reason silently about the target's visual appearance and distinguish it from similar nearby controls. Do not output the reasoning. "  # noqa: E501
        prompt = (
            'Locate the clickable UI element described by this instruction:\n'
            f'{str(line["question"]).strip()}\n\n'
            f'{SCREENSPOT_OUTPUT_PROMPT.format(disambiguation=disambiguation)}'
        )
        return [dict(type='image', value=path) for path in paths] + [
            dict(type='text', value=prompt)
        ]

    def custom_instruction_prompt_by_dataset(self, dataset):
        if not self._use_custom_prompt:
            if dataset in {'MathVista_MINI', 'MM-IFEval', 'MMVet'}:
                return ''
            return DIRECT_ANSWER_PROMPT

        if dataset in NO_INSTRUCTION_DATASETS or _is_screenspot_v2(dataset):
            return ''
        return DATASET_INSTRUCTION_PROMPTS.get(dataset, DIRECT_ANSWER_PROMPT)

    @staticmethod
    def _content_item(item):
        if item['type'] == 'image':
            value = item['value']
            if isinstance(value, str) and os.path.isfile(value):
                with Image.open(value) as image:
                    value = image.convert('RGB')
            return {'type': 'image', 'url': value}
        if item['type'] == 'text':
            return {'type': 'text', 'text': item['value']}
        raise ValueError(f'Unsupported message item type: {item["type"]!r}')

    def message_to_chat_messages(self, message, instruction_prompt, dataset):
        is_multiturn = message and all('role' in turn and 'content' in turn for turn in message)
        if not is_multiturn:
            system = [item for item in message if item.get('role') == 'system']
            user = [item for item in message if item.get('role') != 'system']
            message = []
            if system:
                message.append({'role': 'system', 'content': system})
            message.append({'role': 'user', 'content': user})

        chat_messages = []
        for turn in message:
            content = [self._content_item(item) for item in turn['content']]
            if dataset == 'MM-IFEval' and turn['role'] == 'user':
                content = (
                    [item for item in content if item['type'] == 'image']
                    + [item for item in content if item['type'] != 'image']
                )
            chat_messages.append({'role': turn['role'], 'content': content})

        if instruction_prompt:
            user_turn = next(turn for turn in reversed(chat_messages) if turn['role'] == 'user')
            user_turn['content'].append({'type': 'text', 'text': instruction_prompt})
        return chat_messages

    def _generation_kwargs(self, dataset):
        kwargs = dict(self.kwargs)
        if dataset == 'MM-IFEval' and not self._max_new_tokens_explicit:
            kwargs['max_new_tokens'] = 1024
        return kwargs

    def _postprocess_response(self, response, dataset):
        if self._use_custom_prompt and _is_screenspot_v2(dataset):
            return _screen_spot_response(response)
        return response

    def generate_inner(self, message, dataset=None):
        instruction_prompt = self.custom_instruction_prompt_by_dataset(dataset)
        chat_messages = self.message_to_chat_messages(message, instruction_prompt, dataset)
        generation_inputs = self.processor.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self.model.device)

        history = self.model.generate(
            **generation_inputs,
            **self._generation_kwargs(dataset),
        )
        # Decoder-only generation returns the prompt followed by the completion.
        input_length = generation_inputs['input_ids'].shape[-1]
        generated_ids = history[:, input_length:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return self._postprocess_response(response, dataset)

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset)
