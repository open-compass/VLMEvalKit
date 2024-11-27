import torch
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen
from PIL import Image

import os
import math


class SmolVLM(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceTB/SmolVLM-Instruct', **kwargs):
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration
        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map='cuda'
        )

        kwargs_default = {'max_new_tokens': 512,
                          'use_cache': True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config.')
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        if dataset in ['MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench',
                       'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_EN_V11',
                       'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'CCBench']:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ['MathVista_MINI']:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in ['MME', 'MMVet', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'TextVQA_VAL',
                         'ChartQA_TEST', 'DocVQA_VAL', 'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST']:
            formatted_messages, formatted_images = self.build_prompt_default(message, add_brief=True)
        elif dataset == 'HallusionBench':
            formatted_messages, formatted_images = self.build_prompt_default(message, add_yes_or_no=True)
        elif dataset in ['MMStar', 'SEEDBench_IMG', 'AI2D_TEST', 'ScienceQA_VAL', 'ScienceQA_TEST']:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        images = [formatted_images] if isinstance(formatted_images, Image.Image) else formatted_images
        inputs = self.processor(text=formatted_messages, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        from transformers.image_utils import load_image
        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value'].strip()
        if add_brief:
            prompt += '\nGive a very brief answer.'
        if add_yes_or_no:
            prompt += '\nAnswer yes or no.'
        prompt += '<end_of_utterance>\nAssistant:'
        return prompt, images

    def build_prompt_puremcq(self, message):
        from transformers.image_utils import load_image
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    def build_prompt_mt(self, message):
        from transformers.image_utils import load_image
        prompt, images = '', []
        for msg in message:
            if msg['role'] == 'user':
                prompt += 'User: '
            elif msg['role'] == 'assistant':
                prompt += 'Assistant: '
            for item in msg['content']:
                if item['type'] == 'image':
                    img = load_image(item['value'])
                    images.append(img)
                elif item['type'] == 'text':
                    prompt += item['value'].strip()
                prompt += '<end_of_utterance>\n'
        return prompt + 'Assistant: '

    def build_prompt_mmbench(self, message):
        from transformers.image_utils import load_image
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with a letter.',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if instruction.startswith('Hint:'):
                    hint, question = instruction.split('\nQuestion:')
                    question, choices = question.split('\nChoices:')
                    instruction = (
                        'Question:' + question + '\n' + hint + '\nChoices:' + choices
                    )
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    def build_prompt_mmmu(self, message):
        from transformers.image_utils import load_image
        replace_mapping = {
            'Question:': '',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
            '\nOptions:': '\nChoices:',
        }

        prompt, images, img_counter = 'User: Question: ', [], 1
        for msg in message:
            if msg['type'] == 'image':
                prompt += f'<image {img_counter}>:<image>\n'
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += f' <image {img_counter}> '
                img_counter += 1
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    def build_prompt_mathvista(self, message):
        from transformers.image_utils import load_image
        replace_mapping = {
            '(A) ': 'A. ',
            '(B) ': 'B. ',
            '(C) ': 'C. ',
            '(D) ': 'D. ',
            '(E) ': 'E. ',
            '(F) ': 'F. ',
            '(G) ': 'G. ',
            '(H) ': 'H. ',
            '\nOptions:': '\nChoices:',
            'Hint: ': '',
        }

        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()

        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        images = [formatted_images] if isinstance(formatted_images, Image.Image) else formatted_images

        resulting_messages = [{"role": "user", "content": [{"type": "image"}]
                               + [{"type": "text", "text": formatted_messages}]}]
        prompt = self.processor.apply_chat_template(resulting_messages, add_generation_prompt=True)

        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True
        )[0]

        return generated_text.strip()
