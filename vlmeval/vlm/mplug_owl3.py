import sys
import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
from transformers import AutoTokenizer, AutoModelForCausalLM


class mPLUG_Owl3(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='mPLUG/mPLUG-Owl3-7B-240728', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.half,
            trust_remote_code=True
        )
        self.model = model.eval().cuda()
        self.processor = self.model.init_processor(self.tokenizer)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset == 'MMVet':
            prompt = question + '\nAnswer the question directly. '
        elif DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'Hint: {hint}\n' if hint is not None else ''
            prompt += f'{question}\n'
            prompt += (
                f'{options_prompt}\nAnswer with the option’s letter from the given choices directly. '
                if len(options) else 'Answer the question directly. '
            )
        else:
            raise NotImplementedError

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def preproc_image(self, fname):
        image = Image.open(fname).convert('RGB')
        # TAG-DSY: need or not?
        # max_edge = max(image.size)
        # image = image.resize((max_edge, max_edge))
        return image

    def generate_inner(self, message, dataset=None):
        num_images = len([x for x in message if x['type'] == 'image'])
        assert num_images >= 0

        images = []
        prompt_full = ""

        # 需要修改，这个模型能interleave
        # if num_images == 1:
        #     prompt, image = self.message_to_promptimg(message, dataset=dataset)
        #     prompt_full += f'<|image|>{prompt}'
        #     images.append(image)
        # else:
        #     for msg in message:
        #         if msg['type'] == 'image':
        #             images.append(msg['value'])
        #             prompt_full += '<|image|>'
        #         elif msg['type'] == 'text':
        #             prompt_full += msg['value']
        #     prompt_full += '\nASSISTANT: '

        needed_messages = [
            {"role": "user", "content": prompt_full},
            {"role": "assistant", "content": ""}
        ]
        images = [self.preproc_image(fname) for fname in images]

        inputs = self.processor(needed_messages, images=images, videos=None)

        inputs.to('cuda')
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens': 100,
            'decode_text': True,
        })

        g = self.model.generate(**inputs)
        print(g)

        return g
