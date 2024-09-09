import torch
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
from transformers import AutoTokenizer, AutoModel


class mPLUG_Owl3(BaseModel):
    # No separate model module is required, but the dependencies must be met.
    # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl3/requirements.txt
    INSTALL_REQ = True
    INTERLEAVE = True
    INSTALL_REQ_TXT = 'https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl3/requirements.txt'

    def __init__(self, model_path=None, **kwargs):
        assert model_path is not None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.half,
            trust_remote_code=True
        )
        self.model.eval().cuda()
        self.processor = self.model.init_processor(self.tokenizer)
        self.logger = get_logger('mPLUG_Owl3')
        if self.INSTALL_REQ:
            self.logger.info(
                f'Please remember to meet the requirements first\n'
                f'Here: {self.INSTALL_REQ_TXT}'
            )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    # Currently same to mPLUG_Owl2
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
        from PIL import Image
        image = Image.open(fname).convert('RGB')
        # resize to max_size
        max_size = 448 * 16
        if max(image.size) > max_size:
            w, h = image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        return image

    def generate_inner(self, message, dataset=None):
        num_images = len([x for x in message if x['type'] == 'image'])
        assert num_images >= 0

        images = []
        prompt_full = ''

        for msg in message:
            if msg['type'] == 'image':
                images.append(msg['value'])
                prompt_full += '<|image|>'
            elif msg['type'] == 'text':
                prompt_full += msg['value']

        needed_messages = [
            {'role': 'user', 'content': prompt_full},
            {'role': 'assistant', 'content': ''}
        ]

        images = [self.preproc_image(fname) for fname in images]

        inputs = self.processor(needed_messages, images=images, videos=None)

        inputs.to('cuda')
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens': 1024,
            'decode_text': True,
        })

        g = self.model.generate(**inputs)
        return g[0]
