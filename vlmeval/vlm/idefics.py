import torch
from PIL import Image
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen, listinstr
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


class IDEFICS(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_pth='HuggingFaceM4/idefics-9b-instruct',
                 **kwargs):
        assert osp.exists(model_pth) or splitlen(model_pth) == 2
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_pth, torch_dtype=torch.bfloat16, device_map='auto')
        self.processor = AutoProcessor.from_pretrained(model_pth)
        kwargs_default = {'max_length': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.file_root = osp.dirname(__file__)
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        prompts = ['Users:'] + [x['value'] for x in message] + ['<end_of_utterance>', '\nAssistant: ']
        inputs = self.processor(prompts, add_end_of_utterance_token=False, return_tensors='pt').to('cuda')
        exit_condition = self.processor.tokenizer('<end_of_utterance>', add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(
            ['<image>', '<fake_token_around_image>'],
            add_special_tokens=False).input_ids

        generated_ids = self.model.generate(
            **inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, **self.kwargs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = generated_text[0].split('\nAssistant: ')[-1]
        return text


class IDEFICS2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='HuggingFaceM4/idefics2-8b',
                 **kwargs):
        assert model_path is not None
        self.model_path = model_path
        # set size={"longest_edge": 448, "shortest_edge": 378} for limited maximum image resolution, such as MMMU.
        self.processor = AutoProcessor.from_pretrained(model_path, size={'longest_edge': 448, 'shortest_edge': 378})
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, device_map='cuda')
        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        prompt, images = '', []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value']
        prompt = f'User:{prompt}<end_of_utterance>\nAssistant:'
        inputs = self.processor(text=prompt, images=images, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = generated_text.split('\nAssistant:')[-1].strip()
        return response
