import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..utils import DATASET_TYPE
from transformers import AutoModelForCausalLM, LlamaTokenizer


class CogVlm(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self,
                 name='cogvlm-chat',
                 tokenizer_name='lmsys/vicuna-7b-v1.5',
                 **kwargs):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        self.kwargs = kwargs
        self.model = AutoModelForCausalLM.from_pretrained(
            f'THUDM/{name}-hf',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to('cuda').eval()

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question = hint + '\n' + question

            option_candidate = string.ascii_uppercase
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = question

            if not cn_string(prompt):
                prompt = prompt + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                prompt = prompt + '\n' + '请直接回答选项字母。'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])

        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)

        image = Image.open(image_path).convert('RGB')
        inputs = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {'max_length': 2048, 'do_sample': False}
        gen_kwargs.update(self.kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
        response = response.split('</s>')[0].strip()
        return response
