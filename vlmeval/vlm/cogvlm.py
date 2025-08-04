import re
import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class GLM4v(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='THUDM/glm-4v-9b', **kwargs):
        from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
        gen_kwargs = {'max_length': 2048, 'do_sample': False}
        gen_kwargs.update(kwargs)
        self.kwargs = gen_kwargs
        self.end_text_token = '<|endoftext|>'

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        if dataset is not None and DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            prompt += '\nShort Answer.'
        inputs = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'image': image, 'content': prompt}],
            add_generation_prompt=True, tokenize=True, return_tensors='pt', return_dict=True
        )
        inputs = inputs.to('cuda')

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
        return response.split(self.end_text_token)[0]

class GLM4_1V9BThinking(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='THUDM/GLM-4.1V-9B-Thinking', **kwargs):
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        self.device = 'cuda'
        
        print(f"Loading processor from {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True
        )

        self.model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True
        ).to(self.device)

    def build_msgs(self, msgs_raw, system_prompt=None, dataset=None):
        msgs = cp.deepcopy(msgs_raw)
        content = []
        for i, msg in enumerate(msgs):
            if msg['type'] == 'text':
                content.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                content.append(dict(type='image', 
                                    url=f"data:image/jpeg;base64,{encode_image_file_to_base64(msg['value'])}"))
        if dataset in {'HallusionBench', 'POPE'}:
            content.append(dict(type="text", text="Please answer yes or no."))
        ret = [dict(role='user', content=content)]
        return ret
    
    def extract_answer(self, response_text, dataset=None):
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
        if dataset in {'OCRBench', 'MMLongBench_DOC'}:
            return response_text
        # extract box
        pattern_box = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
        match = re.search(pattern_box, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response_text

    def generate_inner(self, message, dataset=None):
        try:
            inputs = message
            assert isinstance(inputs, str) or isinstance(inputs, list)
            inputs = [inputs] if isinstance(inputs, str) else inputs

            messages = self.build_msgs(msgs_raw=inputs, dataset=dataset)

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            # ✅ 执行生成
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)

            # ✅ 解码输出
            answer = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return self.extract_answer(answer, dataset=dataset)
        except Exception as err:
            print(err)
            return 'Failed to obtain answer.'

class CogVlm(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='THUDM/cogvlm2-llama3-chat-19B', tokenizer_name=None, **kwargs):
        from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
        assert model_path is not None
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to('cuda').eval()

        self.kwargs = kwargs
        if tokenizer_name:
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
            gen_kwargs = {'max_length': 2048, 'do_sample': False}
            self.end_text_token = '</s>'
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            gen_kwargs = {'max_new_tokens': 2048, 'pad_token_id': 128002}
            self.end_text_token = '<|end_of_text|>'
        self.kwargs.update(gen_kwargs)
        self.tokenizer = tokenizer
        self.model = model

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
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
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        if dataset is not None and DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            prompt += '\nShort Answer.'

        image = Image.open(image_path).convert('RGB')
        inputs = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
        response = response.split(self.end_text_token)[0].strip()
        return response
