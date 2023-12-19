from ..smp import *
import os, openai
from openai import OpenAI
from .base import BaseAPI

APIBASES = {
    'OFFICIAL': "https://api.openai.com/v1/chat/completions",
    'INTERNAL': "https://ai-proxy.shlab.tech/internal"
}


def GPT_context_window(model):
    length_map = {
        'gpt-4-1106-preview': 128000, 
        'gpt-4-vision-preview': 128000, 
        'gpt-4': 8192,
        'gpt-4-32k': 32768,
        'gpt-4-0613': 8192, 
        'gpt-4-32k-0613': 32768,
        'gpt-3.5-turbo-1106': 16385, 
        'gpt-3.5-turbo': 4096, 
        'gpt-3.5-turbo-16k': 16385, 
        'gpt-3.5-turbo-instruct': 4096, 
        'gpt-3.5-turbo-0613': 4096, 
        'gpt-3.5-turbo-16k-0613': 16385, 
    }
    if model in length_map:
        return length_map[model]
    else:
        return 4096

class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self, 
                 model: str = 'gpt-3.5-turbo-0613', 
                 retry: int = 5,
                 wait: int = 5, 
                 key: str = None,
                 verbose: bool = True, 
                 system_prompt: str = None,
                 temperature: float = 0,
                 api_base: str = 'OFFICIAL',
                 max_tokens: int = 1024,
                 img_size: int = 512, 
                 img_detail: str = 'low',
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        openai_key = os.environ.get('OPENAI_API_KEY', None) if key is None else key
        self.openai_key = openai_key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail

        self.vision = False
        if model == 'gpt-4-vision-preview':
            self.vision = True
        
        assert isinstance(openai_key, str) and openai_key.startswith('sk-'), f'Illegal openai_key {openai_key}. Please set the environment variable OPENAI_API_KEY to your openai key. '

        if api_base in APIBASES:
            self.client = OpenAI(api_key=openai_key, base_url=APIBASES[api_base])
        else:
            self.client = OpenAI(api_key=openai_key)
        
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        if isinstance(inputs, str):
            input_msgs.append(dict(role='user', content=inputs))
            return input_msgs
        assert isinstance(inputs, list)
        dict_flag = [isinstance(x, dict) for x in inputs]
        if np.all(dict_flag):
            input_msgs.extend(inputs)
            return input_msgs
        str_flag = [isinstance(x, str) for x in inputs]
        if np.all(str_flag):
            img_flag = [x.startswith('http') or osp.exists(x) for x in inputs]
            if np.any(img_flag):
                content_list = []
                for fl, msg in zip(img_flag, inputs):
                    if not fl:
                        content_list.append(dict(type='text', text=msg))
                    elif msg.startswith('http'):
                        content_list.append(dict(type='image_url', image_url={'url': msg, 'detail': self.img_detail}))
                    elif osp.exists(msg):
                        from PIL import Image
                        img = Image.open(msg)
                        b64 = encode_image_to_base64(img, target_size=self.img_size)
                        img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                        content_list.append(dict(type='image_url', image_url=img_struct))
                input_msgs.append(dict(role='user', content=content_list))
                return input_msgs
            else:
                roles = ['user', 'assistant'] if len(inputs) % 2 == 1 else ['assistant', 'user']
                roles = roles * len(inputs)
                for role, msg in zip(roles, inputs):
                    input_msgs.append(dict(role=role, content=msg))
                return input_msgs
        raise NotImplemented("list of list prompt not implemented now. ")

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        context_window = GPT_context_window(self.model)
        max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            self.logger.warning('Less than 100 tokens left, may exceed the context window with some additional meta symbols. ')
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=input_msgs,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=temperature,
                **kwargs)
            
            result = response.choices[0].message.content.strip()
            return 0, result, 'API Call Succeed'
        except:
            if self.verbose:
                self.logger.warning(f'OPENAI KEY {self.openai_key} FAILED !!!')
                try:
                    self.logger.warning(response)
                except:
                    pass
            return -1, self.fail_msg, 'API Call Failed'

    def get_token_len(self, inputs) -> int:
        import tiktoken
        enc = tiktoken.encoding_for_model(self.model)
        if isinstance(inputs, str):
            if inputs.startswith('http') or osp.exists(inputs):
                return 65 if self.img_detail == 'low' else 130
            else:
                return len(enc.encode(inputs))
        elif isinstance(inputs, dict):
            assert 'content' in inputs
            return self.get_token_len(inputs['content'])
        assert isinstance(inputs, list)
        res = 0
        for item in inputs:
            res += self.get_token_len(item)
        return res
    

class GPT4V(OpenAIWrapper):

    def generate(self, image_path, prompt, dataset=None):
        assert self.model == 'gpt-4-vision-preview'
        return super(GPT4V, self).generate([image_path, prompt])
    
    def multi_generate(self, image_paths, prompt, dataset=None):
        assert self.model == 'gpt-4-vision-preview'
        return super(GPT4V, self).generate(image_paths + [prompt])
    
    def interleave_generate(self, ti_list, dataset=None):
        assert self.model == 'gpt-4-vision-preview'
        return super(GPT4V, self).generate(ti_list)
    