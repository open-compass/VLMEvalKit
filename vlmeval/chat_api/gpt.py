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
                 openai_key: str = None,
                 verbose: bool = True, 
                 system_prompt: str = None,
                 temperature: float = 0,
                 api_base: str = 'OFFICIAL',
                 max_tokens: int = 1024,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        openai_key = os.environ.get('OPENAI_API_KEY', None) if openai_key is None else openai_key
        self.openai_key = openai_key
        
        assert isinstance(openai_key, str) and openai_key.startswith('sk-'), f'Illegal openai_key {openai_key}. Please set the environment variable OPENAI_API_KEY to your openai key. '

        if api_base in APIBASES:
            self.client = OpenAI(api_key=openai_key, base_url=APIBASES[api_base])
        else:
            self.client = OpenAI(api_key=openai_key)
        
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))

        if isinstance(inputs, str):
            input_msgs.append(dict(role='user', content=inputs))
        elif isinstance(inputs[0], str):
            roles = ['user', 'assistant'] if len(inputs) % 2 == 1 else ['assistant', 'user']
            roles = roles * len(inputs)
            for role, msg in zip(roles, inputs):
                input_msgs.append(dict(role=role, content=msg))
        elif isinstance(inputs[0], dict):
            input_msgs.extend(inputs)
        else:
            raise NotImplementedError
        
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

    def get_token_len(self, prompt: str) -> int:
        import tiktoken
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(prompt))
