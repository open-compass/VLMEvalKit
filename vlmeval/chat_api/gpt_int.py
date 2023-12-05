import json
import warnings
import requests
from ..smp import *
from .gpt import GPT_context_window
from .base import BaseAPI


url = "http://ecs.sv.us.alles-apin.openxlab.org.cn/v1/openai/v2/text/chat"
headers = {
    "Content-Type": "application/json"
}

class OpenAIWrapperInternal(BaseAPI):

    is_api: bool = True

    def __init__(self, 
                 model: str = 'gpt-3.5-turbo-0613', 
                 retry: int = 5,
                 wait: int = 3,
                 verbose: bool = True,
                 temperature: float = 0, 
                 timeout: int = 30,
                 system_prompt: str = None, 
                 max_tokens: int = 1024,
                 **kwargs):
        self.model = model

        if 'KEYS' in os.environ and osp.exists(os.environ['KEYS']):
            keys = load(os.environ['KEYS'])
            headers['alles-apin-token'] = keys.get('alles-apin-token', '')
        self.headers = headers
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

        super().__init__(
            wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

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

        # Held out 100 tokens as buffer
        context_window = GPT_context_window(self.model)
        max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            warnings.warn('Less than 100 tokens left, may exceed the context window with some additional meta symbols. ')
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        payload = dict(
            model=self.model, 
            messages=input_msgs, 
            max_tokens=max_tokens,
            n=1,
            stop=None,
            timeout=self.timeout,
            temperature=temperature,
            **kwargs)
        
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            assert resp_struct['msg'] == 'ok' and resp_struct['msgCode'] == '10000', resp_struct
            answer = resp_struct['data']['choices'][0]['message']['content'].strip()
        except:
            pass
        return ret_code, answer, response
    
    def get_token_len(self, prompt: str) -> int:
        import tiktoken
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(prompt))