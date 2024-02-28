from vlmeval.smp import *
from vlmeval.api.base import BaseAPI

headers = 'Content-Type: application/json'

class GeminiWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self, 
                 retry: int = 5,
                 wait: int = 5, 
                 key: str = None,
                 verbose: bool = True, 
                 temperature: float = 0.0, 
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 **kwargs):

        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        if key is None:
            key = os.environ.get('GOOGLE_API_KEY', None)
        assert key is not None
        self.api_key = key
        if proxy is not None:
            proxy_set(proxy)
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)
    
    @staticmethod
    def build_msgs(msgs_raw, system_prompt=None):
        msgs = cp.deepcopy(msgs_raw) 
        assert len(msgs) % 2 == 1

        if system_prompt is not None:
            msgs[0] = [system_prompt, msgs[0]]
        ret = []
        for i, msg in enumerate(msgs):
            role = 'user' if i % 2 == 0 else 'model'
            parts = msg if isinstance(msg, list) else [msg]
            ret.append(dict(role=role, parts=parts))
        return ret

    def generate_inner(self, inputs, **kwargs) -> str:
        import google.generativeai as genai
        assert isinstance(inputs, str) or isinstance(inputs, list)
        pure_text = True
        if isinstance(inputs, list):
            for pth in inputs:
                if osp.exists(pth) or pth.startswith('http'):
                    pure_text = False
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-pro') if pure_text else genai.GenerativeModel('gemini-pro-vision')
        if isinstance(inputs, str):
            messages = [inputs] if self.system_prompt is None else [self.system_prompt, inputs]
        elif pure_text:
            messages = self.build_msgs(inputs, self.system_prompt)
        else:
            messages = [] if self.system_prompt is None else [self.system_prompt]
            for s in inputs:
                if osp.exists(s):
                    messages.append(Image.open(s))
                elif s.startswith('http'):
                    pth = download_file(s)
                    messages.append(Image.open(pth))
                    shutil.remove(pth)
                else:
                    messages.append(s)
        gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)    
        gen_config.update(self.kwargs)
        try:
            answer = model.generate_content(messages, generation_config=genai.types.GenerationConfig(**gen_config)).text
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(err)
                self.logger.error(f"The input messages are {inputs}.")

            return -1, '', ''
        


class GeminiProVision(GeminiWrapper):

    def generate(self, image_path, prompt, dataset=None):
        return super(GeminiProVision, self).generate([image_path, prompt])
    
    def interleave_generate(self, ti_list, dataset=None):
        return super(GeminiProVision, self).generate(ti_list)