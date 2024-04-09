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

    def build_msgs(self, inputs):
        messages = [] if self.system_prompt is None else [self.system_prompt]
        for inp in inputs:
            if inp['type'] == 'text':
                messages.append(inp['value'])
            elif inp['type'] == 'image':
                messages.append(Image.open(inp['value']))
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        import google.generativeai as genai
        assert isinstance(inputs, list)
        pure_text = np.all([x['type'] == 'text' for x in inputs])
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-pro') if pure_text else genai.GenerativeModel('gemini-pro-vision')
        messages = self.build_msgs(inputs)
        gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)
        gen_config.update(kwargs)
        try:
            answer = model.generate_content(messages, generation_config=genai.types.GenerationConfig(**gen_config)).text
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(err)
                self.logger.error(f'The input messages are {inputs}.')

            return -1, '', ''


class GeminiProVision(GeminiWrapper):

    def generate(self, message, dataset=None):
        return super(GeminiProVision, self).generate(message)
