from vlmeval.smp import *
from vlmeval.api.base import BaseAPI

url = 'https://api.aimlapi.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}',
    "HTTP-Referer": "https://github.com/open-compass/VLMEvalKit",
    "X-Title": "VLMEvalKit",
}


class AIMLAPI_INT(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-4-turbo',
                 retry: int = 10,
                 wait: int = 3,
                 key: str = None,
                 temperature: float = 0,
                 max_tokens: int = 300,
                 verbose: bool = True,
                 system_prompt: str = None,
                 **kwargs):
        self.model = model
        self.fail_msg = 'Fail to obtain answer via API.'
        self.headers = dict(headers)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.key = key or os.environ.get('AIML_API_KEY', '')
        self.headers['Authorization'] = self.headers['Authorization'].format(self.key)

        super().__init__(retry=retry, wait=wait, verbose=verbose, system_prompt=system_prompt, **kwargs)

    @staticmethod
    def build_msgs(msgs_raw):
        messages = []
        content = []

        for msg in msgs_raw:
            if msg['type'] == 'text':
                content.append({
                    "type": "text",
                    "text": msg['value']
                })
            elif msg['type'] == 'image':
                image_b64 = encode_image_file_to_base64(msg['value'])
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{image_b64}"
                    }
                })

        messages.append({
            "role": "user",
            "content": content
        })
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        payload = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=self.build_msgs(inputs),
            **kwargs
        )
        response = requests.post(url, headers=self.headers, data=json.dumps(payload))
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response


class AIMLAPI(AIMLAPI_INT):

    def generate(self, message, dataset=None):
        return super(AIMLAPI_INT, self).generate(message)


if __name__ == '__main__':
    # export AIML_API_KEY=''
    model = AIMLAPI_INT(verbose=True)
    inputs = [
        {'type': 'image', 'value': '../../assets/apple.jpg'},
        {'type': 'text', 'value': 'Please describe this image in detail.'},
    ]
    code, answer, resp = model.generate_inner(inputs)
    print(code, answer, resp)
