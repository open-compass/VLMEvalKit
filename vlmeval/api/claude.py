from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from time import sleep
import base64
import mimetypes

url = 'https://openxlab.org.cn/gw/alles-apin-hub/v1/claude/v1/text/chat'
headers = {
    'alles-apin-token': '',
    'Content-Type': 'application/json'
}


class Claude_Wrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'claude-3-opus-20240229',
                 key: str = None,
                 retry: int = 10,
                 wait: int = 3,
                 system_prompt: str = None,
                 verbose: bool = True,
                 temperature: float = 0,
                 max_tokens: int = 1024,
                 **kwargs):

        self.model = model
        self.headers = headers
        self.temperature = temperature
        self.max_tokens = max_tokens
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get('ALLES', '')
        self.headers['alles-apin-token'] = self.key

        super().__init__(retry=retry, wait=wait, verbose=verbose, system_prompt=system_prompt, **kwargs)

    def build_msgs(self, msgs_raw):

        messages = []
        message = {'role': 'user', 'content': []}
        for msg in msgs_raw:
            if msg['type'] == 'image':
                pth = msg['value']
                suffix = osp.splitext(pth)[-1].lower()
                media_type = mimetypes.types_map.get(suffix, None)
                assert media_type is not None

                item = {
                    'type': 'image',
                    'source': {'type': 'base64', 'media_type': media_type, 'data': encode_image_file_to_base64(pth)}
                }

            elif msg['type'] == 'text':
                item = {'type': 'text', 'text': msg['value']}
            else:
                raise NotImplementedError(f'Unsupported message type: {msg["type"]}')

            message['content'].append(item)
        messages.append(message)
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:

        payload = json.dumps({
            'model': self.model,
            'max_tokens': self.max_tokens,
            'messages': self.build_msgs(msgs_raw=inputs),
            **kwargs
        })
        response = requests.request('POST', url, headers=headers, data=payload)

        ret_code = response.status_code
        retry = self.retry
        while ret_code == 429 and retry > 0:
            sleep(15)
            response = requests.request('POST', url, headers=headers, data=payload)
            ret_code = response.status_code
            retry -= 1

        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg

        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['data']['content'][0]['text'].strip()
        except:
            pass

        return ret_code, answer, response


class Claude3V(Claude_Wrapper):

    def generate(self, message, dataset=None):
        return super(Claude_Wrapper, self).generate(message)
