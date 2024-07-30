from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from time import sleep
import base64
import mimetypes
from PIL import Image

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

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    pth = msg['value']
                    suffix = osp.splitext(pth)[-1].lower()
                    media_type = mimetypes.types_map.get(suffix, None)
                    assert media_type is not None

                    content_list.append(dict(
                        type='image',
                        source={
                            'type': 'base64',
                            'media_type': media_type,
                            'data': encode_image_file_to_base64(pth, target_size=4096)
                        }))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:

        payload = json.dumps({
            'model': self.model,
            'max_tokens': self.max_tokens,
            'messages': self.prepare_inputs(inputs),
            'system': self.system_prompt,
            **kwargs
        })
        response = requests.request('POST', url, headers=headers, data=payload)
        ret_code = response.status_code
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
