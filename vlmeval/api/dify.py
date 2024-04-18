import base64
import time
from io import BytesIO
from dify_client import ChatClient

from vlmeval.smp import *
from vlmeval.api.base import BaseAPI


class DifyWrapper(BaseAPI):
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
            key = os.environ.get('DIFY_API_KEY', None)
        assert key is not None
        self.api_key = key
        if proxy is not None:
            proxy_set(proxy)
        self.chat_client = ChatClient(self.api_key)
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def build_msgs(self, inputs):
        text, image = '', ''
        for inp in inputs:
            if inp['type'] == 'text':
                text = inp['value']
            elif inp['type'] == 'image':
                image = self.upload(inp['value'])
        return text, image

    def upload(self, file_path):
        file_name = f"{int(time.time() * 1000)}.jpeg"
        mime_type = "image/jpeg"
        if '.' in file_path:
            file = open(file_path, "rb")
        else:
            file = BytesIO(base64.b64decode(file_path))
        files = {
            "file": (file_name, file, mime_type)
        }
        response = self.chat_client.file_upload("VLMEvalKit", files)
        result = response.json()
        return result.get("id")

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, list)
        text, image_id = self.build_msgs(inputs)
        print(text, image_id)
        files = [{
            "type": "image",
            "transfer_method": "local_file",
            "upload_file_id": image_id
        }]
        try:
            chat_response = self.chat_client.create_chat_message(inputs={}, query=text, user="VLMEvalKit",
                                                                 response_mode="blocking", files=files)

            chat_response.raise_for_status()
            result = chat_response.json()
            answer = result.get('answer')
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(err)
                self.logger.error(f'The input messages are {inputs}.')
            return -1, '', ''


class DifyVision(DifyWrapper):

    def generate(self, message, dataset=None):
        return super(DifyVision, self).generate(message)
