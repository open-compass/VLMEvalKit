from vlmeval.smp import *
import os
import sys
from vlmeval.api.base import BaseAPI
import math
from vlmeval.dataset import DATASET_TYPE
from vlmeval.dataset import img_root_map
from io import BytesIO
import pandas as pd
import requests
import json
import base64
import time


class HunyuanWrapper(BaseAPI):

    is_api: bool = True
    _apiVersion = '2024-12-31'
    _service = 'hunyuan'

    def __init__(self,
                 model: str = 'hunyuan-standard-vision',
                 retry: int = 5,
                 wait: int = 5,
                 secret_key: str = None,
                 secret_id: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = 'hunyuan.tencentcloudapi.com',
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.temperature = temperature

        warnings.warn('You may need to set the env variable HUNYUAN_SECRET_ID & HUNYUAN_SECRET_KEY to use Hunyuan. ')

        secret_key = os.environ.get('HUNYUAN_SECRET_KEY', secret_key)
        assert secret_key is not None, 'Please set the environment variable HUNYUAN_SECRET_KEY. '
        secret_id = os.environ.get('HUNYUAN_SECRET_ID', secret_id)
        assert secret_id is not None, 'Please set the environment variable HUNYUAN_SECRET_ID. '

        self.model = model
        self.endpoint = api_base
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.timeout = timeout

        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.hunyuan.v20230901 import hunyuan_client
        except ImportError as err:
            self.logger.critical('Please install tencentcloud-sdk-python to use Hunyuan API. ')
            raise err

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        cred = credential.Credential(self.secret_id, self.secret_key)
        httpProfile = HttpProfile(reqTimeout=300)
        httpProfile.endpoint = self.endpoint
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.client = hunyuan_client.HunyuanClient(cred, '', clientProfile)
        self.logger.info(
            f'Using Endpoint: {self.endpoint}; API Secret ID: {self.secret_id}; API Secret Key: {self.secret_key}'
        )

    def dump_image(self, line, dataset):
        """Dump the image(s) of the input line to the corresponding dataset folder.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str | list[str]: The paths of the dumped images.
        """
        ROOT = LMUDataRoot()
        assert isinstance(dataset, str)

        img_root = os.path.join(ROOT, 'images', img_root_map(dataset) if dataset in img_root_map(dataset) else dataset)
        os.makedirs(img_root, exist_ok=True)
        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def use_custom_prompt(self, dataset_name):
        if DATASET_TYPE(dataset_name) == 'MCQ':
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Answer with the option letter from the given choices directly.'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(Type='text', Text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img)
                    img_struct = dict(Url=f'data:image/jpeg;base64,{b64}')
                    content_list.append(dict(Type='image_url', ImageUrl=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(Type='text', Text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(Role='system', Content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(Role=item['role'], Contents=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(Role='user', Contents=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
        from tencentcloud.hunyuan.v20230901 import models

        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)

        payload = dict(
            Model=self.model,
            Messages=input_msgs,
            Temperature=temperature,
            TopK=1,
            **kwargs)

        try:
            req = models.ChatCompletionsRequest()
            req.from_json_string(json.dumps(payload))
            resp = self.client.ChatCompletions(req)
            resp = json.loads(resp.to_json_string())
            answer = resp['Choices'][0]['Message']['Content']
            return 0, answer, resp
        except TencentCloudSDKException as e:
            self.logger.error(f'Got error code: {e.get_code()}')
            if e.get_code() == 'ClientNetworkError':
                return -1, self.fail_msg + e.get_code(), None
            elif e.get_code() in ['InternalError', 'ServerNetworkError']:
                return -1, self.fail_msg + e.get_code(), None
            elif e.get_code() in ['LimitExceeded']:
                return -1, self.fail_msg + e.get_code(), None
            else:
                return -1, self.fail_msg + str(e), None


class HunyuanVision(HunyuanWrapper):

    def generate(self, message, dataset=None):
        return super(HunyuanVision, self).generate(message)
