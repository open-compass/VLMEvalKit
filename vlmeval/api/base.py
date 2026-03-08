import os
import time
import random as rd
import json
import requests
import validators
from abc import abstractmethod
import os.path as osp
import copy as cp
import numpy as np
from ..smp import get_logger, parse_file, concat_images_vlmeval, encode_image_to_base64, parse_json, get_mime_type
from PIL import Image


class BaseAPI:

    allowed_types = ['text', 'image', 'video']
    INTERLEAVE: bool = True
    INSTALL_REQ: bool = False
    SUPPORT_GEN: bool = False
    is_api: bool = True

    def __init__(self,
                 retry=10,
                 wait=1,
                 system_prompt=None,
                 verbose=True,
                 fail_msg='Failed to obtain answer via API.',
                 keep_stats=False,
                 retryable_error_codes: list = [429, -4003],
                 retryable_error_patterns: list = ['Error code: 429', '"error":{"message":'],
                 text_pos: str = 'default',
                 min_edge: int = -1,
                 **kwargs):
        """Base Class for all APIs.

        Args:
            retry (int, optional): The retry times for `generate_inner`. Defaults to 10.
            wait (int, optional): The wait time after each failed retry of `generate_inner`. Defaults to 1.
            system_prompt (str, optional): Defaults to None.
            verbose (bool, optional): Defaults to True.
            fail_msg (str, optional): The message to return when failed to obtain answer.
                Defaults to 'Failed to obtain answer via API.'.
            keep_stats (bool, optional): Defaults to False.
            retryable_error_codes (list, optional): Defaults to [429, -4003].
            retryable_error_patterns (list, optional): Defaults to ['Error code: 429', '"error":{"message":'].
            text_pos (str, optional): The position of the text in the prompt. Defaults to 'default'.
                If 'default', will follow the default behavior.
                If 'end', will ensure query appears in the end, if the vqa is not an interleaved one.
                If 'start', will ensure query appears in the begining, if the vqa is not an interleaved one.
            min_edge (int, optional): The minimum edge length of the image. Defaults to -1.
            **kwargs: Other kwargs for `generate_inner`.
        """

        self.wait = wait
        self.retry = retry
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fail_msg = fail_msg
        self.logger = get_logger('ChatAPI')
        self.retryable_error_codes = retryable_error_codes
        self.retryable_error_patterns = retryable_error_patterns
        self.keep_stats = keep_stats
        self.text_pos = text_pos
        assert text_pos in ['default', 'end', 'start'], f'Invalid text_pos: {text_pos}'
        self.min_edge = min_edge
        if min_edge > 0:
            os.environ['VLMEVAL_MIN_IMAGE_EDGE'] = str(min_edge)

        if len(kwargs):
            self.logger.info(f'BaseAPI received the following kwargs: {kwargs}')
            self.logger.info('Will try to use them as kwargs for `generate`. ')
        self.default_kwargs = kwargs

    @abstractmethod
    def generate_inner(self, inputs, **kwargs):
        """The inner function to generate the answer.

        Returns:
            tuple(int, str, str): ret_code, response, log
        """
        self.logger.warning('For APIBase, generate_inner is an abstract method. ')
        assert 0, 'generate_inner not defined'
        ret_code, answer, log = None, None, None
        # if ret_code is 0, means succeed
        return ret_code, answer, log

    def working(self):
        """If the API model is working, return True, else return False.

        Returns:
            bool: If the API model is working, return True, else return False.
        """
        self.old_timeout = None
        if hasattr(self, 'timeout'):
            self.old_timeout = self.timeout
            self.timeout = 120

        retry = 5
        while retry > 0:
            ret = self.generate('hello')
            if ret is not None and ret != '' and self.fail_msg not in ret:
                if self.old_timeout is not None:
                    self.timeout = self.old_timeout
                return True
            retry -= 1

        if self.old_timeout is not None:
            self.timeout = self.old_timeout
        return False

    @classmethod
    def check_content(cls, msgs):
        """Check the content type of the input. Four types are allowed: str, dict, liststr, listdict.

        Args:
            msgs: Raw input messages.

        Returns:
            str: The message type.
        """
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [cls.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    @classmethod
    def preproc_content(cls, inputs):
        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if cls.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif cls.check_content(inputs) == 'dict':
            # Handle
            if 'content' in inputs:
                role = inputs.get('role', 'user')
                content = inputs['content']
                content = cls.preproc_content(content)
                for item in content:
                    item['role'] = role
                return content
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif cls.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif cls.check_content(inputs) == 'listdict':
            results = []
            for item in inputs:
                if 'content' in item:
                    role = item.get('role', 'user')
                    content = item['content']
                    content = cls.preproc_content(content)
                    for item in content:
                        item['role'] = role
                    results.extend(content)
                else:
                    mime, s = parse_file(item['value'])
                    if mime is None:
                        assert item['type'] == 'text', item['value']
                    else:
                        assert mime.split('/')[0] == item['type']
                        item['value'] = s
                    results.append(item)
            return results
        else:
            return None

    @classmethod
    def adjust_msg_order(cls, msgs, text_pos='default'):
        if text_pos == 'default':
            return msgs
        else:
            system_msgs = [x for x in msgs if x.get('role', 'user') == 'system']
            user_msgs = [x for x in msgs if x.get('role', 'user') == 'user']
            num_switch = 0
            for i in range(len(user_msgs) - 1):
                if user_msgs[i]['type'] != user_msgs[i + 1]['type']:
                    num_switch += 1
            if num_switch > 1:
                return msgs
            user_text_msgs = [x for x in user_msgs if x['type'] == 'text']
            user_image_msgs = [x for x in user_msgs if x['type'] == 'image']
            if text_pos == 'start':
                msgs = system_msgs + user_text_msgs + user_image_msgs
                return msgs
            elif text_pos == 'end':
                msgs = system_msgs + user_image_msgs + user_text_msgs
                return msgs

    # May exceed the context windows size, so try with different turn numbers.
    def chat_inner(self, inputs, **kwargs):
        _ = kwargs.pop('dataset', None)
        while len(inputs):
            try:
                return self.generate_inner(inputs, **kwargs)
            except Exception as e:
                if self.verbose:
                    self.logger.info(f'{type(e)}: {e}')
                inputs = inputs[1:]
                while len(inputs) and inputs[0]['role'] != 'user':
                    inputs = inputs[1:]
                continue
        return -1, self.fail_msg + ': ' + 'Failed with all possible conversation turns.', None

    def is_error_struct(self, response):
        try:
            data = json.loads(response)
            if 'error' in data:
                struct = data['error']
                assert 'code' in struct and 'message' in struct, struct
                struct['code'] = int(struct['code'])
                return struct
            else:
                return None
        except:
            return None

    def is_good_response(self, response):
        if response == '' or response is None:
            return False
        if isinstance(response, str) and self.fail_msg in response:
            return False
        for s in self.retryable_error_patterns:
            if s in response:
                return False
        if isinstance(response, Image.Image):
            return True
        # default to True, more cases to be included
        return True

    def chat(self, messages, **kwargs1):
        """The main function for multi-turn chatting. Will call `chat_inner` with the preprocessed input messages."""
        assert hasattr(self, 'chat_inner'), 'The API model should has the `chat_inner` method. '
        for msg in messages:
            assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, msg
            assert self.check_content(msg['content']) in ['str', 'dict', 'liststr', 'listdict'], msg
            msg['content'] = self.preproc_content(msg['content'])
        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        assert messages[-1]['role'] == 'user'

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.chat_inner(messages, **kwargs)
                if self.is_error_struct(answer):
                    struct = self.is_error_struct(answer)
                    ret_code = struct['code']
                    answer = struct['message']
                if self.is_good_response(answer):
                    if self.verbose:
                        print(answer)
                    return answer
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception as e:
                            self.logger.warning(f'Failed to parse {log} as an http response: {str(e)}. ')
                    self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}: ')
                    self.logger.error(f'{type(err)}: {err}')
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        return self.fail_msg if answer in ['', None] else answer

    @staticmethod
    def _openai_image_url_struct(value):
        # Now there can be 3 cases: b64, local_file, or url
        if value.startswith('data:image'):
            return dict(url=value)
        elif osp.exists(value):
            img = Image.open(value)
            suffix = osp.splitext(value)[1].lower()
            mime = get_mime_type(suffix)
            if mime == 'unknown':
                mime = 'image/png'
            fmt = 'JPEG' if 'jpeg' in mime else 'PNG'
            b64 = encode_image_to_base64(img, fmt=fmt)
            return dict(url=f'data:{mime};base64,{b64}')
        elif validators.url(value):
            im = Image.open(requests.get(value, stream=True).raw)
            suffix = osp.splitext(value)[1].lower()
            mime = get_mime_type(suffix)
            if mime == 'unknown':
                mime = 'image/png'
            fmt = 'JPEG' if 'jpeg' in mime else 'PNG'
            b64 = encode_image_to_base64(im, fmt=fmt)
            return dict(url=f'data:{mime};base64,{b64}')
        else:
            raise NotImplementedError(f'Unknown image type: {value}')

    @staticmethod
    def _compress_openai_image_url_struct(img_struct, target_size=-1, image_pixel_limit=None, compress='JPEG'):
        assert not (target_size > 0 and image_pixel_limit is not None), '"target_size" and "pixel_limits" can not be both activated. '  # noqa: E501
        b64 = img_struct['url'].split(';base64,')[-1]
        old_mime = img_struct['url'].split(';base64,')[0].split('data:')[1]
        from ..smp import decode_base64_to_image, encode_image_to_base64, resize_image_by_pixel_limits
        img = decode_base64_to_image(b64)
        if compress is None:
            compress = 'PNG' if 'png' in old_mime.lower() else 'JPEG'
        if target_size > 0:
            b64_new = encode_image_to_base64(img, target_size=target_size, fmt=compress)
        elif image_pixel_limit is not None:
            img = resize_image_by_pixel_limits(img, image_pixel_limit)
            b64_new = encode_image_to_base64(img, fmt=compress)
        mime = old_mime if compress != 'JPEG' else 'image/jpeg'
        img_struct['url'] = f'data:{mime};base64,{b64_new}'
        return img_struct

    def preprocess_message_with_role(self, message):
        system_prompt = ''
        new_message = []

        for data in message:
            assert isinstance(data, dict)
            role = data.pop('role', 'user')
            if role == 'system':
                system_prompt += data['value'] + '\n'
            else:
                new_message.append(data)

        if system_prompt != '':
            if self.system_prompt is None:
                self.system_prompt = system_prompt
            else:
                if system_prompt not in self.system_prompt:
                    self.system_prompt += '\n' + system_prompt
        return new_message

    def generate(self, message, **kwargs1):
        """The main function to generate the answer. Will call `generate_inner` with the preprocessed input messages.

        Args:
            message: raw input messages.

        Returns:
            str: The generated answer of the Failed Message if failed to obtain answer.
        """
        if self.check_content(message) == 'listdict':
            message = self.preprocess_message_with_role(message)

        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

        # merge kwargs
        _ = kwargs1.pop('dataset', None)
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        retry = self.retry

        while retry > 0:
            ret_code = -1
            try:
                ret_code, answer, log = self.generate_inner(message, **kwargs)
                if self.is_error_struct(answer):
                    struct = self.is_error_struct(answer)
                    ret_code = struct['code']
                    answer = struct['message']
                if self.is_good_response(answer):
                    if self.verbose:
                        print(answer)
                    return answer if not self.keep_stats else dict(response=answer, stats=log)
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception as e:
                            self.logger.warning(f'Failed to parse {log} as an http response: {str(e)}. ')
                    self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {self.retry - retry}: ')
                    self.logger.error(f'{type(err)}: {err}')
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)
            if ret_code in self.retryable_error_codes:
                self.logger.info(
                    f'Encountered retryable error code {ret_code}, the next retry will not count, '
                    f'remaining retry times: {retry}')
            elif answer is not None and any([x in answer for x in self.retryable_error_patterns]):
                self.logger.info(
                    f'Encountered retryable error pat {answer}, the next retry will not count, '
                    f'remaining retry times: {retry}')
            else:
                retry -= 1

        if answer in ['', None]:
            return self.fail_msg if not self.keep_stats else dict(response=self.fail_msg, stats=None)
        return answer if not self.keep_stats else dict(response=answer, stats=log)

    def message_to_promptimg(self, message, dataset=None):
        assert not self.INTERLEAVE
        model_name = self.__class__.__name__
        import warnings
        warnings.warn(
            f'Model {model_name} does not support interleaved input. '
            'Will use the first image and aggregated texts as prompt. ')
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        elif num_images == 1:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image'][0]
        else:
            prompt = '\n'.join([x['value'] if x['type'] == 'text' else '<image>' for x in message])
            if dataset == 'BLINK':
                image = concat_images_vlmeval(
                    [x['value'] for x in message if x['type'] == 'image'],
                    target_size=512)
            else:
                image = [x['value'] for x in message if x['type'] == 'image'][0]
        return prompt, image

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def pydantic_model_validate(self, resp_text, response_format):
        try:
            eval_data = parse_json(resp_text)
            if hasattr(response_format, 'model_validate'):
                _ = response_format.model_validate(eval_data)
            elif hasattr(response_format, 'validate'):
                _ = response_format.validate(eval_data)
            return eval_data
        except Exception as e:
            self.logger.warning(f'{type(e)}: {str(e)}')
            self.logger.warning(f'json parsing failed, model {response_format}, data {resp_text}')
            return None


class BasicAPIWrapper(BaseAPI):
    KEY_NAME = 'OPENAI_API_KEY'
    URL_NAME = 'OPENAI_API_BASE'
    DEFAULT_URL = 'https://api.openai.com/v1/chat/completions'

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        env_key = os.environ.get(self.KEY_NAME, None)
        if key is None:
            key = env_key
        assert isinstance(key, str) and len(key), key
        self.key = key
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        env_url = os.environ.get(self.URL_NAME, None)
        if api_base is None:
            if isinstance(env_url, str) and len(env_url) and validators.url(env_url):
                api_base = env_url
            else:
                api_base = self.DEFAULT_URL
        assert api_base is not None, api_base
        self.api_base = api_base
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url=api_base.split('/chat/completions')[0])
        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

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
                    # Now there can be 3 cases: b64, local_file, or url
                    img_struct = self._openai_image_url_struct(msg['value'])
                    if getattr(self, 'img_detail', None):
                        img_struct['detail'] = self.img_detail
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
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
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        response_format = kwargs.pop('response_format', None)

        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs)
        if self.verbose:
            print('Payload kwargs:', kwargs)

        if response_format is not None:
            payload['response_format'] = response_format
            response = self.client.beta.chat.completions.parse(**payload)
            api_result = response.choices[0].message.parsed
            res = api_result.model_dump() if hasattr(api_result, 'model_dump') else api_result.dict()
            return 0, res, response

        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout)

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
