import os
import re
import math
from typing import List, Dict, Any, Tuple
from ..smp import *
from .base import BaseAPI
from io import BytesIO
import threading
import random

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def convert_openai_to_gemini_format(
    input_messages: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], str]:
    """
    将OpenAI格式的消息转换为Gemini API兼容格式（修复字段层级问题）
    
    参数:
    input_messages: [
        {"role": "system/user/assistant", "content": "文本或内容数组"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }
    ]
    
    返回: (contents, system_instruction)
    正确结构:
        contents = [{
            "role": "user",  # 顶层字段
            "parts": [{"text": "内容"}, {"inlineData": ...}]  # 顶层字段
        }]
    """
    contents = []
    system_instruction = ""
    IMAGE_PATTERN = re.compile(r'data:image/(\w+);base64,(.+)')

    # 提取系统消息
    system_msgs = [msg["content"] for msg in input_messages if msg.get("role") == "system"]
    if system_msgs:
        if all(isinstance(msg, str) for msg in system_msgs):
            system_instruction = "\n".join(system_msgs)

    def create_parts(content: Any) -> List[Dict]:
        """创建Gemini parts数组"""
        parts = []
        if isinstance(content, list):
            for item in content:
                if item["type"] == "text":
                    parts.append({"text": item["text"]})
                elif item["type"] == "image_url":
                    match = IMAGE_PATTERN.match(item["image_url"]["url"])
                    if match:
                        parts.append({
                            "inlineData": {  # 使用正确字段名
                                "mimeType": f"image/{match.group(1).lower()}",
                                "data": match.group(2)
                            }
                        })
        elif isinstance(content, str):
            parts.append({"text": content})
        return parts

    # 处理对话消息
    for msg in input_messages:
        role = msg.get("role")
        if role in ["system", "function"]:
            continue  # 跳过系统消息

        # 角色映射
        gemini_role = "user" if role == "user" else "model"
        
        parts = create_parts(msg.get("content"))
        if parts:
            # 正确结构：role和parts在顶层
            contents.append({
                "role": gemini_role,  # 顶层字段
                "parts": parts         # 顶层字段
            })
    
    return contents, system_instruction

def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

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
                 img_size: int = -1,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'internvl2-pro' in model:
            env_key = os.environ.get('InternVL2_PRO_KEY', '')
            if key is None:
                key = env_key
        elif 'abab' in model:
            env_key = os.environ.get('MiniMax_API_KEY', '')
            if key is None:
                key = env_key
        elif 'moonshot' in model:
            env_key = os.environ.get('MOONSHOT_API_KEY', '')
            if key is None:
                key = env_key
        elif 'grok' in model:
            env_key = os.environ.get('XAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'gemini' in model:
            # Will only handle preview models
            env_key = os.environ.get('GOOGLE_API_KEY', '')
            if key is None:
                key = env_key
            api_base = os.environ.get('GOOGLE_API_BASE', "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        elif 'ernie' in model:
            env_key = os.environ.get('BAIDU_API_KEY', '')
            if key is None:
                key = env_key
            api_base = 'https://qianfan.baidubce.com/v2/chat/completions'
            self.baidu_appid = os.environ.get('BAIDU_APP_ID', None)
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None or key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key
                else:
                    assert isinstance(key, str), (
                        f'Illegal openai_key {key}. '
                        'Please set the environment variable OPENAI_API_KEY to your openai key. '
                    )

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        self.is_max_completion_tokens = ('o1' in model) or ('o3' in model) or ('o4' in model) or ('gpt-5' in model)
        self.is_o_model = ('o1' in model) or ('o3' in model) or ('o4' in model)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            if api_base is None:
                api_base_template = (
                    '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
                )
                endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
                assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
                deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
                assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
                api_version = os.getenv('OPENAI_API_VERSION', None)
                assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

                self.api_base = api_base_template.format(
                    endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                    deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                    api_version=os.getenv('OPENAI_API_VERSION')
                )
            else:
                self.api_base = api_base
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            elif isinstance(api_base, list):
                for api in api_base:
                    assert api.startswith('http')
                self.api_base = api_base
            else:
                self.logger.error('Unknown API Base. ')
                raise NotImplementedError
            if os.environ.get('BOYUE', None):
                self.api_base = os.environ.get('BOYUE_API_BASE')
                self.key = os.environ.get('BOYUE_API_KEY')

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
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
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

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        elif 'internvl2-pro' in self.model:
            headers = {'Content-Type': 'application/json', 'Authorization': self.key}
        elif self.key is None:
            headers = {'Content-Type': 'application/json'}
        elif 'gemini' in self.model:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}

        if 'gemini' in self.model:
            input_msgs, system_instruction = convert_openai_to_gemini_format(input_msgs)
            payload = {
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "thinkingConfig": {"includeThoughts": True}
                },
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": input_msgs
            }
            if kwargs.get('thinking_level', None) in ["minimal", "low", "middle", "high"]:
                payload["generationConfig"]["thinkingConfig"]["thinking_level"] = kwargs.get('thinking_level', None)
            elif kwargs.get('thinking_level', None) is not None:
                raise ValueError(f"Unknown thinking_level value: {kwargs.get('thinking_level', None)}, must be in [minimal, low, middle, high]")
        else:   
            payload = dict(
                model=self.model,
                messages=input_msgs,
                n=1,
                temperature=temperature,
                **kwargs)
        if hasattr(self, 'baidu_appid'):
            headers['appid'] = self.baidu_appid

        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            if 'gemini' in self.model:
                answer = resp_struct["candidates"][0]["content"]["parts"][-1]["text"].strip()
                print(answer)
            else:
                answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception as err:
            if 'gemini' in self.model:
                print(response.text)
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if 'gpt' in self.model.lower():
                if self.verbose:
                    self.logger.warning(f'{type(err)}: {err}')
                enc = tiktoken.encoding_for_model('gpt-4')
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


def get_image_fmt_from_image_path(image_path):
    # 根据图片后缀获取图片格式
    _, ext = os.path.splitext(image_path)
    if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
        return 'jpeg'
    elif ext.lower() == '.png':
        return 'png'
    else:
        return 'jpeg'

class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)

def compress_image(base64_str, quality=85, format='JPEG'):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))

    # 检查是否有透明度通道
    # if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
    if img.mode in ('RGBA', 'LA', 'P'):
        # 创建一个白色背景的图像
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGB", img.size, (255, 255, 255) + (255,))
        bg.paste(img, mask=alpha)
        img = bg

    # 检查图像尺寸
    max_size = 36000000  # doubao的最大宽高乘积
    if img.size[0] * img.size[1] > max_size:
        ratio = (max_size / (img.size[0] * img.size[1])) ** 0.5
        new_width = int(img.size[0] * ratio)
        new_height = int(img.size[1] * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)


    # 压缩并转换格式
    buf = BytesIO()
    img.save(buf, format=format, quality=quality)
    byte_img = buf.getvalue()
    return base64.b64encode(byte_img).decode('utf-8')

def custom_encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    max_size = os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e9)
    min_edge = os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 1e2)
    max_size = int(max_size)
    min_edge = int(min_edge)

    image_size = img.size[0] * img.size[1]
    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')
        image_size = image_new.size[0] * image_new.size[1]
    
    factor = 1
    while image_size > max_size:
        factor = math.sqrt(max_size / image_size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')
        image_size = image_new.size[0] * image_new.size[1]

    if factor < 1:
        new_w, new_h = image_new.size
        current_time = time.localtime()
        if current_time.tm_sec == 0:
            print(
                f'Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, '
                f'resize to {factor:.2f} of original size: ({new_w}, {new_h})'
            )

    return ret

class VLLMAPIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = -1,
                 reasoning_effort: str = None,
                 **kwargs):
        
        if model is None:
            if 'VLLM_MODEL_NAME' in os.environ and os.environ['VLLM_MODEL_NAME'] != '':
                model = os.environ['VLLM_MODEL_NAME']
        self.model = model
        assert self.model, f"You must set model name."
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.img_size = img_size

        env_key = os.environ.get('VLLM_API_KEY', '')
        if key is None:
            key = env_key
        else:
            assert isinstance(key, str), (f'Illegal openai_key {key}. ')

        self.key = key
        self.timeout = timeout
        self.cur_idx = 0
        self.cur_idx_lock = threading.Lock()
        self.reasoning_effort = reasoning_effort

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if api_base is None:
            if 'VLLM_API_BASE' in os.environ and os.environ['VLLM_API_BASE'] != '':
                self.logger.info('Environment variable VLLM_API_BASE is set. Will use it as api_base. ')
                api_base = os.environ['VLLM_API_BASE']
                if "," in api_base:
                    api_base = api_base.split(",")

            assert api_base is not None

            if isinstance(api_base, str) and api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif isinstance(api_base, str) and api_base.startswith('http'):
                self.api_base = api_base
            elif isinstance(api_base, list):
                for api in api_base:
                    assert api.startswith('http')
                self.api_base = api_base
            else:
                self.logger.error(f'Unknown API Base. {api_base}')
                raise NotImplementedError
        else:
            self.api_base = api_base

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
                    from PIL import Image
                    fmt=get_image_fmt_from_image_path(msg['value'])
                    img = Image.open(msg['value'])
                    b64 = custom_encode_image_to_base64(img, target_size=self.img_size, fmt=fmt)
                    if self.model == os.environ.get("DOUBAO_MODEL_NAME"):
                        b64 = compress_image(b64, format=fmt)
                    img_struct = dict(url=f'data:image/{fmt};base64,{b64}')
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
        if os.environ.get('ADDITIONAL_MESSAGE_CONTENT', None) is not None:
            input_msgs.append(
                {"role": "assistant", "content":[{"type":"text", "text":os.environ.get('ADDITIONAL_MESSAGE_CONTENT')}]}
            )
        return input_msgs

    def _next_api_base(self):
        # 线程安全轮询
        if isinstance(self.api_base, str):
            return self.api_base
        with self.cur_idx_lock:
            _api_base = self.api_base[self.cur_idx]
            self.cur_idx = (self.cur_idx + 1) % len(self.api_base)
        print(f"USING {_api_base}")
        return _api_base

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # Will send request if use Azure, dk how to use openai client for it
        if self.key is None:
            headers = {'Content-Type': 'application/json'}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            temperature=temperature,
            **kwargs)
        if temperature == 0:
            payload.pop('temperature')
            payload['top_k'] = 1
        if self.model == os.environ.get("DOUBAO_MODEL_NAME") and self.think_mode is False:
            payload['thinking'] = {"type" : "disabled"}
        else:
            payload['thinking'] = {"type" : "enabled"}
            if  self.reasoning_effort is not None:
                payload['reasoning_effort'] = self.reasoning_effort
        payload['max_tokens'] = max_tokens

        if 'gemini' in self.model:
            payload.pop('max_tokens')
            payload.pop('n')
            payload['reasoning_effort'] = 'high'
        
        if self.model == 'openai/gpt-oss-120b':
            payload['reasoning_effort'] = self.reasoning_effort

        try_times = 0
        while try_times < 3:
            response = requests.post(
                self._next_api_base(),
                headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
            ret_code = response.status_code
            ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
            answer = self.fail_msg
            try:
                resp_struct = json.loads(response.text)
                answer = resp_struct['choices'][0]['message']['content'].strip()
                if answer == '':
                    answer = resp_struct['choices'][0]['message']['reasoning_content'].strip()
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'{type(err)}: {err}')
                    self.logger.error(response.text if hasattr(response, 'text') else response)

            return ret_code, answer, response

XHSVLMAPIWrapper = VLLMAPIWrapper

class VLLMAPI(VLLMAPIWrapper):

    def generate(self, message, dataset=None):
        return super(VLLMAPI, self).generate(message)

class XHSVLMAPI(XHSVLMAPIWrapper):
    """内部API"""
    def generate(self, message, dataset=None):
        return super(XHSVLMAPI, self).generate(message)

class XHSSEEDVL(VLLMAPIWrapper):
    """内部API"""

    def __init__(self, model: str = None, key: str = None, api_base: str = None, think_mode = True, **kwargs):
        assert model is None and key is None and api_base is None, "使用环境变量设置"
        model=os.environ.get("DOUBAO_MODEL_NAME", None)
        key=os.environ.get("DOUBAO_VL_KEY", None)
        api_base=os.environ.get("DOUBAO_API_BASE", None)
        
        assert model is not None and key is not None and api_base is not None, (
            "使用环境变量设置 `DOUBAO_MODEL_NAME=`, `DOUBAO_VL_KEY=`, `DOUBAO_API_BASE=`")
        self.think_mode = think_mode

        super(XHSSEEDVL,self).__init__(model=model, key=key, api_base=api_base, **kwargs)
        
