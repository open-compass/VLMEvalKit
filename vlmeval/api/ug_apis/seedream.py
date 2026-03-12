import base64
import os
from io import BytesIO

import httpx
import requests
from PIL import Image

from ..base import BaseAPI


class Seedream4ImageWrapper(BaseAPI):

    SUPPORT_GEN = True
    EXPERTISE = ['T2I', 'TI2I', 'TI2TI']

    def __init__(
        self,
        model: str | None = None,
        retry: int = 5,
        key: str | None = None,
        verbose: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0,
        timeout: int = 600,
        api_base: str = 'https://ark-cn-beijing.bytedance.net/api/v3',
        size: str = '1k',
        response_format: str = 'url',
        watermark: bool = False,
        sequential_image_generation: str = 'disabled',
        **kwargs,
    ):
        # self.model = 'ep-20250929193538-7pcfk'
        self.model = model
        assert isinstance(self.model, str) and len(self.model), (
            'Seedream4 requires an Ark endpoint ID. Set env `SEEDREAM4_ENDPOINT` or pass `model=`.'
        )

        self.fail_msg = 'Failed to obtain answer via API. '
        self.temperature = temperature
        self.timeout = timeout
        self.api_base = api_base
        self.size = size
        self.response_format = response_format
        self.watermark = watermark
        self.sequential_image_generation = sequential_image_generation

        env_key = os.environ.get('ARK_API_KEY', '')
        key = 'YOUR_API_KEY'
        if key is None:
            key = env_key
        assert isinstance(key, str) and len(key), 'Missing Ark API key: set env `ARK_API_KEY` or pass `key=`.'
        self.key = key

        try:
            from volcenginesdkarkruntime import Ark
        except Exception as e:
            raise ImportError(
                'Missing dependency `volcenginesdkarkruntime` required for Seedream4.'
            ) from e

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.key,
            timeout=httpx.Timeout(timeout=self.timeout),
            base_url=self.api_base,
        )

        # self.client = Ark(
        #     api_key=self.key,
        #     timeout=httpx.Timeout(timeout=self.timeout),
        #     base_url=self.api_base,
        # )
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    @staticmethod
    def _image_to_data_url(img_or_path) -> str:
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path)
        else:
            img = img_or_path

        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        fmt = (img.format or 'PNG').upper()
        if fmt == 'JPG':
            fmt = 'JPEG'

        buf = BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        mime = fmt.lower()
        if mime == 'jpeg':
            mime = 'jpeg'
        return f'data:image/{mime};base64,{b64}'

    @staticmethod
    def _download_image(url: str) -> Image.Image:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))

    def generate_inner(self, message, **kwargs):
        prompt_parts = []
        ref_images = []
        for msg in message:
            if msg['type'] == 'text':
                prompt_parts.append(msg['value'])
            elif msg['type'] == 'image':
                ref_images.append(msg['value'])

        prompt = ''.join(prompt_parts).strip()
        if not prompt:
            return -1, self.fail_msg, 'Empty prompt'

        extra = {
            'watermark': self.watermark,
            'sequential_image_generation': self.sequential_image_generation,
        }

        if len(ref_images):
            extra['image'] = [self._image_to_data_url(p) for p in ref_images]

        size = kwargs.pop('size', self.size)
        response_format = kwargs.pop('response_format', self.response_format)

        try:
            resp = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size,
                response_format=response_format,
                extra_body=extra,
            )
            if not resp.data:
                return -1, self.fail_msg, {
                    'error': 'empty response', 'resp': getattr(resp, 'model_dump', lambda: str(resp))()}

            item = resp.data[0]
            if response_format == 'url':
                url = getattr(item, 'url', None)
                if not url:
                    return -1, self.fail_msg, resp
                image = self._download_image(url)
                return 0, image, resp
            elif response_format == 'b64_json':
                b64 = getattr(item, 'b64_json', None)
                if not b64:
                    return -1, self.fail_msg, resp
                img_bytes = base64.b64decode(b64)
                image = Image.open(BytesIO(img_bytes))
                return 0, image, resp
            else:
                return -1, self.fail_msg, {'error': f'unsupported response_format={response_format}', 'resp': resp}
        except Exception as e:
            return -1, f'{self.fail_msg}{type(e)} {str(e)}', ''


class SeedreamImage(Seedream4ImageWrapper):

    def __init__(self, model='ep-20251224010846-qwmqh', **kwargs):
        super().__init__(model=model, **kwargs)

    def generate(self, message, dataset=None):
        return super().generate(message)
