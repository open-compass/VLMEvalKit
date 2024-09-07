from __future__ import annotations

import os

from vlmeval.api.base import BaseAPI
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


class QwenVLAPI0809(Qwen2VLPromptMixin, BaseAPI):
    is_api: bool = True

    def __init__(
        self,
        model: str = 'qwen-vl-max-0809',
        key: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_length=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        seed=3407,
        use_custom_prompt: bool = True,
        **kwargs,
    ):
        import dashscope

        self.model = model
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        key = os.environ.get('DASHSCOPE_API_KEY', None) if key is None else key
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        super().__init__(use_custom_prompt=use_custom_prompt, **kwargs)

    def _prepare_content(self, inputs: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, inputs, **kwargs) -> str:
        import dashscope

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(inputs)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # generate
        generation_kwargs = self.generate_kwargs.copy()
        kwargs.pop('dataset', None)
        generation_kwargs.update(kwargs)
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=messages,
                **generation_kwargs,
            )
            if self.verbose:
                print(response)
            answer = response.output.choices[0]['message']['content'][0]['text']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(err)
                self.logger.error(f'The input messages are {inputs}.')
            return -1, '', ''
