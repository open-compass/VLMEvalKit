import math
import os

import numpy as np

from ..smp import encode_image_to_base64, get_logger
from .base import BaseAPI

logger = get_logger(__name__)

OFFICIAL_BASE_URL = 'https://api.anthropic.com'
MAX_IMAGE_DIMENSION = 8000
MAX_IMAGE_DIMENSION_FOR_MANY_IMAGES = 3000


def _normalize_claude_base_url(api_base):
    if api_base in (None, '', 'OFFICIAL'):
        return OFFICIAL_BASE_URL

    api_base = api_base.rstrip('/')
    for suffix in ('/messages', '/responses', '/chat/completions', '/completions'):
        if api_base.endswith(suffix):
            api_base = api_base[:-len(suffix)]
            break
    if api_base.endswith('/v1'):
        api_base = api_base[:-len('/v1')]
    return api_base


class ClaudeSDKWrapper(BaseAPI):
    """Anthropic Messages API wrapper using the official Anthropic Python SDK.

    ``thinking`` accepts the SDK's thinking configuration dictionary. ``effort``
    is sent as ``output_config.effort`` without changing the thinking mode.

    ``convert_rgb=False`` preserves image mode and transparency using PNG
    (JPEG for CMYK). Set it to ``True`` for the legacy RGB/JPEG encoding.
    """

    is_api: bool = True

    def __init__(
        self,
        model: str = 'claude-sonnet-5-5',
        retry: int = 5,
        wait: int = 5,
        key: str = None,
        verbose: bool = True,
        timeout: int = 300,
        api_base: str = None,
        system_prompt: str = None,
        stream: bool = False,
        img_size: int = -1,
        total_img_size: int = -1,
        max_file_size: int = 1e9,
        custom_prompt=None,
        model_adapter=None,
        thinking: dict = None,
        effort: str = None,
        convert_rgb: bool = False,
        **kwargs,
    ):
        for unsupported in ('video_llm', 'local_media'):
            if kwargs.pop(unsupported, False):
                logger.warning(
                    f'ClaudeSDKWrapper ignores unsupported argument `{unsupported}`.'
                )

        self.model = model
        self.stream = stream
        self.timeout = timeout
        self.thinking = thinking
        self.effort = effort

        self.key = key or os.environ.get('ANTHROPIC_API_KEY')
        assert self.key, 'Please set `key` or the ANTHROPIC_API_KEY environment variable.'
        self.api_base = _normalize_claude_base_url(api_base or OFFICIAL_BASE_URL)

        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert total_img_size > 0 or total_img_size == -1
        self.total_img_size = total_img_size
        self.max_file_size = max_file_size
        self.convert_rgb = convert_rgb

        self.adapter = None
        adapter = custom_prompt if custom_prompt is not None else model_adapter
        if adapter is not None:
            if isinstance(adapter, str):
                from .adapters import build_adapter
                self.adapter = build_adapter(adapter)
            else:
                self.adapter = adapter

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

        self.client = self._create_fresh_client()
        logger.info(f'ClaudeSDKWrapper: model={self.model}')

    def set_dump_image(self, dump_image_func):
        if self.adapter is not None:
            self.adapter.dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset) -> bool:
        if self.adapter is not None:
            return self.adapter.use_custom_prompt(dataset, self.system_prompt)
        return False

    def build_prompt(self, line, dataset=None):
        if self.adapter is not None:
            return self.adapter.build_prompt(line, dataset)
        raise NotImplementedError

    def _create_fresh_client(self):
        from anthropic import Anthropic

        return Anthropic(
            base_url=self.api_base,
            api_key=self.key,
            timeout=self.timeout,
            max_retries=0,
        )

    def _get_image_target_size(self, image_num):
        image_num = max(image_num, 1)
        target_size = math.inf
        if self.img_size > 0:
            target_size = self.img_size
        if self.total_img_size > 0:
            target_size = min(
                target_size,
                max(1, int(self.total_img_size / (image_num ** 0.5))),
            )
        return -1 if math.isinf(target_size) else target_size

    def prepare_itlist(self, inputs, max_image_dimension=None):
        assert np.all([isinstance(x, dict) for x in inputs])
        if any(x['type'] == 'video' for x in inputs):
            raise NotImplementedError('Claude SDK wrapper does not support video inputs.')

        image_num = sum(x['type'] == 'image' for x in inputs)
        if not image_num:
            assert all(x['type'] == 'text' for x in inputs)
            return [dict(type='text', text='\n'.join(x['value'] for x in inputs))]

        from PIL import Image

        content_list = []
        target_size = self._get_image_target_size(image_num)
        dimension_limit = max_image_dimension or (
            MAX_IMAGE_DIMENSION_FOR_MANY_IMAGES
            if image_num > 20
            else MAX_IMAGE_DIMENSION
        )
        if target_size < 0 or target_size > dimension_limit:
            target_size = dimension_limit
        for msg in inputs:
            if msg['type'] == 'text' and msg['value'].strip():
                content_list.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                with Image.open(msg['value']) as image:
                    fmt = 'JPEG' if self.convert_rgb or image.mode == 'CMYK' else 'PNG'
                    b64 = encode_image_to_base64(
                        image,
                        target_size=target_size,
                        fmt=fmt,
                        max_file_size=self.max_file_size,
                        convert_rgb=self.convert_rgb,
                    )
                content_list.append(
                    dict(
                        type='image',
                        source=dict(
                            type='base64',
                            media_type=Image.MIME[fmt],
                            data=b64,
                        ),
                    )
                )
        return content_list

    def prepare_inputs(self, inputs, system_prompt=None):
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert (
            np.all(['type' in x for x in inputs])
            or np.all(['role' in x for x in inputs])
        ), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            image_num = sum(
                part['type'] == 'image'
                for message in inputs
                if message['role'] != 'system'
                for part in message['content']
            )
            max_image_dimension = (
                MAX_IMAGE_DIMENSION_FOR_MANY_IMAGES
                if image_num > 20
                else MAX_IMAGE_DIMENSION
            )
            system_messages = [x for x in inputs if x['role'] == 'system']
            if system_messages:
                system_content = '\n'.join(
                    part['value']
                    for item in system_messages
                    for part in item['content']
                    if part['type'] == 'text'
                )
                system_prompt = '\n'.join(
                    filter(None, (system_prompt, system_content))
                ) or None
            messages = [x for x in inputs if x['role'] != 'system']
            result = [
                dict(
                    role=x['role'],
                    content=self.prepare_itlist(x['content'], max_image_dimension),
                )
                for x in messages
            ]
        else:
            result = [
                dict(
                    role='user',
                    content=self.prepare_itlist(inputs),
                )
            ]
        return result, system_prompt

    def generate_inner(self, inputs, dataset=None, system_prompt=None, **kwargs) -> tuple:
        if self.adapter is not None:
            model_args = self.adapter.override_model_args(dataset, kwargs)
            system_prompt = model_args.pop('system_prompt', system_prompt)
            inputs = self.adapter.process_inputs(inputs, dataset)
            kwargs.update(model_args)

        stream = kwargs.pop('stream', self.stream)
        thinking = kwargs.pop('thinking', self.thinking)
        effort = kwargs.pop('effort', self.effort)
        max_tokens = kwargs.pop(
            'max_tokens', self.default_kwargs.get('max_tokens', 16384)
        )
        messages, system_prompt = self.prepare_inputs(inputs, system_prompt)
        payload = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        if system_prompt is not None:
            payload['system'] = system_prompt
        if thinking is not None:
            payload['thinking'] = thinking
        if effort is not None:
            payload['output_config'] = {
                **(payload.get('output_config') or {}),
                'effort': effort,
            }

        if self.adapter is not None:
            payload = self.adapter.process_payload(payload, dataset=dataset)

        try:
            if stream:
                with self.client.messages.stream(**payload) as message_stream:
                    response = message_stream.get_final_message()
            else:
                response = self.client.messages.create(**payload)
            answer = ''.join(
                block.text for block in response.content if block.type == 'text'
            ).strip()
            usage = response.usage
            usage_details = getattr(usage, 'output_tokens_details', None)
            if isinstance(usage, dict):
                usage_details = usage.get('output_tokens_details')
            if isinstance(usage_details, dict):
                reasoning_tokens = usage_details.get('thinking_tokens')
            else:
                reasoning_tokens = getattr(usage_details, 'thinking_tokens', None)
            input_tokens = (
                usage.get('input_tokens')
                if isinstance(usage, dict)
                else usage.input_tokens
            )
            output_tokens = (
                usage.get('output_tokens')
                if isinstance(usage, dict)
                else usage.output_tokens
            )
            if reasoning_tokens is None:
                reasoning_tokens = 'unavailable (not reported by API)'
            logger.info(
                f'Finish reason: {response.stop_reason}; input_tokens={input_tokens}; '
                f'output_tokens={output_tokens}; reasoning_tokens={reasoning_tokens}'
            )
            if response.stop_reason not in ('end_turn', 'stop_sequence'):
                logger.warning(
                    f'Claude response may be incomplete: {response.stop_reason}'
                )
            if self.adapter is not None:
                answer = self.adapter.postprocess(answer, dataset=dataset)
            return 0, answer, response.model_dump_json()
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err).__name__}: {err}')
            return -1, self.fail_msg, str(err)


class ClaudeSDKAPI(ClaudeSDKWrapper):

    def generate(self, message, dataset=None, **kwargs):
        return super().generate(message, dataset=dataset, **kwargs)
