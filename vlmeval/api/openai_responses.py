import json
import math
import os

import httpx
import numpy as np

from ..smp import encode_image_to_base64, get_logger
from .base import BaseAPI

logger = get_logger(__name__)

OFFICIAL_BASE_URL = 'https://api.openai.com/v1'


def _normalize_responses_base_url(api_base):
    if api_base in (None, '', 'OFFICIAL'):
        return OFFICIAL_BASE_URL

    api_base = api_base.rstrip('/')
    for suffix in ('/responses', '/chat/completions', '/completions'):
        if api_base.endswith(suffix):
            return api_base[:-len(suffix)]
    return api_base


def _get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_response_text(response):
    output_text = _get_attr(response, 'output_text')
    if output_text:
        return output_text.strip()

    chunks = []
    for item in _get_attr(response, 'output', []) or []:
        for part in _get_attr(item, 'content', []) or []:
            part_type = _get_attr(part, 'type')
            if part_type in ('output_text', 'text'):
                text = _get_attr(part, 'text', '')
                if text:
                    chunks.append(text)
    return ''.join(chunks).strip()


def _extract_finish_reason(response):
    if response is None:
        return None

    finish_reason = _get_attr(response, 'finish_reason')
    if finish_reason:
        return finish_reason

    incomplete_details = _get_attr(response, 'incomplete_details')
    reason = _get_attr(incomplete_details, 'reason')
    if reason:
        return reason

    choices = _get_attr(response, 'choices', []) or []
    if choices:
        reason = _get_attr(choices[0], 'finish_reason')
        if reason:
            return reason

    for item in _get_attr(response, 'output', []) or []:
        reason = _get_attr(item, 'finish_reason')
        if reason:
            return reason

    return _get_attr(response, 'status')


def _is_finished_successfully(finish_reason):
    return finish_reason in ('stop', 'completed')


def _sanitize_payload_for_log(payload):
    if isinstance(payload, dict):
        sanitized = {}
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in ('api_key', 'key', 'authorization'):
                sanitized[key] = '<redacted>'
            elif key_lower in ('input', 'message', 'messages'):
                sanitized[key] = '<omitted>'
            else:
                sanitized[key] = _sanitize_payload_for_log(value)
        return sanitized

    if isinstance(payload, list):
        return [_sanitize_payload_for_log(x) for x in payload]

    if isinstance(payload, str) and payload.startswith('data:'):
        media_type = payload[5:].split(';', 1)[0]
        return f'<{media_type} data omitted, length={len(payload)}>'

    return payload


class OpenAIResponsesWrapper(BaseAPI):
    """OpenAI Responses API wrapper using the official OpenAI Python SDK."""

    is_api: bool = True

    def __init__(
        self,
        model: str = 'gpt-4o',
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
        img_detail: str = 'auto',
        custom_prompt=None,
        model_adapter=None,
        **kwargs,
    ):
        for unsupported in ('video_llm', 'local_media'):
            if unsupported in kwargs:
                logger.warning(f'OpenAIResponsesWrapper ignores unsupported argument `{unsupported}`.')
                kwargs.pop(unsupported)

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.stream = stream
        self.timeout = timeout

        key = key or os.environ.get('OPENAI_API_KEY', None)
        assert key is not None and key != '', 'Please set the environment variable OPENAI_API_KEY.'
        self.key = key

        api_base = (
            api_base
            or os.environ.get('OPENAI_API_BASE', None)
            or os.environ.get('OPENAI_BASE_URL', None)
            or os.environ.get('OPENAI_API_URL', None)
            or os.environ.get('OPENAI_URL', None)
        )
        self.api_base = _normalize_responses_base_url(api_base)

        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert total_img_size > 0 or total_img_size == -1
        self.total_img_size = total_img_size
        self.max_file_size = max_file_size
        assert img_detail in ['high', 'low', 'auto']
        self.img_detail = img_detail

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
        logger.info(f'OpenAIResponsesWrapper: model={self.model}; api_base={self.api_base}')

    def _create_fresh_client(self):
        """Create an OpenAI SDK client for the Responses API."""
        from openai import OpenAI as OpenAIClient

        limits = httpx.Limits(
            max_keepalive_connections=2048,
            max_connections=4096,
        )
        http_client = httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            limits=limits,
        )
        return OpenAIClient(
            base_url=self.api_base,
            api_key=self.key,
            http_client=http_client,
        )

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

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        if any(x['type'] == 'video' for x in inputs):
            raise NotImplementedError('OpenAI Responses API wrapper does not support video inputs.')

        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            image_num = sum(x['type'] == 'image' for x in inputs)
            image_target_size = self._get_image_target_size(image_num)
            for msg in inputs:
                if msg['type'] == 'text' and msg['value'].strip():
                    content_list.append(dict(type='input_text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(
                        img,
                        target_size=image_target_size,
                        max_file_size=self.max_file_size,
                    )
                    img_struct = dict(
                        type='input_image',
                        image_url=f'data:image/jpeg;base64,{b64}',
                        detail=self.img_detail,
                    )
                    extra_args = {k: v for k, v in msg.items() if k not in ('type', 'value')}
                    img_struct.update(extra_args)
                    content_list.append(img_struct)
            return content_list

        assert all(x['type'] == 'text' for x in inputs)
        text = '\n'.join(x['value'] for x in inputs)
        return [dict(type='input_text', text=text)]

    def prepare_inputs(self, inputs, system_prompt):
        input_msgs = []
        if system_prompt is not None:
            input_msgs.append(dict(role='system', content=system_prompt))

        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert (
            np.all(['type' in x for x in inputs])
            or np.all(['role' in x for x in inputs])
        ), inputs

        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(
                    dict(role=item['role'], content=self.prepare_itlist(item['content']))
                )
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def _collect_streaming_response(self, response):
        answer_parts = []
        final_response = response
        for event in response:
            event_type = _get_attr(event, 'type')
            if event_type == 'response.output_text.delta':
                delta = _get_attr(event, 'delta', '')
                if delta:
                    answer_parts.append(delta)
            elif event_type == 'response.completed':
                final_response = _get_attr(event, 'response', event)

        answer = ''.join(answer_parts).strip()
        if not answer and final_response is not response:
            answer = _extract_response_text(final_response)
        return answer, final_response

    def generate_inner(self, inputs, dataset=None, **kwargs) -> tuple:
        if self.adapter is not None:
            model_args = self.adapter.override_model_args(dataset, kwargs)
            system_prompt = model_args.pop('system_prompt', self.system_prompt)
            inputs = self.adapter.process_inputs(inputs, dataset)
            kwargs.update(model_args)
        else:
            system_prompt = self.system_prompt

        input_msgs = self.prepare_inputs(inputs, system_prompt)
        temperature = kwargs.pop('temperature', self.default_kwargs.get('temperature', None))
        max_tokens = kwargs.pop('max_tokens', self.default_kwargs.get('max_tokens', None))
        stream = kwargs.pop('stream', self.stream)

        payload = dict(
            model=self.model,
            input=input_msgs,
            **kwargs,
        )
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens is not None and 'max_output_tokens' not in payload:
            payload['max_output_tokens'] = max_tokens
        if stream:
            payload['stream'] = True

        if self.adapter is not None:
            payload = self.adapter.process_payload(payload, dataset=dataset)

        response = None
        try:
            if self.verbose:
                log_payload = json.dumps(_sanitize_payload_for_log(payload), ensure_ascii=False)
                logger.info(f'Responses API request payload: {log_payload}')
            response = self.client.responses.create(**payload)
            if stream:
                answer, response = self._collect_streaming_response(response)
            else:
                answer = _extract_response_text(response)
            finish_reason = _extract_finish_reason(response)
            if self.verbose:
                logger.info(f'Finish reason: {finish_reason}')
            if not _is_finished_successfully(finish_reason):
                log = (
                    f'Finish reason indicates an incomplete response: {finish_reason}. '
                    f'Raw response: {response}'
                )
                logger.warning(log)
            if self.adapter is not None:
                answer = self.adapter.postprocess(answer, dataset=dataset)
            return 0, answer, response
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err).__name__}: {err}')
                logger.error(f'Finish reason: {_extract_finish_reason(response)}')
                logger.error(response.text if hasattr(response, 'text') else response)
            return -1, self.fail_msg, str(err)


class OpenAIResponsesAPI(OpenAIResponsesWrapper):

    def generate(self, message, dataset=None):
        return super().generate(message, dataset=dataset)
