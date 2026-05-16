"""
LiteLLM API provider for VLMEvalKit — unified interface to 100+ LLM providers.

Requires: pip install 'litellm>=1.55,<1.85'
Set provider-specific env vars (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY)
or pass key= for providers that accept a single key.
"""
import os

import numpy as np

from ..smp import encode_image_to_base64, get_logger
from .base import BaseAPI

logger = get_logger(__name__)

try:
    import litellm
except ImportError:
    litellm = None


class LiteLLMAPI(BaseAPI):
    """VLM/LLM API using LiteLLM (https://docs.litellm.ai/docs/providers).

    Model strings follow LiteLLM conventions:
        - OpenAI:     "gpt-4o", "gpt-4o-mini"
        - Anthropic:  "anthropic/claude-sonnet-4-20250514"
        - Azure:      "azure/<deployment-name>"
        - Bedrock:    "bedrock/anthropic.claude-3-sonnet"
        - Vertex AI:  "vertex_ai/gemini-1.5-pro"
        - Together:   "together_ai/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
    """

    is_api: bool = True

    def __init__(
        self,
        model: str = 'gpt-4o',
        key: str = None,
        api_base: str = None,
        retry: int = 10,
        wait: int = 1,
        system_prompt: str = None,
        verbose: bool = True,
        temperature: float = 0,
        max_tokens: int = 2048,
        timeout: int = 300,
        img_size: int = -1,
        **kwargs,
    ):
        self.model = model
        self.api_key = key or os.environ.get('LITELLM_API_KEY', None)
        self.api_base = api_base or os.environ.get('LITELLM_API_BASE', None)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.img_size = img_size
        self.litellm_kwargs = kwargs.pop('litellm_kwargs', {})

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

        logger.info(f'LiteLLMAPI: model={self.model}')

    def _prepare_content(self, inputs):
        assert all(isinstance(x, dict) for x in inputs)
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for item in inputs:
                if item['type'] == 'text' and item['value']:
                    content_list.append({'type': 'text', 'text': item['value']})
                elif item['type'] == 'image':
                    from PIL import Image
                    img = Image.open(item['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    content_list.append({
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                    })
            return content_list
        text = '\n'.join([x['value'] for x in inputs if x['type'] == 'text'])
        return [{'type': 'text', 'text': text or ''}]

    def _prepare_messages(self, inputs):
        out = []
        if self.system_prompt:
            out.append({'role': 'system', 'content': self.system_prompt})
        if inputs and 'role' in inputs[0]:
            for item in inputs:
                out.append({
                    'role': item['role'],
                    'content': self._prepare_content(item['content']),
                })
        else:
            out.append({'role': 'user', 'content': self._prepare_content(inputs)})
        return out

    def generate_inner(self, inputs, **kwargs):
        if litellm is None:
            raise ImportError(
                "LiteLLM is required for LiteLLMAPI. "
                "Install it with: pip install 'litellm>=1.55,<1.85'"
            )

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        messages = self._prepare_messages(inputs)

        completion_kwargs = {
            **self.litellm_kwargs,
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'timeout': self.timeout,
            'drop_params': True,
        }
        if self.api_key:
            completion_kwargs['api_key'] = self.api_key
        if self.api_base:
            completion_kwargs['api_base'] = self.api_base

        try:
            response = litellm.completion(**completion_kwargs)
            answer = response.choices[0].message.content.strip()
            return 0, answer, response
        except Exception as err:
            if self.verbose:
                logger.error(f'{type(err).__name__}: {err}')
            return -1, self.fail_msg, str(err)
