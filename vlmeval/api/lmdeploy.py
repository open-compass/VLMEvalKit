import base64
import os
from pathlib import Path

import numpy as np

from ..smp import encode_image_to_base64, get_logger
from .adapters import build_adapter
from .openai_sdk import OpenAISDKWrapper

logger = get_logger(__name__)


class LMDeployWrapper(OpenAISDKWrapper):
    """OpenAI-compatible wrapper for lmdeploy-served models.

    Handles lmdeploy-specific details:

    * ``LMDEPLOY_API_KEY`` / ``LMDEPLOY_API_BASE`` environment variables
    * lmdeploy image-encoding format via :meth:`prepare_itlist`

    Model-specific prompt building and payload post-processing are
    delegated to a :class:`~vlmeval.api.adapters.ModelAdapter` that is
    selected automatically or specified via ``custom_prompt``.
    """

    def __init__(self,
                 model,
                 retry=5,
                 wait=5,
                 key='sk-123456',
                 verbose=True,
                 timeout=120,
                 api_base=None,
                 system_prompt=None,
                 custom_prompt=None,
                 video_llm: bool = False,
                 local_media: bool = False,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.timeout = timeout

        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = api_base or os.environ.get('LMDEPLOY_API_BASE', None)
        assert key is not None, 'Please set the environment variable LMDEPLOY_API_KEY.'
        assert api_base, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.key = key
        self.api_base = api_base

        self.VIDEO_LLM = video_llm
        self.local_media = local_media or (os.getenv('VLMEVAL_LOCAL_MEDIA', '0') == '1')
        if self.local_media:
            logger.info(
                f'lmdeploy: `local_media={self.local_media}`, pass local media file path directly.')
        else:
            logger.info(
                f'lmdeploy: `local_media={self.local_media}`, pass media file base64.')

        kwargs = {'max_tokens': 16384, 'temperature': 0.0, **kwargs}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Call OpenAISDKWrapper without model_adapter; we set self.adapter below
        # after resolving the model name.
        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

        self.model = model
        logger.info(f'lmdeploy evaluate model: {self.model}')

        # Resolve and instantiate adapter
        if custom_prompt is not None:
            self.adapter = build_adapter(custom_prompt)
            logger.info(f'Using model adapter: {custom_prompt}')

    # ------------------------------------------------------------------
    # HTTP message formatting (lmdeploy-specific)
    # ------------------------------------------------------------------

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        multimedia = sum(x['type'] in ('image', 'video') for x in inputs)
        if multimedia:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    if self.local_media:
                        image_data_url = f"file://{Path(msg['value']).resolve()}"
                    else:
                        from PIL import Image
                        img = Image.open(msg['value'])
                        b64 = encode_image_to_base64(img)
                        image_data_url = f'data:image/jpeg;base64,{b64}'
                    extra_args = {k: v for k, v in msg.items() if k not in ('type', 'value')}
                    img_struct = dict(url=image_data_url, **extra_args)
                    content_list.append(dict(type='image_url', image_url=img_struct))
                elif msg['type'] == 'video':
                    if self.local_media:
                        video_data_url = f"file://{Path(msg['value']).resolve()}"
                    else:
                        with open(msg['value'], 'rb') as f:
                            video_b64 = base64.b64encode(f.read()).decode("utf-8")
                        ext = Path(msg['value']).suffix
                        mime_map = {
                            ".mp4": "video/mp4", ".avi": "video/avi", ".mkv": "video/x-matroska",
                            ".mov": "video/quicktime", ".webm": "video/webm", ".flv": "video/x-flv",
                        }
                        mime = mime_map.get(ext, "video/mp4")
                        video_data_url = f"data:{mime};base64,{video_b64}"
                    extra_args = {k: v for k, v in msg.items() if k not in ('type', 'value')}
                    vid_struct = dict(url=video_data_url, **extra_args)
                    content_list.append(dict(type='video_url', video_url=vid_struct))
        else:
            assert all(x['type'] == 'text' for x in inputs)
            text = '\n'.join(x['value'] for x in inputs)
            content_list = [dict(type='text', text=text)]
        return content_list

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


class LMDeployAPI(LMDeployWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super().generate(message, dataset=dataset)
