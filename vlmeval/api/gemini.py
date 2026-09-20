import os
import time

from PIL import Image

from vlmeval.api.base import BaseAPI
from vlmeval.smp import get_logger, normalize_audio, proxy_set, resolve_media_source
from vlmeval.smp.audio import validate_audio_size_limit

headers = 'Content-Type: application/json'


logger = get_logger(__name__)


class GeminiWrapper(BaseAPI):
    """Wrapper for Google's ``google-genai`` GenerateContent API.

    Audio is deliberately opt-in.  The public Gemini aliases which have been
    checked against the model capability table pass ``audio_input=True``;
    existing image/video aliases retain the default text/image/video contract.
    """

    is_api: bool = True
    # The inline limit is an engineering limit, below the API request limit, so
    # that prompt and JSON/base64 overhead cannot unexpectedly exceed it.
    DEFAULT_AUDIO_INLINE_MAX_FILE_SIZE = 16 * 1024 ** 2
    DEFAULT_AUDIO_MAX_FILE_SIZE = 2 * 1024 ** 3 - 1
    DEFAULT_AUDIO_UPLOAD_TIMEOUT = 300.0
    DEFAULT_AUDIO_UPLOAD_POLL_INTERVAL = 2.0

    def __init__(self,
                 model: str = 'gemini-1.0-pro',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 2048,
                 proxy: str = None,
                 backend='genai',
                 project_id='vlmeval',
                 thinking_budget: int = None,  # range from 0 to 24576
                 # see https://ai.google.dev/gemini-api/docs/thinking
                 fps: int = 1,
                 media_resolution: str = None,
                 audio_input: bool = False,
                 audio_inline_max_file_size: int = DEFAULT_AUDIO_INLINE_MAX_FILE_SIZE,
                 audio_max_file_size: int = DEFAULT_AUDIO_MAX_FILE_SIZE,
                 audio_upload_timeout: float = DEFAULT_AUDIO_UPLOAD_TIMEOUT,
                 audio_upload_poll_interval: float = DEFAULT_AUDIO_UPLOAD_POLL_INTERVAL,
                 audio_cache_dir: str = None,
                 **kwargs):
        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.fps = fps
        self.audio_input = bool(audio_input)
        self.audio_cache_dir = audio_cache_dir
        self.audio_inline_max_file_size = validate_audio_size_limit(
            audio_inline_max_file_size
        )
        self.audio_max_file_size = validate_audio_size_limit(audio_max_file_size)
        if audio_upload_timeout is None:
            audio_upload_timeout = self.DEFAULT_AUDIO_UPLOAD_TIMEOUT
        if isinstance(audio_upload_timeout, bool) or not isinstance(
            audio_upload_timeout, (int, float)
        ) or audio_upload_timeout < 0:
            raise ValueError('audio_upload_timeout must be a non-negative number.')
        if isinstance(audio_upload_poll_interval, bool) or not isinstance(
            audio_upload_poll_interval, (int, float)
        ) or audio_upload_poll_interval < 0:
            raise ValueError(
                'audio_upload_poll_interval must be a non-negative number.'
            )
        self.audio_upload_timeout = float(audio_upload_timeout)
        self.audio_upload_poll_interval = float(audio_upload_poll_interval)

        self.media_resolution = media_resolution
        if self.media_resolution:
            assert self.media_resolution in ['low', 'medium', 'high']
        if key is None:
            key = os.environ.get('GOOGLE_API_KEY', None)
        # Try to load backend from environment variable
        be = os.environ.get('GOOGLE_API_BACKEND', None)
        if be is not None and be in ['genai', 'vertex']:
            backend = be

        assert backend in ['genai', 'vertex']
        if backend == 'genai':
            assert key is not None
            try:
                from google import genai
                from google.genai import types
            except ImportError as e:
                raise ImportError(
                    "Could not import 'google.genai'. Please install it with:\n"
                    "    pip install --upgrade google-genai"
                ) from e
            self.media_resolution_dict = {
                'low': types.MediaResolution.MEDIA_RESOLUTION_LOW,
                'medium': types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                'high': types.MediaResolution.MEDIA_RESOLUTION_HIGH
            }
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=key)

        self.backend = backend
        self.project_id = project_id
        self.api_key = key

        # Capability is instance-scoped.  Do not add audio to BaseAPI's global
        # allow-list: unsupported Gemini aliases must fail before any request.
        self.allowed_types = list(BaseAPI.allowed_types)
        if self.backend == 'genai' and self.audio_input:
            self.allowed_types.append('audio')

        if proxy is not None:
            proxy_set(proxy)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        logger.info(
            f'Using provider=Gemini, model={self.model}, backend={self.backend}, '
            f'audio_mode={"native" if "audio" in self.allowed_types else "unsupported"}'
        )

    def upload_media_genai(self, media_path, mime_type=None, media_type=None):
        """Upload media, wait until it is usable, and return a URI part."""
        upload = self.client.files.upload
        upload_config = getattr(self.types, 'UploadFileConfig', None)
        config = upload_config(mime_type=mime_type) if mime_type and upload_config else None
        if config is None:
            myfile = upload(file=media_path)
        else:
            myfile = upload(file=media_path, config=config)

        name = getattr(myfile, 'name', None)
        deadline = time.monotonic() + self.audio_upload_timeout
        poll_delay = 0.0
        while hasattr(myfile, 'state'):
            state = getattr(myfile, 'state', None)
            state = getattr(state, 'name', getattr(state, 'value', state))
            state = str(state).rsplit('.', 1)[-1].upper() if state is not None else None
            if state in (None, 'ACTIVE'):
                break
            if (
                'FAIL' in state or 'ERROR' in state or 'CANCEL' in state
                or state in {'EXPIRED', 'DELETED'}
            ):
                raise RuntimeError(f'Gemini media upload failed for {name!r} (state={state}).')
            if not name:
                raise RuntimeError('Gemini media upload returned no file name.')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f'Timed out waiting {self.audio_upload_timeout:g}s for '
                    f'Gemini media upload {name!r} to become ACTIVE.'
                )
            if poll_delay:
                time.sleep(min(poll_delay, remaining))
            myfile = self.client.files.get(name=name)
            poll_delay = self.audio_upload_poll_interval

        uri = getattr(myfile, 'uri', None)
        if not uri:
            raise RuntimeError('Gemini media upload returned no file URI.')
        uploaded_mime = (
            getattr(myfile, 'mime_type', None)
            or getattr(myfile, 'mimeType', None)
            or mime_type
        )
        if not uploaded_mime:
            raise RuntimeError('Gemini media upload returned no MIME type.')
        part = self.types.Part.from_uri(file_uri=uri, mime_type=uploaded_mime)
        if media_type == 'video':
            part.video_metadata = self.types.VideoMetadata(fps=self.fps)
        return part

    def upload_video_genai(self, video_path):
        return self.upload_media_genai(
            video_path, mime_type='video/mp4', media_type='video'
        )

    def preproc_content(self, inputs):
        """Apply Gemini's audio capability and one-file constraints."""
        content_type = self.check_content(inputs)
        if content_type in ('dict', 'listdict'):
            items = [inputs] if content_type == 'dict' else inputs
            audio_count = sum(item.get('type') == 'audio' for item in items)
            if audio_count > 1:
                raise ValueError(
                    'Gemini native audio supports at most one audio item per request.'
                )
            if audio_count and self.backend != 'genai':
                raise ValueError(
                    f"Gemini backend '{self.backend}' does not support native audio input."
                )
            if audio_count and not self.audio_input:
                raise ValueError(
                    f"Gemini model '{self.model}' was not configured with audio_input=True."
                )
        return super().preproc_content(inputs)

    def build_msgs_genai(self, inputs):
        """Build ordered GenerateContent parts, including native audio."""
        assert isinstance(inputs, list)
        video_in_msg = False
        messages = [] if self.system_prompt is None else [self.system_prompt]
        audio_count = 0

        for inp in inputs:
            item_type = inp['type']
            if item_type == 'text':
                messages.append(inp['value'])
            elif item_type == 'image':
                messages.append(Image.open(inp['value']))
            elif item_type == 'video':
                messages.append(self.upload_video_genai(inp['value']))
                video_in_msg = True
            elif item_type == 'audio':
                audio_count += 1
                if audio_count > 1:
                    raise ValueError(
                        'Gemini native audio supports at most one audio item per request.'
                    )
                local_path = resolve_media_source(
                    inp['value'], max_file_size=self.audio_max_file_size
                )
                payload = normalize_audio(
                    local_path,
                    target_format=None,
                    cache_dir=self.audio_cache_dir,
                    max_file_size=self.audio_max_file_size,
                )
                if (
                    self.audio_inline_max_file_size is not None
                    and len(payload.data) <= self.audio_inline_max_file_size
                ):
                    messages.append(self.types.Part.from_bytes(
                        data=payload.data, mime_type=payload.mime_type
                    ))
                else:
                    messages.append(self.upload_media_genai(
                        payload.source, mime_type=payload.mime_type, media_type='audio'
                    ))
            else:
                raise ValueError(f'Gemini GenAI does not support input type: {item_type!r}.')

        return messages, video_in_msg

    def build_msgs_vertex(self, inputs):
        from vertexai.generative_models import Image, Part
        messages = [] if self.system_prompt is None else [self.system_prompt]
        for inp in inputs:
            if inp['type'] == 'text':
                messages.append(inp['value'])
            elif inp['type'] == 'image':
                messages.append(Part.from_image(Image.load_from_file(inp['value'])))
            elif inp['type'] == 'audio':
                raise ValueError(
                    "Gemini backend 'vertex' does not support native audio input."
                )
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        if self.backend == 'genai':
            assert isinstance(inputs, list)
            model = self.model
            messages, video_in_msg = self.build_msgs_genai(inputs)

            # Configure generation parameters
            config_args = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
            # set resolution for vision input
            if self.media_resolution:
                if video_in_msg:
                    assert self.media_resolution != 'high', (
                        'For video input, only support medium and low resolution'
                    )
                config_args["media_resolution"] = self.media_resolution_dict[
                    self.media_resolution
                ]

            # If thinking_budget is specified, add thinking_config
            if self.thinking_budget is not None:
                config_args["thinking_config"] = self.types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            config_args.update(kwargs)

            try:
                resp = self.client.models.generate_content(
                    model=model,
                    contents=messages,
                    config=self.types.GenerateContentConfig(**config_args)
                )
                answer = resp.text
                return 0, answer, 'Succeeded! '
            except Exception as err:
                if self.verbose:
                    logger.error(f'{type(err)}: {err}')
                    logger.error(f'The input messages are {inputs}.')

                return -1, '', ''
        elif self.backend == 'vertex':
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=self.project_id, location='us-central1')
            model_name = 'gemini-1.0-pro-vision' if self.model == 'gemini-1.0-pro' else self.model
            model = GenerativeModel(model_name=model_name)
            messages = self.build_msgs_vertex(inputs)
            try:
                resp = model.generate_content(messages)
                answer = resp.text
                return 0, answer, 'Succeeded! '
            except Exception as err:
                if self.verbose:
                    logger.error(f'{type(err)}: {err}')
                    logger.error(f'The input messages are {inputs}.')

                return -1, '', ''


class Gemini(GeminiWrapper):
    VIDEO_LLM = True

    def generate(self, message, dataset=None):
        return super(Gemini, self).generate(message)
