from __future__ import annotations
import inspect

from ..qwen2_vl.prompt import Qwen2VLPromptMixin
from .base import RBLNChatVLMBase


class RBLNQwen2VL(Qwen2VLPromptMixin, RBLNChatVLMBase):
    """optimum-rbln backend for Qwen2-VL / Qwen2.5-VL.

    Prompt construction mirrors upstream ``Qwen2VLChat`` exactly via
    ``Qwen2VLPromptMixin``.

    Qwen3-VL is handled by ``RBLNQwen3VL`` (which inherits this class
    but overrides the prompt mixin to ``Qwen3VLPromptMixin`` to match
    upstream ``Qwen3VLChat``). Architecture-based NPU model dispatch in
    ``_pick_rbln_class`` still keeps a Qwen3 branch as a safety net for
    direct instantiation, but the auto-dispatcher routes Qwen3-VL
    artifacts to ``RBLNQwen3VL`` so prompt mixin alignment is preserved.
    """

    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        fps: int | None = 2,
        nframe: int | None = 128,
        # Sampling defaults match upstream vlm.Qwen2VLChat (qwen2_vl/model.py:186-212).
        # The upstream wrapper uses near-greedy sampling (T=0.01, top_p=0.001)
        # rather than pure greedy — keep both backends emitting the same
        # distribution to make tie-broken tokens deterministic across GPU/RBLN.
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 0.01,
        top_p: float = 0.001,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> None:
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.nframe = nframe
        self.FRAME_FACTOR = 2
        super().__init__(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    def _pick_rbln_class(self):
        archs = self._read_architectures()
        if 'qwen3vl' in archs:
            from optimum.rbln import RBLNQwen3VLForConditionalGeneration
            return RBLNQwen3VLForConditionalGeneration
        if 'qwen2_5_vl' in archs:
            from optimum.rbln import RBLNQwen2_5_VLForConditionalGeneration
            return RBLNQwen2_5_VLForConditionalGeneration
        from optimum.rbln import RBLNQwen2VLForConditionalGeneration
        return RBLNQwen2VLForConditionalGeneration

    def _load_rbln_model_and_processor(self):
        from transformers import AutoProcessor

        rbln_cls = self._resolve_rbln_class()
        if self.verbose:
            self._debug_log(f'dispatching to {rbln_cls.__name__}', color='yellow')
        model = self._from_pretrained(rbln_cls)
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor

    def _prepare_content(self, inputs, dataset: str | None = None):
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': self._ensure_media_url(s['value'], 'image')}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    self._warn_once(
                        'ocrbench_min_pixels',
                        f"OCRBench dataset uses custom min_pixels={item['min_pixels']}",
                    )
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': self._ensure_media_url(s['value'], 'video')}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    @staticmethod
    def _process_vision_info_compat(messages):
        """Call ``qwen_vl_utils.process_vision_info`` with or without
        ``return_video_kwargs`` depending on the installed version.

        Returns ``(images, videos, video_kwargs)``. ``video_kwargs`` is
        empty when the installed util predates Qwen2.5-VL.
        """
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as err:
            raise ImportError(
                'qwen_vl_utils is required for RBLNQwen2VL. '
                'Install with: pip install qwen-vl-utils'
            ) from err

        sig = inspect.signature(process_vision_info)
        if 'return_video_kwargs' in sig.parameters:
            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True,
            )
            return images, videos, (video_kwargs or {})
        images, videos = process_vision_info(messages)
        return images, videos, {}

    def generate_inner(self, message, dataset=None):
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append(
            {'role': 'user', 'content': self._prepare_content(message, dataset=dataset)}
        )
        self._debug_log(messages, color='red')

        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True,
        )
        images, videos, video_kwargs = self._process_vision_info_compat([messages])
        inputs = self.processor(
            text=text, images=images, videos=videos,
            padding=True, return_tensors='pt',
            **video_kwargs,
        )

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        response = self._decode_generated(inputs, generated_ids)
        self._debug_log(response, color='green')
        return response
