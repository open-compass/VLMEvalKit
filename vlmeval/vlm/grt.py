"""Optional bridge to the frozen DIVE-Bench GRT implementation.

The runtime is imported only when a GRT model is instantiated. Stock VLMEvalKit
model classes and their generation paths are not modified.
"""

import importlib
import importlib.metadata
import importlib.resources
import json
import os
import random

from .base import BaseModel


TASKS = {
    'dive_bench_educational_high_fps', 'dive_bench_high_motion_high_fps',
    'dive_bench_high_motion_high_fps_preview1000', 'densevideo', 'densevideo_highmotion',
}
MODEL_CLASSES = {
    'llava_hf': ('lmms_eval.models.llava_hf', 'LlavaHf'),
    'qwen2_5_vl': ('lmms_eval.models.qwen2_5_vl', 'Qwen2_5_VL'),
    'qwen2_5_vl_dual_route_floor': (
        'densevideo_qwen_dual_plugin.models.qwen2_5_vl_dual_route_floor', 'Qwen2_5_VL_DualRouteFloor',
    ),
}


def _load_runtime(profile, role):
    try:
        importlib.metadata.version('dive-bench')
        resource = importlib.resources.files('tools.densevideo').joinpath('profiles.json')
        profiles = json.loads(resource.read_text())
    except (ImportError, FileNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
        raise ImportError(
            'GRT requires the optional pinned DIVE-Bench runtime. See docs/en/DIVE-Bench.md. '
            'Install it in a dedicated environment, not over another lmms-eval installation.'
        ) from exc
    if profiles.get('schema') != 1 or profiles.get('source_commit') != (
        '78284318c9c8664df5fc6785410a0a9c4e494436'
    ):
        raise ValueError('GRT runtime profile provenance differs from the audited release')
    selected = profiles['profiles'][profile]
    method = selected['roles'][role]
    module_name, class_name = MODEL_CLASSES[method['model']]
    model_class = getattr(importlib.import_module(module_name), class_name)
    from lmms_eval.api.instance import Instance
    from lmms_eval.utils import simple_parse_args_string
    return selected, method, model_class, Instance, simple_parse_args_string


def _configure_reproduction():
    import numpy as np
    import torch
    if os.environ.get('PYTHONHASHSEED') != '0' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Start Python with PYTHONHASHSEED=0 and CUBLAS_WORKSPACE_CONFIG=:4096:8')
    distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1
    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1 or distributed:
        raise ValueError('Released GRT profiles require one visible CUDA GPU and one process')
    random.seed(0)
    # Preserve the effective historical evaluator seed tuple: 0,1234,1234,1234.
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class GRT(BaseModel):
    VIDEO_LLM = True
    INTERLEAVE = False

    def __init__(self, profile='route31', role='candidate', use_vllm=False):
        super().__init__()
        if profile not in {'route31', 'qwen3', 'qwen7'} or role not in {'base', 'all', 'candidate'}:
            raise ValueError('Unknown GRT profile/role; see docs/en/DIVE-Bench.md')
        if use_vllm:
            raise ValueError('GRT implements vision-token reuse in the Transformers runtime, not vLLM')
        selected, method, model_class, instance_class, parse_args = _load_runtime(profile, role)
        _configure_reproduction()
        self.profile = profile
        self.role = role
        self.method = method['method']
        self.nframe = 8
        self.fps = -1
        self.max_new_tokens = selected['max_new_tokens']
        self._instance_class = instance_class
        self.backend = model_class(**parse_args(method['model_args']), batch_size=1)
        self._request_id = 0

    def use_custom_prompt(self, dataset):
        return dataset in TASKS

    def build_prompt(self, line, dataset, video_llm=True):
        # The released runtime samples the same endpoint-inclusive frames as the target.
        return dataset.build_grt_prompt(line)

    def generate_inner(self, message, dataset=None):
        if dataset not in TASKS:
            raise ValueError('The released GRT profiles are scoped to DIVE-Bench tasks')
        if self.nframe != 8:
            raise ValueError('GRT frame budget was overridden; released profiles require eight frames')
        videos = [item['value'] for item in message if item['type'] == 'video']
        texts = [item['value'] for item in message if item['type'] == 'text']
        if len(videos) != 1 or not texts or any(item['type'] not in {'video', 'text'} for item in message):
            raise ValueError('GRT expects exactly one video and text, without image/audio messages')
        context = '\n'.join(texts)
        # No answer, label, or annotation is passed to the inference model.
        doc = {'video_path': videos[0], 'question': context}
        task, split, doc_id = 'vlmeval_dive_bench', 'test', self._request_id
        self._request_id += 1
        generation = {
            'until': ['ASSISTANT:'], 'image_aspect_ratio': 'original',
            'max_new_tokens': self.max_new_tokens, 'temperature': 0,
            'top_p': 1.0, 'num_beams': 1, 'do_sample': False,
        }
        request = self._instance_class(
            request_type='generate_until',
            arguments=(context, generation, lambda row: [row['video_path']], doc_id, task, split),
            idx=0, metadata={'task': task, 'doc_id': doc_id, 'repeats': 1},
        )
        previous = self.backend.task_dict
        self.backend.task_dict = {task: {split: {doc_id: doc}}}
        try:
            output = self.backend.generate_until([request])
        finally:
            self.backend.task_dict = previous
        if len(output) != 1 or not isinstance(output[0], str) or not output[0].strip():
            raise RuntimeError('GRT runtime returned an invalid or empty generation')
        return output[0]
