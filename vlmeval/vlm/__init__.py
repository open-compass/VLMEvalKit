import torch
torch.set_grad_enabled(False)
torch.manual_seed(1234)
from functools import partial
from .qwen_vl import QwenVL
from .pandagpt import PandaGPT
from .open_flamingo import OpenFlamingo
from .idefics import IDEFICS
from .llava import LLaVA
from .instructblip import InstructBLIP
from .visualglm import VisualGLM
from .otteri import OtterI
from .viscpm_api import VisCPM_API
from .minigpt4 import MiniGPT4
from .minigpt4_mllm import MiniGPT4_mllm
from .sphinx import Sphinx
from .xcomposer import XComposer
from .mplug_owl2 import mPLUG_Owl2


model_cls_map = {
    'qwen_base': partial(QwenVL, name='qwen_base'), 
    'qwen_chat': partial(QwenVL, name='qwen_chat'),
    'PandaGPT_13B': partial(PandaGPT, name='PandaGPT_13B'),
    'flamingov2': partial(OpenFlamingo, name='v2'),
    'flamingov2_fs': partial(OpenFlamingo, name='v2', with_context=True),
    'idefics_9b_instruct': partial(IDEFICS, name='idefics_9b_instruct'), 
    'idefics_80b_instruct': partial(IDEFICS, name='idefics_80b_instruct'),
    'idefics_9b_instruct_fs': partial(IDEFICS, name='idefics_9b_instruct', with_context=True), 
    'idefics_80b_instruct_fs': partial(IDEFICS, name='idefics_80b_instruct', with_context=True),
    'llava_v1.5_7b': partial(LLaVA, name='llava_v1.5_7b'),
    'llava_v1.5_13b': partial(LLaVA, name='llava_v1.5_13b'),
    'llava_v1_7b': partial(LLaVA, name='llava_v1_7b'),
    'instructblip_7b': partial(InstructBLIP, name='instructblip_7b'),
    'instructblip_13b': partial(InstructBLIP, name='instructblip_13b'),
    'VisualGLM_6b': VisualGLM,
    'MiniGPT-4-v2': partial(MiniGPT4, mode='v2'),
    'MiniGPT-4-v1-7B': partial(MiniGPT4, mode='v1_7b'),
    'MiniGPT-4-v1-13B': partial(MiniGPT4, mode='v1_13b'),
    'MiniGPT4_mmbench_v2': partial(MiniGPT4_mllm, mode = 'v2'),
    'MiniGPT4_mmbench_v1_7B': partial(MiniGPT4_mllm, mode = 'v1_7b'),
    'MiniGPT4_mmbench_v1_13B': partial(MiniGPT4_mllm, mode = 'v1_13b'),
    "XComposer": XComposer, 
    "mPLUG-Owl2": mPLUG_Owl2
}