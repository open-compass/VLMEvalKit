from .vlm import *
from functools import partial

PandaGPT_ROOT = None
MPT_7B_PTH = None
Flamingov2_CKPT_PTH = None
MiniGPT4_ROOT = None
TransCore_ROOT = None

idefics_model_path_map = {
    'idefics_9b_instruct': "HuggingFaceM4/idefics-9b-instruct",
    'idefics_80b_instruct': "HuggingFaceM4/idefics-80b-instruct"
}

llava_model_path_map = {
    'llava_v1.5_7b': 'liuhaotian/llava-v1.5-7b',
    'llava_v1.5_13b': 'liuhaotian/llava-v1.5-13b',
    'llava_v1_7b': 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '
}

supported_VLM = {
    'qwen_base': partial(QwenVL, model_path='Qwen/Qwen-VL'),
    'TransCore_M': partial(TransCoreM, root=TransCore_ROOT),
    'qwen_chat': partial(QwenVLChat, model_path='Qwen/Qwen-VL-Chat'),
    'PandaGPT_13B': partial(PandaGPT, name='PandaGPT_13B', root=PandaGPT_ROOT),
    'flamingov2': partial(OpenFlamingo, name='v2', mpt_pth=MPT_7B_PTH, ckpt_pth=Flamingov2_CKPT_PTH),
    'flamingov2_fs': partial(OpenFlamingo, name='v2', with_context=True, mpt_pth=MPT_7B_PTH, ckpt_pth=Flamingov2_CKPT_PTH),
    'idefics_9b_instruct': partial(IDEFICS, name='idefics_9b_instruct', model_path_map=idefics_model_path_map), 
    'idefics_80b_instruct': partial(IDEFICS, name='idefics_80b_instruct', model_path_map=idefics_model_path_map),
    'idefics_9b_instruct_fs': partial(IDEFICS, name='idefics_9b_instruct', model_path_map=idefics_model_path_map, with_context=True), 
    'idefics_80b_instruct_fs': partial(IDEFICS, name='idefics_80b_instruct', model_path_map=idefics_model_path_map, with_context=True),
    'llava_v1.5_7b': partial(LLaVA, name='llava_v1.5_7b', model_path_map=llava_model_path_map),
    'sharegpt4v_7b': partial(LLaVA, name='Lin-Chen/ShareGPT4V-7B', model_path_map={}),
    'llava_v1.5_13b': partial(LLaVA, name='llava_v1.5_13b', model_path_map=llava_model_path_map),
    'llava_v1_7b': partial(LLaVA, name='llava_v1_7b', model_path_map=llava_model_path_map),
    'instructblip_7b': partial(InstructBLIP, name='instructblip_7b'),
    'instructblip_13b': partial(InstructBLIP, name='instructblip_13b'),
    'VisualGLM_6b': partial(VisualGLM, model_path="THUDM/visualglm-6b"),
    'MiniGPT-4-v2': partial(MiniGPT4, mode='v2', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-7B': partial(MiniGPT4, mode='v1_7b', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-13B': partial(MiniGPT4, mode='v1_13b', root=MiniGPT4_ROOT),
    "XComposer": partial(XComposer, model_path='internlm/internlm-xcomposer-vl-7b'), 
    "mPLUG-Owl2": partial(mPLUG_Owl2, model_path='MAGAer13/mplug-owl2-llama2-7b')
}
