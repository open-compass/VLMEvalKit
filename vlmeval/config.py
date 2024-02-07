from .vlm import *
from .api import GPT4V, GeminiProVision, GPT4V_Internal, QwenVLAPI
from functools import partial

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '

models = {
    'qwen_base': partial(QwenVL, model_path='Qwen/Qwen-VL'),
    'TransCore_M': partial(TransCoreM, root=TransCore_ROOT),
    'qwen_chat': partial(QwenVLChat, model_path='Qwen/Qwen-VL-Chat'),
    'PandaGPT_13B': partial(PandaGPT, name='PandaGPT_13B', root=PandaGPT_ROOT),
    'flamingov2': partial(OpenFlamingo, name='v2', mpt_pth='anas-awadalla/mpt-7b', ckpt_pth='openflamingo/OpenFlamingo-9B-vitl-mpt7b'),
    'flamingov2_fs': partial(OpenFlamingo, name='v2', with_context=True, mpt_pth='anas-awadalla/mpt-7b', ckpt_pth='openflamingo/OpenFlamingo-9B-vitl-mpt7b'),
    'idefics_9b_instruct': partial(IDEFICS, model_pth="HuggingFaceM4/idefics-9b-instruct"),
    'idefics_80b_instruct': partial(IDEFICS, model_pth="HuggingFaceM4/idefics-80b-instruct"),
    'idefics_9b_instruct_fs': partial(IDEFICS, model_pth="HuggingFaceM4/idefics-9b-instruct", with_context=True),
    'idefics_80b_instruct_fs': partial(IDEFICS, model_pth="HuggingFaceM4/idefics-80b-instruct", with_context=True),
    'llava_v1.5_7b': partial(LLaVA, model_pth='liuhaotian/llava-v1.5-7b'),
    'llava_v1.5_13b': partial(LLaVA, model_pth='liuhaotian/llava-v1.5-13b'),
    'llava_v1_7b': partial(LLaVA, model_pth=LLAVA_V1_7B_MODEL_PTH),
    'sharegpt4v_7b': partial(LLaVA, model_pth='Lin-Chen/ShareGPT4V-7B'),
    'sharegpt4v_13b': partial(LLaVA, model_pth='Lin-Chen/ShareGPT4V-13B'),
    'instructblip_7b': partial(InstructBLIP, name='instructblip_7b'),
    'instructblip_13b': partial(InstructBLIP, name='instructblip_13b'),
    'VisualGLM_6b': partial(VisualGLM, model_path="THUDM/visualglm-6b"),
    'MiniGPT-4-v2': partial(MiniGPT4, mode='v2', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-7B': partial(MiniGPT4, mode='v1_7b', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-13B': partial(MiniGPT4, mode='v1_13b', root=MiniGPT4_ROOT),
    "XComposer": partial(XComposer, model_path='internlm/internlm-xcomposer-vl-7b'),
    "XComposer2": partial(XComposer2, model_path='internlm/internlm-xcomposer2-vl-7b'),
    "mPLUG-Owl2": partial(mPLUG_Owl2, model_path='MAGAer13/mplug-owl2-llama2-7b'),
    'cogvlm-grounding-generalist':partial(CogVlm, name='cogvlm-grounding-generalist',tokenizer_name ='lmsys/vicuna-7b-v1.5'),
    'cogvlm-chat':partial(CogVlm, name='cogvlm-chat',tokenizer_name ='lmsys/vicuna-7b-v1.5'),
    'sharedcaptioner':partial(SharedCaptioner, model_path='Lin-Chen/ShareCaptioner'),
    'emu2':partial(Emu, name='emu2'),
    'emu2_chat':partial(Emu, name='emu2_chat'),
    'monkey':partial(Monkey, model_path='echo840/Monkey'),
    'monkey-chat':partial(MonkeyChat, model_path='echo840/Monkey-Chat'),
    'Yi_VL_6B':partial(Yi_VL, model_path='01-ai/Yi-VL-6B', root=Yi_ROOT),
    'Yi_VL_34B':partial(Yi_VL, model_path='01-ai/Yi-VL-34B', root=Yi_ROOT),
    'MMAlaya':partial(MMAlaya, model_path='DataCanvas/MMAlaya'),
    'MiniCPM-V':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'OmniLMM_12B':partial(OmniLMM12B, model_path='openbmb/OmniLMM-12B', root=OmniLMM_ROOT),
}

api_models = {
    'GPT4V': partial(GPT4V, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4V_INT': partial(GPT4V_Internal, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4V_SHORT': partial(
        GPT4V, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10, 
        system_prompt="Please responde to the following question / request in a short reply. "),
    'GPT4V_SHORT_INT': partial(
        GPT4V_Internal, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10,
        system_prompt="Please responde to the following question / request in a short reply. "),
    'GeminiProVision': partial(GeminiProVision, temperature=0, retry=10),
    'QwenVLPlus': partial(QwenVLAPI, model='qwen-vl-plus', temperature=0, retry=10),
    'QwenVLMax': partial(QwenVLAPI, model='qwen-vl-max', temperature=0, retry=10),
}

xtuner_models = {
    'llava-internlm2-7b': partial(LLaVA_XTuner, llm_path='internlm/internlm2-chat-7b', llava_path='xtuner/llava-internlm2-7b', visual_select_layer=-2, prompt_template='internlm2_chat'),
    'llava-internlm2-20b': partial(LLaVA_XTuner, llm_path='internlm/internlm2-chat-20b', llava_path='xtuner/llava-internlm2-20b', visual_select_layer=-2, prompt_template='internlm2_chat'),
    'llava-internlm-7b': partial(LLaVA_XTuner, llm_path='internlm/internlm-chat-7b', llava_path='xtuner/llava-internlm-7b', visual_select_layer=-2, prompt_template='internlm_chat'),
    'llava-v1.5-7b-xtuner': partial(LLaVA_XTuner, llm_path='lmsys/vicuna-7b-v1.5', llava_path='xtuner/llava-v1.5-7b-xtuner', visual_select_layer=-2, prompt_template='vicuna'),
    'llava-v1.5-13b-xtuner': partial(LLaVA_XTuner, llm_path='lmsys/vicuna-13b-v1.5', llava_path='xtuner/llava-v1.5-13b-xtuner', visual_select_layer=-2, prompt_template='vicuna'),
}

supported_VLM = {}
for model_set in [models, api_models, xtuner_models]:
    supported_VLM.update(model_set)
