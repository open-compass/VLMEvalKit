from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '

ungrouped = {
    'TransCore_M': partial(TransCoreM, root=TransCore_ROOT),
    'PandaGPT_13B': partial(PandaGPT, name='PandaGPT_13B', root=PandaGPT_ROOT),
    'flamingov2': partial(OpenFlamingo, name='v2', mpt_pth='anas-awadalla/mpt-7b', ckpt_pth='openflamingo/OpenFlamingo-9B-vitl-mpt7b'),
    'VisualGLM_6b': partial(VisualGLM, model_path='THUDM/visualglm-6b'),
    'mPLUG-Owl2': partial(mPLUG_Owl2, model_path='MAGAer13/mplug-owl2-llama2-7b'),
    'emu2_chat': partial(Emu, model_path='BAAI/Emu2-Chat',),
    'MMAlaya': partial(MMAlaya, model_path='DataCanvas/MMAlaya'),
    'OmniLMM_12B': partial(OmniLMM12B, model_path='openbmb/OmniLMM-12B', root=OmniLMM_ROOT),
    'MGM_7B': partial(Mini_Gemini, model_path='YanweiLi/MGM-7B-HD', root=Mini_Gemini_ROOT),
    'Bunny-llama3-8B': partial(BunnyLLama3, model_path='BAAI/Bunny-Llama-3-8B-V'),
    'VXVERSE': partial(VXVERSE, model_name='XVERSE-V-13B', root=VXVERSE_ROOT),
    'paligemma-3b-mix-448': partial(PaliGemma, model_path='google/paligemma-3b-mix-448'),
    '360VL-70B': partial(QH_360VL, model_path='qihoo360/360VL-70B'),
    'Phi-3-Vision': partial(Phi3Vision, model_path='microsoft/Phi-3-vision-128k-instruct'),
}

api_models = {
    # GPT-4V series
    'GPT4V': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4V_HIGH': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=-1, img_detail='high', retry=10),
    'GPT4V_20240409': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4V_20240409_HIGH': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=-1, img_detail='high', retry=10),
    'GPT4o': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4o_HIGH': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=-1, img_detail='high', retry=10),
    # Gemini-V
    'GeminiProVision': partial(GeminiProVision, model='gemini-1.0-pro', temperature=0, retry=10),
    'GeminiPro1-5': partial(GeminiProVision, model='gemini-1.5-pro', temperature=0, retry=10),
    # Qwen-VL Series
    'QwenVLPlus': partial(QwenVLAPI, model='qwen-vl-plus', temperature=0, retry=10),
    'QwenVLMax': partial(QwenVLAPI, model='qwen-vl-max', temperature=0, retry=10),
    # Reka Series
    'RekaEdge': partial(Reka, model='reka-edge-20240208'), 
    'RekaFlash': partial(Reka, model='reka-flash-20240226'), 
    'RekaCore': partial(Reka, model='reka-core-20240415'), 
    # Step1V Series
    'Step1V': partial(GPT4V, model='step-1v-8k', api_base="https://api.stepfun.com/v1/chat/completions", temperature=0, retry=10),
    # Internal Only
    # 'GPT4V_INT': partial(GPT4V_Internal, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    # 'Step1V_INT': partial(Step1V_INT, temperature=0, retry=10),
    'Claude3V_Opus': partial(Claude3V, model='claude-3-opus-20240229', temperature=0, retry=10),
    'Claude3V_Sonnet': partial(Claude3V, model='claude-3-sonnet-20240229', temperature=0, retry=10),
    'Claude3V_Haiku': partial(Claude3V, model='claude-3-haiku-20240307', temperature=0, retry=10),
    # GLM4V
    'GLM4V': partial(GLMVisionAPI, model='glm4v-biz-eval', temperature=0, retry=10),
}

minicpm_series = {
    'MiniCPM-V': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5': partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
}

xtuner_series = {
    'llava-internlm2-7b': partial(LLaVA_XTuner, llm_path='internlm/internlm2-chat-7b', llava_path='xtuner/llava-internlm2-7b', visual_select_layer=-2, prompt_template='internlm2_chat'),
    'llava-internlm2-20b': partial(LLaVA_XTuner, llm_path='internlm/internlm2-chat-20b', llava_path='xtuner/llava-internlm2-20b', visual_select_layer=-2, prompt_template='internlm2_chat'),
    'llava-internlm-7b': partial(LLaVA_XTuner, llm_path='internlm/internlm-chat-7b', llava_path='xtuner/llava-internlm-7b', visual_select_layer=-2, prompt_template='internlm_chat'),
    'llava-v1.5-7b-xtuner': partial(LLaVA_XTuner, llm_path='lmsys/vicuna-7b-v1.5', llava_path='xtuner/llava-v1.5-7b-xtuner', visual_select_layer=-2, prompt_template='vicuna'),
    'llava-v1.5-13b-xtuner': partial(LLaVA_XTuner, llm_path='lmsys/vicuna-13b-v1.5', llava_path='xtuner/llava-v1.5-13b-xtuner', visual_select_layer=-2, prompt_template='vicuna'),
    'llava-llama-3-8b': partial(LLaVA_XTuner, llm_path='xtuner/llava-llama-3-8b-v1_1', llava_path='xtuner/llava-llama-3-8b-v1_1', visual_select_layer=-2, prompt_template='llama3_chat'),
}

qwen_series = {
    'qwen_base': partial(QwenVL, model_path='Qwen/Qwen-VL'),
    'qwen_chat': partial(QwenVL, model_path='Qwen/Qwen-VL-Chat'),
    'monkey': partial(Monkey, model_path='echo840/Monkey'),
    'monkey-chat': partial(MonkeyChat, model_path='echo840/Monkey-Chat')
}

llava_series = {
    'llava_v1.5_7b': partial(LLaVA, model_pth='liuhaotian/llava-v1.5-7b'),
    'llava_v1.5_13b': partial(LLaVA, model_pth='liuhaotian/llava-v1.5-13b'),
    'llava_v1_7b': partial(LLaVA, model_pth=LLAVA_V1_7B_MODEL_PTH),
    'sharegpt4v_7b': partial(LLaVA, model_pth='Lin-Chen/ShareGPT4V-7B'),
    'sharegpt4v_13b': partial(LLaVA, model_pth='Lin-Chen/ShareGPT4V-13B'),
    'llava_next_vicuna_7b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-vicuna-7b-hf'),
    'llava_next_vicuna_13b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-vicuna-13b-hf'),
    'llava_next_mistral_7b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-mistral-7b-hf'),
    'llava_next_yi_34b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-34b-hf'),
}

internvl_series = {
    'InternVL-Chat-V1-1': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-1'),
    'InternVL-Chat-V1-2': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-2'),
    'InternVL-Chat-V1-2-Plus': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-2-Plus'),
    'InternVL-Chat-V1-5': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-5'),
    'Mini-InternVL-Chat-2B-V1-5': partial(InternVLChat, model_path='OpenGVLab/Mini-InternVL-Chat-2B-V1-5'),
    'Mini-InternVL-Chat-4B-V1-5': partial(InternVLChat, model_path='OpenGVLab/Mini-InternVL-Chat-4B-V1-5'),
}

yivl_series = {
    'Yi_VL_6B': partial(Yi_VL, model_path='01-ai/Yi-VL-6B', root=Yi_ROOT),
    'Yi_VL_34B': partial(Yi_VL, model_path='01-ai/Yi-VL-34B', root=Yi_ROOT),
}

xcomposer_series = {
    'XComposer': partial(XComposer, model_path='internlm/internlm-xcomposer-vl-7b'),
    'sharecaptioner': partial(ShareCaptioner, model_path='Lin-Chen/ShareCaptioner'),
    'XComposer2': partial(XComposer2, model_path='internlm/internlm-xcomposer2-vl-7b'),
    'XComposer2_1.8b': partial(XComposer2, model_path='internlm/internlm-xcomposer2-vl-1_8b'),
    'XComposer2_4KHD': partial(XComposer2_4KHD, model_path='internlm/internlm-xcomposer2-4khd-7b'),
}

minigpt4_series = {
    'MiniGPT-4-v2': partial(MiniGPT4, mode='v2', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-7B': partial(MiniGPT4, mode='v1_7b', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-13B': partial(MiniGPT4, mode='v1_13b', root=MiniGPT4_ROOT),
}

idefics_series = {
    'idefics_9b_instruct': partial(IDEFICS, model_pth='HuggingFaceM4/idefics-9b-instruct'),
    'idefics_80b_instruct': partial(IDEFICS, model_pth='HuggingFaceM4/idefics-80b-instruct'),
    'idefics2_8b': partial(IDEFICS2, model_path='HuggingFaceM4/idefics2-8b'),
}

instructblip_series = {
    'instructblip_7b': partial(InstructBLIP, name='instructblip_7b'),
    'instructblip_13b': partial(InstructBLIP, name='instructblip_13b'),
}

deepseekvl_series = {
    'deepseek_vl_7b': partial(DeepSeekVL, model_path='deepseek-ai/deepseek-vl-7b-chat'),
    'deepseek_vl_1.3b': partial(DeepSeekVL, model_path='deepseek-ai/deepseek-vl-1.3b-chat'),
}

cogvlm_series = {
    'cogvlm-grounding-generalist': partial(CogVlm, model_path='THUDM/cogvlm-grounding-generalist-hf', tokenizer_name='lmsys/vicuna-7b-v1.5'),
    'cogvlm-chat': partial(CogVlm, model_path='THUDM/cogvlm-chat-hf', tokenizer_name='lmsys/vicuna-7b-v1.5'),
    'cogvlm2-llama3-chat-19B': partial(CogVlm, model_path='THUDM/cogvlm2-llama3-chat-19B'),
    'glm-4v-9b': partial(GLM4v, model_path='THUDM/glm-4v-9b')
}

wemm_series = {
    'WeMM': partial(WeMM, model_path='feipengma/WeMM'),
}

supported_VLM = {}

model_groups = [
    ungrouped, api_models, 
    xtuner_series, qwen_series, llava_series, internvl_series, yivl_series,
    xcomposer_series, minigpt4_series, idefics_series, instructblip_series,
    deepseekvl_series, minicpm_series, cogvlm_series, wemm_series
]

for grp in model_groups:
    supported_VLM.update(grp)
