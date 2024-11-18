from vlmeval.api.bailingmm import bailingMMAPI
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
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '

video_models = {
    'Video-LLaVA-7B':partial(VideoLLaVA, model_path='LanguageBind/Video-LLaVA-7B'),
    'Video-LLaVA-7B-HF':partial(VideoLLaVA_HF, model_path='LanguageBind/Video-LLaVA-7B-hf'),
    'VideoChat2-HD':partial(VideoChat2_HD, model_path='OpenGVLab/VideoChat2_HD_stage4_Mistral_7B', root=VideoChat2_ROOT, config_file='./vlmeval/vlm/video_llm/configs/videochat2_hd.json'),
    'Chat-UniVi-7B': partial(Chatunivi, model_path="Chat-UniVi/Chat-UniVi"),
    'Chat-UniVi-7B-v1.5': partial(Chatunivi, model_path="Chat-UniVi/Chat-UniVi-7B-v1.5"),
    'LLaMA-VID-7B': partial(LLaMAVID, model_path='YanweiLi/llama-vid-7b-full-224-video-fps-1'),
    'Video-ChatGPT': partial(VideoChatGPT, model_path='MBZUAI/Video-ChatGPT-7B', dir_root=VideoChatGPT_ROOT),
    'PLLaVA-7B': partial(PLLaVA, model_path='ermu2001/pllava-7b', dir_root=PLLaVA_ROOT),
    'PLLaVA-13B': partial(PLLaVA, model_path='ermu2001/pllava-13b', dir_root=PLLaVA_ROOT),
    'PLLaVA-34B': partial(PLLaVA, model_path='ermu2001/pllava-34b', dir_root=PLLaVA_ROOT),
}

ungrouped = {
    'TransCore_M': partial(TransCoreM, root=TransCore_ROOT),
    'PandaGPT_13B': partial(PandaGPT, name='PandaGPT_13B', root=PandaGPT_ROOT),
    'flamingov2': partial(OpenFlamingo, name='v2', mpt_pth='anas-awadalla/mpt-7b', ckpt_pth='openflamingo/OpenFlamingo-9B-vitl-mpt7b'),
    'VisualGLM_6b': partial(VisualGLM, model_path='THUDM/visualglm-6b'),
    'mPLUG-Owl2': partial(mPLUG_Owl2, model_path='MAGAer13/mplug-owl2-llama2-7b'),
    'mPLUG-Owl3': partial(mPLUG_Owl3, model_path='mPLUG/mPLUG-Owl3-7B-240728'),
    'emu2_chat': partial(Emu, model_path='BAAI/Emu2-Chat'),
    'OmniLMM_12B': partial(OmniLMM12B, model_path='openbmb/OmniLMM-12B', root=OmniLMM_ROOT),
    'MGM_7B': partial(Mini_Gemini, model_path='YanweiLi/MGM-7B-HD', root=Mini_Gemini_ROOT),
    'Bunny-llama3-8B': partial(BunnyLLama3, model_path='BAAI/Bunny-v1_1-Llama-3-8B-V'),
    'VXVERSE': partial(VXVERSE, model_name='XVERSE-V-13B', root=VXVERSE_ROOT),
    'paligemma-3b-mix-448': partial(PaliGemma, model_path='google/paligemma-3b-mix-448'),
    '360VL-70B': partial(QH_360VL, model_path='qihoo360/360VL-70B'),
    'Llama-3-MixSenseV1_1': partial(LLama3Mixsense, model_path='Zero-Vision/Llama-3-MixSenseV1_1'),
    'Parrot': partial(Parrot, model_path='AIDC-AI/Parrot-7B'),
    'OmChat': partial(OmChat, model_path='omlab/omchat-v2.0-13B-single-beta_hf'),
    'RBDash_72b': partial(RBDash, model_path='RBDash-Team/RBDash-v1.5', root=RBDash_ROOT),
    'Pixtral-12B': partial(Pixtral, model_path='mistralai/Pixtral-12B-2409'),
    'Falcon2-VLM-11B': partial(Falcon2VLM, model_path='tiiuae/falcon-11B-vlm')
}

api_models = {
    # GPT
    'GPT4V': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10, verbose=False),
    'GPT4V_HIGH': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    'GPT4V_20240409': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=512, img_detail='low', retry=10, verbose=False),
    'GPT4V_20240409_HIGH': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    'GPT4o': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=512, img_detail='low', retry=10, verbose=False),
    'GPT4o_HIGH': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    'GPT4o_20240806': partial(GPT4V, model='gpt-4o-2024-08-06', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    'GPT4o_MINI': partial(GPT4V, model='gpt-4o-mini-2024-07-18', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    # Gemini
    'GeminiPro1-0': partial(GeminiProVision, model='gemini-1.0-pro', temperature=0, retry=10),  # now GeminiPro1-0 is only supported by vertex backend
    'GeminiPro1-5': partial(GeminiProVision, model='gemini-1.5-pro', temperature=0, retry=10),
    'GeminiFlash1-5': partial(GeminiProVision, model='gemini-1.5-flash', temperature=0, retry=10),
    # Qwen-VL
    'QwenVLPlus': partial(QwenVLAPI, model='qwen-vl-plus', temperature=0, retry=10),
    'QwenVLMax': partial(QwenVLAPI, model='qwen-vl-max', temperature=0, retry=10),
    # Reka
    'RekaEdge': partial(Reka, model='reka-edge-20240208'),
    'RekaFlash': partial(Reka, model='reka-flash-20240226'),
    'RekaCore': partial(Reka, model='reka-core-20240415'),
    # Step1V
    'Step1V': partial(GPT4V, model='step-1v-32k', api_base="https://api.stepfun.com/v1/chat/completions", temperature=0, retry=10, img_size=-1, img_detail='high'),
    # Yi-Vision
    'Yi-Vision': partial(GPT4V, model='yi-vision', api_base="https://api.lingyiwanwu.com/v1/chat/completions", temperature=0, retry=10),
    # Claude
    'Claude3V_Opus': partial(Claude3V, model='claude-3-opus-20240229', temperature=0, retry=10, verbose=False),
    'Claude3V_Sonnet': partial(Claude3V, model='claude-3-sonnet-20240229', temperature=0, retry=10, verbose=False),
    'Claude3V_Haiku': partial(Claude3V, model='claude-3-haiku-20240307', temperature=0, retry=10, verbose=False),
    'Claude3-5V_Sonnet': partial(Claude3V, model='claude-3-5-sonnet-20240620', temperature=0, retry=10, verbose=False),
    'Claude3-5V_Sonnet_20241022': partial(Claude3V, model='claude-3-5-sonnet-20241022', temperature=0, retry=10, verbose=False),
    # GLM4V
    'GLM4V': partial(GLMVisionAPI, model='glm4v-biz-eval', temperature=0, retry=10),
    # CongRong
    'CloudWalk': partial(CWWrapper, model='cw-congrong-v1.5', temperature=0, retry=10),
    # SenseChat-V
    'SenseChat-5-Vision': partial(SenseChatVisionAPI, model='SenseChat-5-Vision', temperature=0, retry=10),
    'HunYuan-Vision': partial(HunyuanVision, model='hunyuan-vision', temperature=0, retry=10),
    'bailingMM': partial(bailingMMAPI, model='bailingMM-mini', temperature=0, retry=10),
    # BlueLM-V
    "BlueLM_V": partial(BlueLM_V_API, model='BlueLM-VL-v3.0', temperature=0, retry=10),
    # JiuTian-VL
    "JTVL": partial(JTVLChatAPI, model='jt-vl-chat', temperature=0, retry=10),
    "Taiyi": partial(TaiyiAPI, model='taiyi', temperature=0, retry=10),
}

mmalaya_series = {
    'MMAlaya': partial(MMAlaya, model_path='DataCanvas/MMAlaya'),
    'MMAlaya2': partial(MMAlaya2, model_path='DataCanvas/MMAlaya2'),
}

minicpm_series = {
    'MiniCPM-V': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5': partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
    'MiniCPM-V-2_6': partial(MiniCPM_V_2_6, model_path='openbmb/MiniCPM-V-2_6'),
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
    'qwen_chat': partial(QwenVLChat, model_path='Qwen/Qwen-VL-Chat'),
    'monkey': partial(Monkey, model_path='echo840/Monkey'),
    'monkey-chat': partial(MonkeyChat, model_path='echo840/Monkey-Chat'),
    'minimonkey': partial(MiniMonkey, model_path='mx262/MiniMonkey')
}

llava_series = {
    'llava_v1.5_7b': partial(LLaVA, model_path='liuhaotian/llava-v1.5-7b'),
    'llava_v1.5_13b': partial(LLaVA, model_path='liuhaotian/llava-v1.5-13b'),
    'llava_v1_7b': partial(LLaVA, model_path=LLAVA_V1_7B_MODEL_PTH),
    'sharegpt4v_7b': partial(LLaVA, model_path='Lin-Chen/ShareGPT4V-7B'),
    'sharegpt4v_13b': partial(LLaVA, model_path='Lin-Chen/ShareGPT4V-13B'),
    'llava_next_vicuna_7b': partial(LLaVA_Next, model_path='llava-hf/llava-v1.6-vicuna-7b-hf'),
    'llava_next_vicuna_13b': partial(LLaVA_Next, model_path='llava-hf/llava-v1.6-vicuna-13b-hf'),
    'llava_next_mistral_7b': partial(LLaVA_Next, model_path='llava-hf/llava-v1.6-mistral-7b-hf'),
    'llava_next_yi_34b': partial(LLaVA_Next, model_path='llava-hf/llava-v1.6-34b-hf'),
    'llava_next_llama3': partial(LLaVA_Next, model_path='llava-hf/llama3-llava-next-8b-hf'),
    'llava_next_72b': partial(LLaVA_Next, model_path='llava-hf/llava-next-72b-hf'),
    'llava_next_110b': partial(LLaVA_Next, model_path='llava-hf/llava-next-110b-hf'),
    'llava_next_qwen_32b': partial(LLaVA_Next2, model_path='lmms-lab/llava-next-qwen-32b'),
    'llava_next_interleave_7b': partial(LLaVA_Next, model_path='llava-hf/llava-interleave-qwen-7b-hf'),
    'llava_next_interleave_7b_dpo': partial(LLaVA_Next, model_path='llava-hf/llava-interleave-qwen-7b-dpo-hf'),
    'llava_onevision_qwen2_0.5b_si': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-0.5b-si'),
    'llava_onevision_qwen2_7b_si': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-7b-si'),
    'llava_onevision_qwen2_72b_si': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-72b-si'),
    'llava_onevision_qwen2_0.5b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov'),
    'llava_onevision_qwen2_7b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-7b-ov'),
    'llava_onevision_qwen2_72b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-72b-ov-sft'),
    'Aquila-VL-2B': partial(LLaVA_OneVision, model_path='BAAI/Aquila-VL-2B-llava-qwen'),
    'llava_video_qwen2_7b':partial(LLaVA_OneVision, model_path='lmms-lab/LLaVA-Video-7B-Qwen2'),
    'llava_video_qwen2_72b':partial(LLaVA_OneVision, model_path='lmms-lab/LLaVA-Video-72B-Qwen2'),
}

internvl_series = {
    'InternVL-Chat-V1-1': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-1', version='V1.1'),
    'InternVL-Chat-V1-2': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-2', version='V1.2'),
    'InternVL-Chat-V1-2-Plus': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-2-Plus', version='V1.2'),
    'InternVL-Chat-V1-5': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-5', version='V1.5'),
    'Mini-InternVL-Chat-2B-V1-5': partial(InternVLChat, model_path='OpenGVLab/Mini-InternVL-Chat-2B-V1-5', version='V1.5'),
    'Mini-InternVL-Chat-4B-V1-5': partial(InternVLChat, model_path='OpenGVLab/Mini-InternVL-Chat-4B-V1-5', version='V1.5'),
    # InternVL2 series
    'InternVL2-1B': partial(InternVLChat, model_path='OpenGVLab/InternVL2-1B', version='V2.0'),
    'InternVL2-2B': partial(InternVLChat, model_path='OpenGVLab/InternVL2-2B', version='V2.0'),
    'InternVL2-4B': partial(InternVLChat, model_path='OpenGVLab/InternVL2-4B', version='V2.0'),
    'InternVL2-8B': partial(InternVLChat, model_path='OpenGVLab/InternVL2-8B', version='V2.0'),
    'InternVL2-8B-MPO': partial(InternVLChat, model_path='OpenGVLab/InternVL2-8B-MPO', version='V2.0', cot_prompt=True),
    'InternVL2-26B': partial(InternVLChat, model_path='OpenGVLab/InternVL2-26B', version='V2.0'),
    'InternVL2-40B': partial(InternVLChat, model_path='OpenGVLab/InternVL2-40B', version='V2.0', load_in_8bit=True),
    'InternVL2-76B': partial(InternVLChat, model_path='OpenGVLab/InternVL2-Llama3-76B', version='V2.0'),
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
    'XComposer2d5': partial(XComposer2d5, model_path='internlm/internlm-xcomposer2d5-7b'),
}

minigpt4_series = {
    'MiniGPT-4-v2': partial(MiniGPT4, mode='v2', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-7B': partial(MiniGPT4, mode='v1_7b', root=MiniGPT4_ROOT),
    'MiniGPT-4-v1-13B': partial(MiniGPT4, mode='v1_13b', root=MiniGPT4_ROOT),
}

idefics_series = {
    'idefics_9b_instruct': partial(IDEFICS, model_path='HuggingFaceM4/idefics-9b-instruct'),
    'idefics_80b_instruct': partial(IDEFICS, model_path='HuggingFaceM4/idefics-80b-instruct'),
    'idefics2_8b': partial(IDEFICS2, model_path='HuggingFaceM4/idefics2-8b'),

    # Idefics3 follows Idefics2 Pattern
    'Idefics3-8B-Llama3': partial(IDEFICS2, model_path='HuggingFaceM4/Idefics3-8B-Llama3'),

}

instructblip_series = {
    'instructblip_7b': partial(InstructBLIP, name='instructblip_7b'),
    'instructblip_13b': partial(InstructBLIP, name='instructblip_13b'),
}

deepseekvl_series = {
    'deepseek_vl_7b': partial(DeepSeekVL, model_path='deepseek-ai/deepseek-vl-7b-chat'),
    'deepseek_vl_1.3b': partial(DeepSeekVL, model_path='deepseek-ai/deepseek-vl-1.3b-chat'),
}


janus_series = {
    'Janus-1.3B': partial(Janus, model_path='deepseek-ai/Janus-1.3B')
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

cambrian_series = {
    'cambrian_8b': partial(Cambrian, model_path='nyu-visionx/cambrian-8b'),
    'cambrian_13b': partial(Cambrian, model_path='nyu-visionx/cambrian-13b'),
    'cambrian_34b': partial(Cambrian, model_path='nyu-visionx/cambrian-34b'),
}

chameleon_series = {
    'chameleon_7b': partial(Chameleon, model_path='facebook/chameleon-7b'),
    'chameleon_30b': partial(Chameleon, model_path='facebook/chameleon-30b'),
}

vila_series = {
    'VILA1.5-3b': partial(VILA, model_path='Efficient-Large-Model/VILA1.5-3b'),
    'Llama-3-VILA1.5-8b': partial(VILA, model_path='Efficient-Large-Model/Llama-3-VILA1.5-8b'),
    'VILA1.5-13b': partial(VILA, model_path='Efficient-Large-Model/VILA1.5-13b'),
    'VILA1.5-40b': partial(VILA, model_path='Efficient-Large-Model/VILA1.5-40b'),
}

ovis_series = {
    'Ovis1.5-Llama3-8B': partial(Ovis, model_path='AIDC-AI/Ovis1.5-Llama3-8B'),
    'Ovis1.5-Gemma2-9B': partial(Ovis, model_path='AIDC-AI/Ovis1.5-Gemma2-9B'),
    'Ovis1.6-Gemma2-9B': partial(Ovis1_6, model_path='AIDC-AI/Ovis1.6-Gemma2-9B'),
    'Ovis1.6-Llama3.2-3B': partial(Ovis1_6, model_path='AIDC-AI/Ovis1.6-Llama3.2-3B')
}

mantis_series = {
    'Mantis-8B-siglip-llama3': partial(Mantis, model_path='TIGER-Lab/Mantis-8B-siglip-llama3'),
    'Mantis-8B-clip-llama3': partial(Mantis, model_path='TIGER-Lab/Mantis-8B-clip-llama3'),
    'Mantis-8B-Idefics2': partial(Mantis, model_path='TIGER-Lab/Mantis-8B-Idefics2'),
    'Mantis-8B-Fuyu': partial(Mantis, model_path='TIGER-Lab/Mantis-8B-Fuyu')
}

phi3_series = {
    'Phi-3-Vision': partial(Phi3Vision, model_path='microsoft/Phi-3-vision-128k-instruct'),
    'Phi-3.5-Vision': partial(Phi3_5Vision, model_path='microsoft/Phi-3.5-vision-instruct')
}

xgen_mm_series = {
    'xgen-mm-phi3-interleave-r-v1.5': partial(XGenMM, model_path='Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5'),
    'xgen-mm-phi3-dpo-r-v1.5': partial(XGenMM, model_path='Salesforce/xgen-mm-phi3-mini-instruct-dpo-r-v1.5'),
}

qwen2vl_series = {
    'Qwen-VL-Max-0809': partial(Qwen2VLAPI, model='qwen-vl-max-0809', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen-VL-Plus-0809': partial(Qwen2VLAPI, model='qwen-vl-plus-0809', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-72B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-72B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-7B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-7B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-7B-Instruct-AWQ': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-7B-Instruct-AWQ', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-7B-Instruct-GPTQ-Int4': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-7B-Instruct-GPTQ-Int8': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-2B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-2B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-2B-Instruct-AWQ': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-2B-Instruct-AWQ', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-2B-Instruct-GPTQ-Int4': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-2B-Instruct-GPTQ-Int8': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'XinYuan-VL-2B-Instruct': partial(Qwen2VLChat, model_path='Cylingo/Xinyuan-VL-2B', min_pixels=1280*28*28, max_pixels=16384*28*28),
}

slime_series = {
    'Slime-7B': partial(SliME, model_path='yifanzhang114/SliME-vicuna-7B'),
    'Slime-8B': partial(SliME, model_path='yifanzhang114/SliME-Llama3-8B'),
    'Slime-13B': partial(SliME, model_path='yifanzhang114/SliME-vicuna-13B'),
}

eagle_series={
    'Eagle-X4-8B-Plus': partial(Eagle, model_path='NVEagle/Eagle-X4-8B-Plus'),
    'Eagle-X4-13B-Plus': partial(Eagle, model_path='NVEagle/Eagle-X4-13B-Plus'),
    'Eagle-X5-7B': partial(Eagle, model_path='NVEagle/Eagle-X5-7B'),
    'Eagle-X5-13B': partial(Eagle, model_path='NVEagle/Eagle-X5-13B'),
    'Eagle-X5-13B-Chat': partial(Eagle, model_path='NVEagle/Eagle-X5-13B-Chat'),
    'Eagle-X5-34B-Chat': partial(Eagle, model_path='NVEagle/Eagle-X5-34B-Chat'),
    'Eagle-X5-34B-Plus': partial(Eagle, model_path='NVEagle/Eagle-X5-34B-Plus'),
}

moondream_series={
    'Moondream1': partial(Moondream1, model_path='vikhyatk/moondream1'),
    'Moondream2': partial(Moondream2, model_path='vikhyatk/moondream2'),
}

llama_series={
    'Llama-3.2-11B-Vision-Instruct': partial(llama_vision, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct'),
    'Llama-3.2-90B-Vision-Instruct': partial(llama_vision, model_path='meta-llama/Llama-3.2-90B-Vision-Instruct'),
}

molmo_series={
    'molmoE-1B-0924': partial(molmo, model_path='allenai/MolmoE-1B-0924'),
    'molmo-7B-D-0924': partial(molmo, model_path='allenai/Molmo-7B-D-0924'),
    'molmo-7B-O-0924': partial(molmo, model_path='allenai/Molmo-7B-O-0924'),
    'molmo-72B-0924': partial(molmo, model_path='allenai/Molmo-72B-0924'),
}

kosmos_series={
    'Kosmos2': partial(Kosmos2, model_path='microsoft/kosmos-2-patch14-224')
}

points_series = {
    'POINTS-Yi-1.5-9B-Chat': partial(POINTS, model_path='WePOINTS/POINTS-Yi-1-5-9B-Chat'),
    'POINTS-Qwen-2.5-7B-Chat': partial(POINTS, model_path='WePOINTS/POINTS-Qwen-2-5-7B-Chat'),
}

nvlm_series = {
    'NVLM': partial(NVLM, model_path='nvidia/NVLM-D-72B'), 
}

vintern_series = {
    'Vintern-3B-beta': partial(VinternChat, model_path='5CD-AI/Vintern-3B-beta'),
    'Vintern-1B-v2': partial(VinternChat, model_path='5CD-AI/Vintern-1B-v2'),
}

aria_series = {
    "Aria": partial(Aria, model_path='rhymes-ai/Aria')
}

h2ovl_series = {
    'h2ovl-mississippi-2b': partial(H2OVLChat, model_path='h2oai/h2ovl-mississippi-2b'),
    'h2ovl-mississippi-1b': partial(H2OVLChat, model_path='h2oai/h2ovl-mississippi-800m'),
}

supported_VLM = {}

model_groups = [
    ungrouped, api_models,
    xtuner_series, qwen_series, llava_series, internvl_series, yivl_series,
    xcomposer_series, minigpt4_series, idefics_series, instructblip_series,
    deepseekvl_series, janus_series, minicpm_series, cogvlm_series, wemm_series,
    cambrian_series, chameleon_series, video_models, ovis_series, vila_series,
    mantis_series, mmalaya_series, phi3_series, xgen_mm_series, qwen2vl_series, 
    slime_series, eagle_series, moondream_series, llama_series, molmo_series,
    kosmos_series, points_series, nvlm_series, vintern_series, h2ovl_series, aria_series
]

for grp in model_groups:
    supported_VLM.update(grp)

