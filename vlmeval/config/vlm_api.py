from vlmeval.api import *
from functools import partial
import os


openrouter_apis = {
    'openrouter-gemini-2.5-pro': partial(
        OpenRouter,
        model='google/gemini-2.5-pro',
        retry=3,
        temperature=0, 
        verbose=True,
        max_tokens=2**14,
        timeout=1200
    ),
    'openrouter-gemini-2.5-flash': partial(
        OpenRouter,
        model='google/gemini-2.5-flash',
        retry=3,
        temperature=0, 
        verbose=True,
        max_tokens=2**14,
        timeout=1200
    ),
    'GPT-5.4-OR': partial(
        OpenRouter,
        model='openai/gpt-5.4',
        retry=3,
        temperature=1, 
        verbose=False,
        max_tokens=2**15,
        timeout=3600, 
        reasoning_effort='high'
    ),
    'Gemini-3.1-Pro-OR': partial(
        OpenRouter,
        model='google/gemini-3.1-pro-preview',
        retry=3,
        temperature=0, 
        verbose=False,
        max_tokens=2**15,
        timeout=3600, 
        reasoning_effort='high'
    ),
    'Gemini-3.1-Flash-Lite-OR': partial(
        OpenRouter,
        model='google/gemini-3.1-flash-lite-preview',
        retry=3,
        temperature=0, 
        verbose=False,
        max_tokens=2**15,
        timeout=3600, 
        reasoning_effort='high'
    ),
    'Claude-4.6-Opus-OR': partial(
        OpenRouter,
        model='anthropic/claude-opus-4.6',
        retry=3,
        temperature=1, 
        verbose=False,
        max_tokens=2**15,
        timeout=3600, 
        reasoning_effort='high'
    ),
}

o1_key = os.environ.get('O1_API_KEY', None)
o1_base = os.environ.get('O1_API_BASE', None)
o1_apis = {
    'o1': partial(
        GPT4V,
        model="o1-2024-12-17",
        key=o1_key,
        api_base=o1_base,
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,

    ),
    'o3': partial(
        GPT4V,
        model="o3-2025-04-16",
        key=o1_key,
        api_base=o1_base,
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,
    ),
    'o4-mini': partial(
        GPT4V,
        model="o4-mini-2025-04-16",
        key=o1_key,
        api_base=o1_base,
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,
    ),
}

openai_apis = {
    # GPT
    "GPT4V": partial(
        GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_HIGH": partial(
        GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409": partial(
        GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409_HIGH": partial(
        GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o": partial(
        GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4o_HIGH": partial(
        GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20240806": partial(
        GPT4V,
        model="gpt-4o-2024-08-06",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20241120": partial(
        GPT4V,
        model="gpt-4o-2024-11-20",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "ChatGPT4o": partial(
        GPT4V,
        model="chatgpt-4o-latest",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_MINI": partial(
        GPT4V,
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4.5": partial(
        GPT4V,
        model='gpt-4.5-preview-2025-02-27',
        temperature=0,
        timeout=600,
        img_size=-1,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-mini-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-mini-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-nano-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-nano-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-5-2025-08-07": partial(
        GPT4V,
        model="gpt-5-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    "gpt-5-mini-2025-08-07": partial(
        GPT4V,
        model="gpt-5-mini-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    "gpt-5-nano-2025-08-07": partial(
        GPT4V,
        model="gpt-5-nano-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    "gpt-5.1-thinking": partial(
        GPT4V,
        model="gpt-5.1-thinking",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=600
    ),
}

gemini_apis = {
    # Gemini
    "GeminiPro1-0": partial(
        Gemini, model="gemini-1.0-pro", temperature=0, retry=10
    ),  # now GeminiPro1-0 is only supported by vertex backend
    "GeminiPro1-5": partial(
        Gemini, model="gemini-1.5-pro", temperature=0, retry=10
    ),
    "GeminiFlash1-5": partial(
        Gemini, model="gemini-1.5-flash", temperature=0, retry=10
    ),
    "GeminiPro1-5-002": partial(
        GPT4V, model="gemini-1.5-pro-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash1-5-002": partial(
        GPT4V, model="gemini-1.5-flash-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash2-0": partial(
        Gemini, model="gemini-2.0-flash", temperature=0, retry=10
    ),
    "GeminiFlashLite2-0": partial(
        Gemini, model="gemini-2.0-flash-lite", temperature=0, retry=10
    ),
    "GeminiFlash2-5": partial(
        Gemini, model="gemini-2.5-flash", temperature=0, retry=10
    ),
    "GeminiPro2-5": partial(
        Gemini, model="gemini-2.5-pro", temperature=0, retry=10
    ),
}

qwen_vl_apis = {
    # Qwen-VL
    "QwenVLPlus": partial(QwenVLAPI, model="qwen-vl-plus", temperature=0, retry=10),
    "QwenVLMax": partial(QwenVLAPI, model="qwen-vl-max", temperature=0, retry=10),
    "QwenVLMax-250408": partial(QwenVLAPI, model="qwen-vl-max-2025-04-08", temperature=0, retry=10),
    "Qwen-VL-Max-20250813": partial(
        Qwen2VLAPI,
        model="qwen-vl-max-2025-08-13",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_length=8192,
    ),
    "Qwen-VL-Max-0809": partial(
        Qwen2VLAPI,
        model="qwen-vl-max-0809",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen-VL-Plus-0809": partial(
        Qwen2VLAPI,
        model="qwen-vl-plus-0809",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
}

step_apis = {
    # Step1V
    "Step1V": partial(
        GPT4V,
        model="step-1v-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step1.5V-mini": partial(
        GPT4V,
        model="step-1.5v-mini",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step1o": partial(
        GPT4V,
        model="step-1o-vision-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    )
}

kimi_apis = {
    "Kimi-K2.5": partial(
        GPT4V, 
        model='kimi-k2.5',
        api_base='https://api.moonshot.ai/v1/chat/completions',
        temperature=1,
        top_p=0.95, 
        retry=3,
        img_size=-1,
        img_detail="high",
        timeout=3600,
        max_tokens=16384,
        verbose=False,
        extra_body={'thinking': {'type': 'enabled'}},
        key='KIMI_API_KEY'
    ),
    "Kimi-K2.5-MinEdge": partial(
        GPT4V, 
        model='kimi-k2.5',
        api_base='https://api.moonshot.ai/v1/chat/completions',
        temperature=1,
        top_p=0.95, 
        retry=3,
        img_size=-1,
        img_detail="high",
        timeout=3600,
        max_tokens=16384,
        min_edge=672,
        verbose=False,
        key='KIMI_API_KEY'
    ),
    "moonshot-v1-8k": partial(
        GPT4V,
        model="moonshot-v1-8k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-32k": partial(
        GPT4V,
        model="moonshot-v1-32k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-128k": partial(
        GPT4V,
        model="moonshot-v1-128k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
}

glm_apis = {
    # GLM4V
    "GLM4V": partial(GLMVisionAPI, model="glm4v-biz-eval", temperature=0, retry=10),
    "GLM4V_PLUS": partial(GLMVisionAPI, model="glm-4v-plus", temperature=0, retry=10),
    "GLM4V_PLUS_20250111": partial(
        GLMVisionAPI, model="glm-4v-plus-0111", temperature=0, retry=10
    ),
}

minimax_apis = {
    # MiniMax abab
    "abab6.5s": partial(
        GPT4V,
        model="abab6.5s-chat",
        api_base="https://api.minimax.chat/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "abab7-preview": partial(
        GPT4V,
        model="abab7-chat-preview",
        api_base="https://api.minimax.chat/v1/chat/completions",
        temperature=0,
        retry=10,
    )
}

hunyuan_apis = {
    "HunYuan-Vision": partial(
        HunyuanVision, model="hunyuan-vision", temperature=0, retry=10
    ),
    "HunYuan-Standard-Vision": partial(
        HunyuanVision, model="hunyuan-standard-vision", temperature=0, retry=10
    ),
    "HunYuan-Large-Vision": partial(
        HunyuanVision, model="hunyuan-large-vision", temperature=0, retry=10
    ),
    "Qwen2.5-VL-32B-Instruct-SiliconFlow": partial(
        SiliconFlowAPI, model="Qwen/Qwen2.5-VL-32B-Instruct", temperature=0, retry=10),
}

grok_apis = {
    "grok-vision-beta": partial(
        GPT4V,
        model="grok-vision-beta",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-2-vision-1212": partial(
        GPT4V,
        model="grok-2-vision",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-4-0709": partial(
        GPT4V,
        model="grok-4-0709",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=1200,
        max_tokens=16384
    ),
}

ernie_apis = {    
    'ernie4.5-turbo': partial(
        GPT4V,
        model='ernie-4.5-turbo-vl-32k',
        temperature=0,
        retry=3,
        max_tokens=12000,
    ),
    'ernie4.5-a3b': partial(
        GPT4V,
        model='ernie-4.5-vl-28b-a3b',
        temperature=0,
        retry=3,
        max_tokens=8000,
    )
}

seed_apis = {
    "Seed2.0-Pro": partial(
        DoubaoVL, model="doubao-seed-2-0-pro-260215",
        temperature=1, top_p=0.95, timeout=3600, max_tokens=65536, detail='high'),
    "Seed2.0-Lite": partial(
        DoubaoVL, model="doubao-seed-2-0-lite-260215",
        temperature=1, top_p=0.95, timeout=3600, max_tokens=65536, detail='high'),
    "Seed2.0-Mini": partial(
        DoubaoVL, model="doubao-seed-2-0-mini-260215", 
        temperature=1, top_p=0.95, timeout=3600, max_tokens=65536, detail='high'),
    "Seed2.0-Pro-xhigh": partial(
        DoubaoVL, model="doubao-seed-2-0-pro-260215",
        temperature=1, top_p=0.95, timeout=3600, max_tokens=65536, detail='xhigh'),
    "Seed2.0-Lite-xhigh": partial(
        DoubaoVL, model="doubao-seed-2-0-lite-260215",
        temperature=1, top_p=0.95, timeout=3600, max_tokens=65536, detail='xhigh'),
}

import copy as cp
openai_apis['gpt-5'] = cp.deepcopy(openai_apis['gpt-5-2025-08-07'])
openai_apis['gpt-5-mini'] = cp.deepcopy(openai_apis['gpt-5-mini-2025-08-07'])
openai_apis['gpt-5-nano'] = cp.deepcopy(openai_apis['gpt-5-nano-2025-08-07'])

VLM_API_GROUPS = [
    o1_apis, openai_apis, openrouter_apis, gemini_apis, 
    qwen_vl_apis, step_apis, kimi_apis, glm_apis, 
    minimax_apis, hunyuan_apis, grok_apis, ernie_apis, 
    seed_apis, 
]
