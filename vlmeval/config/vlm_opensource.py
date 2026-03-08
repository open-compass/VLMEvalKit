from vlmeval.vlm import *
from functools import partial
import os


ungrouped = {
    "Pixtral-12B": partial(Pixtral, model_path="mistralai/Pixtral-12B-2409"),
}

minicpm_series = {
    "MiniCPM-V-2_6": partial(MiniCPM_V_2_6, model_path="openbmb/MiniCPM-V-2_6"),
    "MiniCPM-o-2_6": partial(MiniCPM_o_2_6, model_path="openbmb/MiniCPM-o-2_6"),
    "MiniCPM-V-4": partial(MiniCPM_V_4, model_path="openbmb/MiniCPM-V-4"),
    "MiniCPM-V-4_5": partial(MiniCPM_V_4_5, model_path="openbmb/MiniCPM-V-4_5"),
}

llava_series = {
    "llava_next_vicuna_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-7b-hf"
    ),
    "llava_next_vicuna_13b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-13b-hf"
    ),
    "llava_next_mistral_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-mistral-7b-hf"
    ),
    "llava_next_yi_34b": partial(LLaVA_Next, model_path="llava-hf/llava-v1.6-34b-hf"),
    "llava_next_llama3": partial(
        LLaVA_Next, model_path="llava-hf/llama3-llava-next-8b-hf"
    ),
    "llava_next_72b": partial(LLaVA_Next, model_path="llava-hf/llava-next-72b-hf"),
    "llava_next_110b": partial(LLaVA_Next, model_path="llava-hf/llava-next-110b-hf"),
    "llava_next_interleave_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-hf"
    ),
    "llava_next_interleave_7b_dpo": partial(
        LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-dpo-hf"
    ),
    "llava-onevision-qwen2-0.5b-ov-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    ),
    "llava-onevision-qwen2-0.5b-si-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    ),
    "llava-onevision-qwen2-7b-ov-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-ov-hf"
    ),
    "llava-onevision-qwen2-7b-si-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-si-hf"
    ),
    "llava_onevision_qwen2_0.5b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-si"
    ),
    "llava_onevision_qwen2_7b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-si"
    ),
    "llava_onevision_qwen2_72b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-si"
    ),
    "llava_onevision_qwen2_0.5b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-ov"
    ),
    "llava_onevision_qwen2_7b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-ov"
    ),
    "llava_onevision_qwen2_72b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-ov-sft"
    ),
    "Aquila-VL-2B": partial(LLaVA_OneVision, model_path="BAAI/Aquila-VL-2B-llava-qwen"),
    "llava_video_qwen2_7b": partial(
        LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
    ),
    "llava_video_qwen2_72b": partial(
        LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-72B-Qwen2"
    ),
}

interns1_mini = {
    "Intern-S1-mini": partial(
        InternS1Chat, model_path="/mnt/shared-storage-user/mllm/lijinsong/models/Intern-S1-mini/"
    ),
}

internvl = {
    "InternVL-Chat-V1-1": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-1", version="V1.1"
    ),
    "InternVL-Chat-V1-2": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-2", version="V1.2"
    ),
    "InternVL-Chat-V1-2-Plus": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-2-Plus", version="V1.2"
    ),
    "InternVL-Chat-V1-5": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL-Chat-V1-5",
        version="V1.5",
    )
}

mini_internvl = {
    "Mini-InternVL-Chat-2B-V1-5": partial(
        InternVLChat, model_path="OpenGVLab/Mini-InternVL-Chat-2B-V1-5", version="V1.5"
    ),
    "Mini-InternVL-Chat-4B-V1-5": partial(
        InternVLChat, model_path="OpenGVLab/Mini-InternVL-Chat-4B-V1-5", version="V1.5"
    ),
}

internvl2 = {
    "InternVL2-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-1B", version="V2.0"
    ),
    "InternVL2-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-2B", version="V2.0"
    ),
    "InternVL2-4B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-4B", version="V2.0"
    ),
    "InternVL2-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-8B", version="V2.0"
    ),
    "InternVL2-26B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-26B", version="V2.0"
    ),
    "InternVL2-40B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-40B", version="V2.0"
    ),
    "InternVL2-76B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-Llama3-76B", version="V2.0"
    ),
    "InternVL2-8B-MPO": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-8B-MPO", version="V2.0"
    ),
    "InternVL2-8B-MPO-CoT": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
}

internvl2_5 = {
    "InternVL2_5-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-1B", version="V2.0"
    ),
    "InternVL2_5-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-2B", version="V2.0"
    ),
    "QTuneVL1-2B": partial(
        InternVLChat, model_path="hanchaow/QTuneVL1-2B", version="V2.0"
    ),
    "InternVL2_5-4B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-4B", version="V2.0"
    ),
    "InternVL2_5-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0"
    ),
    "InternVL2_5-26B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-26B", version="V2.0"
    ),
    "InternVL2_5-38B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-38B", version="V2.0"
    ),
    "InternVL2_5-78B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-78B", version="V2.0"
    ),
    # InternVL2.5 series with Best-of-N evaluation
    "InternVL2_5-8B-BoN-8": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0",
        best_of_n=8, reward_model_path="OpenGVLab/VisualPRM-8B",
    ),
}

internvl2_5_mpo = {
    "InternVL2_5-1B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-1B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-2B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-2B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-4B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-4B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-26B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-26B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-38B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-38B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-78B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-78B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-GUI": partial(
        InternVLChat,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/models/internvl2_5_8b_internlm2_5_7b_dynamic_res_stage1",
        version="V2.0",
        max_new_tokens=512,
        screen_parse=False,
    ),
     "InternVL3-7B-GUI": partial(
        InternVLChat,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/GUI/checkpoints/internvl3_7b_dynamic_res_stage1_56/",
        version="V2.0",
        max_new_tokens=512,
        screen_parse=False,
    ),
}

internvl3 = {
    "InternVL3-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-1B", version="V2.0"
    ),
    "InternVL3-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-2B", version="V2.0"
    ),
    "InternVL3-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-8B", version="V2.0",
    ),
    "InternVL3-9B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-9B", version="V2.0"
    ),
    "InternVL3-14B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-14B", version="V2.0"
    ),
    "InternVL3-38B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-38B", version="V2.0"
    ),
    "InternVL3-78B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-78B", version="V2.0"
    ),
}

internvl3_5 = {
    "InternVL3_5-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-1B", version="V2.0"
    ),
    "InternVL3_5-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-2B", version="V2.0"
    ),
    "InternVL3_5-4B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-4B", version="V2.0"
    ),
    "InternVL3_5-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-8B", version="V2.0"
    ),
    "InternVL3_5-14B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-14B", version="V2.0"
    ),
    "InternVL3_5-GPT-OSS-20B-A4B-Preview": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview", version="V2.0"
    ),
    "InternVL3_5-30B-A3B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-30B-A3B", version="V2.0"
    ),
    "InternVL3_5-38B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-38B", version="V2.0"
    ),
    "InternVL3_5-241B-A28B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-241B-A28B", version="V2.0"
    ),

    "InternVL3_5-1B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-1B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-2B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-2B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-4B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-4B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-8B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-8B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-14B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-14B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-GPT-OSS-20B-A4B-Preview-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-30B-A3B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-30B-A3B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-38B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-38B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-241B-A28B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-241B-A28B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
}

qwen3vl_series = {
    "Qwen3-VL-235B-A22B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-235B-A22B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.0,
        max_new_tokens=32768
    ),
    "Qwen3-VL-235B-A22B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-235B-A22B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.0,
        max_new_tokens=32768
    ),
    "Qwen3-VL-30B-A3B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-30B-A3B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.0,
        max_new_tokens=32768
    ),
    "Qwen3-VL-30B-A3B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-30B-A3B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.0,
        max_new_tokens=32768
    ),
    "Qwen3-VL-8B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-8B-Thinking",
        use_custom_prompt=False,
        temperature=0.7,
        max_new_tokens=16384
    ),
    "Qwen3-VL-4B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-4B-Thinking",
        use_custom_prompt=False,
        temperature=0.7,
        max_new_tokens=16384
    ),
    "Qwen3-VL-8B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=16384,
    ),
    "Qwen3-VL-4B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-4B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=16384
    ),
}

sail_series = {
    "SAIL-VL-2B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-2B"),
    "SAIL-VL-1.5-2B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d5-2B", use_msac = True),
    "SAIL-VL-1.5-8B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d5-8B", use_msac = True),
    "SAIL-VL-1.6-8B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d6-8B", use_msac = True),
    "SAIL-VL-1.7-Thinking-2B-2507": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d7-Thinking-2B-2507", use_msac = True, use_cot=True, max_new_tokens=4096),
    "SAIL-VL-1.7-Thinking-8B-2507": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d7-Thinking-8B-2507", use_msac = True, use_cot=True, max_new_tokens=4096),
    "SAIL-VL2-2B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL2-2B", use_msac = True),
    "SAIL-VL2-8B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL2-8B", use_msac = True),
}

xcomposer_series = {
    "XComposer2_4KHD": partial(
        XComposer2_4KHD, model_path="internlm/internlm-xcomposer2-4khd-7b"
    ),
    "XComposer2d5": partial(
        XComposer2d5, model_path="internlm/internlm-xcomposer2d5-7b"
    ),
}

smolvlm_series = {
    "SmolVLM-256M": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-256M-Instruct"),
    "SmolVLM-500M": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-500M-Instruct"),
    "SmolVLM": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-Instruct"),
    "SmolVLM-DPO": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-Instruct-DPO"),
    "SmolVLM-Synthetic": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-Synthetic"),
    "SmolVLM2-256M": partial(
        SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    ),
    "SmolVLM2-500M": partial(
        SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    ),
    "SmolVLM2": partial(SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
}

deepseek_series = {
    "deepseek_vl2_tiny": partial(
        DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2-tiny"
    ),
    "deepseek_vl2_small": partial(
        DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2-small"
    ),
    "deepseek_vl2": partial(DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2"),
    "deepseek_ocr": partial(DeepSeekOCR, model_path="deepseek-ai/DeepSeek-OCR"),
}

glm_series = {
    "glm-4v-9b": partial(GLM4v, model_path="THUDM/glm-4v-9b"),
    "GLM4_1VThinking-9b": partial(GLMThinking, model_path="THUDM/GLM-4.1V-9B-Thinking"),
    "GLM4_5V": partial(GLMThinking, model_path="THUDM/GLM-4.5V"),
}

cambrian_series = {
    "cambrian_8b": partial(Cambrian, model_path="nyu-visionx/cambrian-8b"),
    "cambrian_13b": partial(Cambrian, model_path="nyu-visionx/cambrian-13b"),
    "cambrian_34b": partial(Cambrian, model_path="nyu-visionx/cambrian-34b"),
    "cambrian-s-0.5b": partial(CambrianS, model_path="nyu-visionx/Cambrian-S-0.5B"),
    "cambrian-s-1.5b": partial(CambrianS, model_path="nyu-visionx/Cambrian-S-1.5B"),
    "cambrian-s-3b": partial(CambrianS, model_path="nyu-visionx/Cambrian-S-3B"),
    "cambrian-s-7b": partial(CambrianS, model_path="nyu-visionx/Cambrian-S-7B"),
}

vila_series = {
    "VILA1.5-3b": partial(VILA, model_path="Efficient-Large-Model/VILA1.5-3b"),
    "Llama-3-VILA1.5-8b": partial(
        VILA, model_path="Efficient-Large-Model/Llama-3-VILA1.5-8b"
    ),
    "VILA1.5-13b": partial(VILA, model_path="Efficient-Large-Model/VILA1.5-13b"),
    "VILA1.5-40b": partial(VILA, model_path="Efficient-Large-Model/VILA1.5-40b"),
    "NVILA-8B": partial(NVILA, model_path="Efficient-Large-Model/NVILA-8B"),
    "NVILA-15B": partial(NVILA, model_path="Efficient-Large-Model/NVILA-15B"),
}

ovis_series = {
    "Ovis1.5-Llama3-8B": partial(Ovis, model_path="AIDC-AI/Ovis1.5-Llama3-8B"),
    "Ovis1.5-Gemma2-9B": partial(Ovis, model_path="AIDC-AI/Ovis1.5-Gemma2-9B"),
    "Ovis1.6-Gemma2-9B": partial(Ovis1_6, model_path="AIDC-AI/Ovis1.6-Gemma2-9B"),
    "Ovis1.6-Llama3.2-3B": partial(Ovis1_6, model_path="AIDC-AI/Ovis1.6-Llama3.2-3B"),
    "Ovis1.6-Gemma2-27B": partial(
        Ovis1_6_Plus, model_path="AIDC-AI/Ovis1.6-Gemma2-27B"
    ),
    "Ovis2-1B": partial(Ovis2, model_path="AIDC-AI/Ovis2-1B"),
    "Ovis2-2B": partial(Ovis2, model_path="AIDC-AI/Ovis2-2B"),
    "Ovis2-4B": partial(Ovis2, model_path="AIDC-AI/Ovis2-4B"),
    "Ovis2-8B": partial(Ovis2, model_path="AIDC-AI/Ovis2-8B"),
    "Ovis2-16B": partial(Ovis2, model_path="AIDC-AI/Ovis2-16B"),
    "Ovis2-34B": partial(Ovis2, model_path="AIDC-AI/Ovis2-34B"),
    "Ovis-U1-3B": partial(OvisU1, model_path="AIDC-AI/Ovis-U1-3B"),
    "Ovis2.5-2B": partial(Ovis2_5, model_path="AIDC-AI/Ovis2.5-2B"),
    "Ovis2.5-9B": partial(Ovis2_5, model_path="AIDC-AI/Ovis2.5-9B")
}

phi_series = {
    "Phi-3-Vision": partial(
        Phi3Vision, model_path="microsoft/Phi-3-vision-128k-instruct"
    ),
    "Phi-3.5-Vision": partial(
        Phi3_5Vision, model_path="microsoft/Phi-3.5-vision-instruct"
    ),
    'Phi-4-Vision': partial(
        Phi4Multimodal, model_path='microsoft/Phi-4-multimodal-instruct'
    ),
}

qwen2vl_series = {
    "Qwen2-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int4": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int8": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int4": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int8": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "XinYuan-VL-2B-Instruct": partial(
        Qwen2VLChat,
        model_path="Cylingo/Xinyuan-VL-2B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2.5-VL-3B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-3B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-32B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-32B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "MiMo-VL-7B-SFT": partial(
        Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True
    ),
    "MiMo-VL-7B-RL": partial(
        Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-RL",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True
    ),
    "Qwen2.5-VL-72B-Instruct-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-Omni-7B-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
        use_audio_in_video=True, # set use audio in video
    ),
    "Qwen2.5-Omni-7B": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    )
}

moondream_series = {
    "Moondream1": partial(Moondream1, model_path="vikhyatk/moondream1"),
    "Moondream2": partial(Moondream2, model_path="vikhyatk/moondream2"),
    "Moondream3": partial(Moondream3, model_path="moondream/moondream3-preview"),
}

llama_series = {
    "Llama-3.2-11B-Vision-Instruct": partial(
        llama_vision, model_path="meta-llama/Llama-3.2-11B-Vision-Instruct"
    ),
    "LLaVA-CoT": partial(llama_vision, model_path="Xkev/Llama-3.2V-11B-cot"),
    "Llama-3.2-90B-Vision-Instruct": partial(
        llama_vision, model_path="meta-llama/Llama-3.2-90B-Vision-Instruct"
    ),
    "Llama-4-Scout-17B-16E-Instruct": partial(
        llama4, model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct", use_vllm=True
    ),
}

gemma_series = {
    "paligemma-3b-mix-448": partial(
        PaliGemma, model_path="google/paligemma-3b-mix-448"
    ),

    # 3B
    "paligemma2-3b-pt-224":  partial(PaliGemma, model_path="google/paligemma2-3b-pt-224"),
    "paligemma2-3b-pt-448":  partial(PaliGemma, model_path="google/paligemma2-3b-pt-448"),
    "paligemma2-3b-mix-224": partial(PaliGemma, model_path="google/paligemma2-3b-mix-224"),
    "paligemma2-3b-mix-448": partial(PaliGemma, model_path="google/paligemma2-3b-mix-448"),

    # 10B
    "paligemma2-10b-pt-224":  partial(PaliGemma, model_path="google/paligemma2-10b-pt-224"),
    "paligemma2-10b-pt-448":  partial(PaliGemma, model_path="google/paligemma2-10b-pt-448"),
    "paligemma2-10b-mix-224": partial(PaliGemma, model_path="google/paligemma2-10b-mix-224"),
    "paligemma2-10b-mix-448": partial(PaliGemma, model_path="google/paligemma2-10b-mix-448"),

    # 28B
    "paligemma2-28b-pt-224":  partial(PaliGemma, model_path="google/paligemma2-28b-pt-224"),
    "paligemma2-28b-pt-448":  partial(PaliGemma, model_path="google/paligemma2-28b-pt-448"),
    "paligemma2-28b-mix-224": partial(PaliGemma, model_path="google/paligemma2-28b-mix-224"),
    "paligemma2-28b-mix-448": partial(PaliGemma, model_path="google/paligemma2-28b-mix-448"),

    'Gemma3-4B': partial(Gemma3, model_path='google/gemma-3-4b-it'),
    'Gemma3-12B': partial(Gemma3, model_path='google/gemma-3-12b-it'),
    'Gemma3-27B': partial(Gemma3, model_path='google/gemma-3-27b-it')
}

aguvis_series = {
    "aguvis_7b": partial(
        Qwen2VLChatAguvis,
        model_path=os.getenv(
            "EVAL_MODEL",
            "xlangai/Aguvis-7B-720P",
        ),
        min_pixels=256 * 28 * 28,
        max_pixels=46 * 26 * 28 * 28,
        use_custom_prompt=False,
        mode='grounding',
    )
}

cosmos_series = {
    'Cosmos-Reason1-7B': partial(Cosmos, model_path='nvidia/Cosmos-Reason1-7B', use_vllm=True),
}

keye_series = {
    "Keye-VL-1.5-8B-auto":partial(KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B"),
    "Keye-VL-1.5-8B-think":partial(KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B", think=True),
    "Keye-VL-1.5-8B-nothink":partial(KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B", no_think=True),
    "Keye-VL-8B-Preview-think":partial(KeyeChat, model_path="Kwai-Keye/Keye-VL-8B-Preview", think=True),
}

qwen3_5_series = {
    # vllm serve command example: 
    # vllm serve Qwen/Qwen3.5-122B-A10B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3
    # Require transformers >=5.2.0
    "Qwen3.5-397B-A17B": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3.5-397B-A17B",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-122B-A10B": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3.5-122B-A10B",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-35B-A3B": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3.5-35B-A3B",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-27B": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3.5-27B",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
}

internvl_groups = [
    internvl, internvl2, internvl2_5, mini_internvl, internvl2_5_mpo,
    internvl3, internvl3_5
]
internvl_series = {}
for group in internvl_groups:
    internvl_series.update(group)

interns1_groups = [
    interns1_mini
]
interns1_series = {}
for group in interns1_groups:
    interns1_series.update(group)

VLM_OPENSOURCE_GROUPS = [
    ungrouped, llava_series, internvl_series, xcomposer_series, deepseek_series,
    minicpm_series, glm_series, cambrian_series, ovis_series, vila_series, 
    phi_series, qwen2vl_series, qwen3vl_series, moondream_series, 
    llama_series, smolvlm_series, sail_series, gemma_series, aguvis_series, 
    cosmos_series, keye_series, interns1_series, qwen3_5_series
]
