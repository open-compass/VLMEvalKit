from vlmeval.api import *
from functools import partial
import os

ug_apis = {
    "GPT-IMAGE-1-LOW": partial(
        GPTImage,
        model="gpt-image-1",
        temperature=0,
        timeout=1200,
        retry=3,
        verbose=True,
        quality='low'
    ),
    "Qwen-IMAGE-EDIT-PLUS": partial(
        QwenImage,
        model="qwen-image-edit-plus",
        temperature=0,
        timeout=1200,
        retry=3,
    ),
    "GPT-IMAGE-1": partial(
        GPTImage,
        model="gpt-image-1",
        temperature=0,
        timeout=1200,
        retry=3,
        verbose=True,
    ),
    "Seedream4Image": partial(
        SeedreamImage,
        model="ep-20251224010846-qwmqh",
        temperature=0,
        timeout=1200,
        retry=3,
        verbose=True,
    ),
    "Seedream4_5Image": partial(
        SeedreamImage,
        model="ep-20251224010811-5cw5f",
        temperature=0,
        timeout=1200,
        retry=3,
        size='2K',   # 2K or 4K
        verbose=True,
    ),
    "GeminiFlash2.5-Gen": partial(
        GeminiImage,
        model="gemini-2.5-flash-image",
        temperature=0,
        timeout=1200,
        retry=3,
        resp_modalities=['IMAGE']
    ),
    "GeminiFlash2.5-UG": partial(
        GeminiImage,
        model="gemini-2.5-flash-image",
        temperature=0,
        timeout=1200,
        retry=3,
        resp_modalities=['IMAGE', 'TEXT']
    ),
    "GeminiFlash3-UG": partial(
        GeminiImage,
        model="gemini-3-pro-image-preview",
        temperature=0,
        timeout=1200,
        retry=3,
        resp_modalities=['IMAGE', 'TEXT']
    )
}

psm_apis = {
    'UG_2b5_debug_PSM': partial(
        UGPSMAPI,
        # Support Multiple PSMs (for the same model), will sample according to the prob set.
        # psm=[
        #     'seed.ray.server_0uljzoudtv692d09fa.service.wlby'
        # ],
        # prob=[2, 1],
        psm='seed.ray.server_9du44wnjrz693fb436.service.wlby',
        # system_prompt="""You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
        #                 <think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.
        #                 You have different modes of thinking:
        #                 Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
        #                 Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking.
        #                 No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer.
        #                 Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.
        #                 Unrestricted think mode with image generation: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem. You can insert generated images in the answer or thinking process.
        #                 No think mode with image generation: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. You can insert generated images only in the answer. 
        #                 In the mode which allows image generation, when images need to be generated, please first output <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]>, and then output a section of image generation parameters enclosed by <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[EOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>. Between <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>, you may describe the planned image’s content, style, composition, details, and any other relevant attributes if necessary, or you may omit such descriptions and proceed directly to the parameter block.
        #                 Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.
        #                 Provided Mode(s):
        #                 No think mode with image generation""",
        verbose=False,
        retry=3,
        temperature=0,
        timeout=1200,
        max_tokens=2**14,
        custom_ug_params = {
            "is_plain_mode": True,
            "infer_mode": "interleave", # [force_und, force_gen, interleave] 若没有指定system_prompt，则使用对应模式的默认system prompt
            "cfg_mode": "naive_text", 
            "diffusion_configs": {"height": 200, "width": 200, "cfg_scale": (1.5, 5.0)},
            "training_mode": "sft"
        },
    ),
    'UG_2b5_RL_PSM': partial(
        UGPSMAPI,
        psm='seed.ray.server_ndb8pqmfrn693fd22e.service.wlby',
        verbose=False,
        retry=3,
        temperature=0,
        timeout=1200,
        max_tokens=2**14,
        custom_ug_params = {
            "is_plain_mode": True,
            "infer_mode": "force_und", # [force_und, force_gen, interleave] 若没有指定system_prompt，则使用对应模式的默认system prompt
            "cfg_mode": "naive_text", 
            "diffusion_configs": {"height": 200, "width": 200, "cfg_scale": (1.5, 5.0)},
            "training_mode": "sft"
        },
        vlm_vae_eval=False,
    ),
    'Interleave_v1_23B_PSM': partial(
        UGPSMAPI,
        psm='seed.ray.server_521felzerr6959eead.service.wlby',
        verbose=False,
        retry=3,
        temperature=0,
        timeout=1200,
        max_tokens=2**14,
        system_prompt="""You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking.
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer.
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.
Unrestricted think mode with image generation: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem. You can insert generated images in the answer or thinking process.
No think mode with image generation: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. You can insert generated images only in the answer. 

In the mode which allows image generation, when images need to be generated, please first output <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]>, and then output a section of image generation parameters enclosed by <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[EOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>. Between <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>, you may describe the planned image’s content, style, composition, details, and any other relevant attributes if necessary, or you may omit such descriptions and proceed directly to the parameter block.

Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.

Provided Mode(s):
Unrestricted think mode with image generation""",
        custom_ug_params = {
            "is_plain_mode": True,
            "infer_mode": "interleave", # [force_und, force_gen, interleave] 若没有指定system_prompt，则使用对应模式的默认system prompt
            "cfg_mode": "naive_text", 
            "diffusion_configs": {"height": 200, "width": 200, "cfg_scale": (1.5, 5.0)},
            "training_mode": "sft"
        },
        vlm_vae_eval=True,
    ),
    'sft32b_think_interleave_PSM': partial(
        UGPSMAPI,
        psm='seed.ray.server_e4xmhmdeox69633a1f.service.wlby',
        verbose=False,
        retry=3,
        temperature=0,
        timeout=1200,
        max_tokens=2**14,
        convert_to_seedquery=True,
        system_prompt="""You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.
You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking.
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer.
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.
Unrestricted think mode with image generation: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem. You can insert generated images in the answer or thinking process.
No think mode with image generation: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. You can insert generated images only in the answer. 
In the mode which allows image generation, when images need to be generated, please first output <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]>, and then output a section of image generation parameters enclosed by <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[EOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>. Between <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>, you may describe the planned image’s content, style, composition, details, and any other relevant attributes if necessary, or you may omit such descriptions and proceed directly to the parameter block.
Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.
Provided Mode(s):
Unrestricted think mode with image generation""",
        custom_ug_params = {
            "is_plain_mode": True,
            "infer_mode": "interleave", # [force_und, force_gen, interleave] 若没有指定system_prompt，则使用对应模式的默认system prompt
            "cfg_mode": "naive_text", 
            "diffusion_configs": {"height": 200, "width": 200, "cfg_scale": (1.5, 5.0)},
            "training_mode": "sft"
        },
        vlm_vae_eval=True, # 评测理解任务时候是否要加入vae embedding
    ),
    'sft32b_think_gen_PSM': partial(
        UGPSMAPI,
        psm='seed.ray.server_e4xmhmdeox69633a1f.service.wlby',
        verbose=False,
        retry=3,
        temperature=0,
        timeout=1200,
        max_tokens=2**14,
        convert_to_seedquery=True,
        system_prompt="""You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.
You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking.
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer.
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.
Unrestricted think mode with image generation: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem. You can insert generated images in the answer or thinking process.
No think mode with image generation: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. You can insert generated images only in the answer. 
In the mode which allows image generation, when images need to be generated, please first output <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]>, and then output a section of image generation parameters enclosed by <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[EOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>. Between <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>, you may describe the planned image’s content, style, composition, details, and any other relevant attributes if necessary, or you may omit such descriptions and proceed directly to the parameter block.
Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.
Provided Mode(s):
Unrestricted think mode with image generation""",
        custom_ug_params = {
            "is_plain_mode": True,
            "infer_mode": "force_gen", # [force_und, force_gen, interleave] 若没有指定system_prompt，则使用对应模式的默认system prompt
            "cfg_mode": "naive_text", 
            "diffusion_configs": {"height": 200, "width": 200, "cfg_scale": (1.5, 5.0)},
            "training_mode": "sft"
        },
        vlm_vae_eval=True, # 评测理解任务时候是否要加入vae embedding
    ),
    'sft32b_nothink_interleave_PSM': partial(
        UGPSMAPI,
        psm='seed.ray.server_e4xmhmdeox69633a1f.service.wlby',
        verbose=False,
        retry=3,
        temperature=0,
        timeout=1200,
        max_tokens=2**14,
        convert_to_seedquery=True,
        system_prompt="""You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking.
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer.
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.
Unrestricted think mode with image generation: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem. You can insert generated images in the answer or thinking process.
No think mode with image generation: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. You can insert generated images only in the answer. 

In the mode which allows image generation, when images need to be generated, please first output <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]>, and then output a section of image generation parameters enclosed by <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[EOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>. Between <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>, you may describe the planned image’s content, style, composition, details, and any other relevant attributes if necessary, or you may omit such descriptions and proceed directly to the parameter block.

Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.

Provided Mode(s):
No think mode with image generation""",
        custom_ug_params = {
            "is_plain_mode": True,
            "infer_mode": "interleave", # [force_und, force_gen, interleave] 若没有指定system_prompt，则使用对应模式的默认system prompt
            "cfg_mode": "naive_text", 
            "diffusion_configs": {"height": 200, "width": 200, "cfg_scale": (1.5, 5.0)},
            "training_mode": "sft"
        },
        vlm_vae_eval=True, # 评测理解任务时候是否要加入vae embedding
    ),
    'sft32b_nothink_gen_PSM': partial(
        UGPSMAPI,
        psm='seed.ray.server_e4xmhmdeox69633a1f.service.wlby',
        verbose=False,
        retry=3,
        temperature=0,
        timeout=1200,
        max_tokens=2**14,
        convert_to_seedquery=True,
        system_prompt="""You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking.
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer.
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.
Unrestricted think mode with image generation: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem. You can insert generated images in the answer or thinking process.
No think mode with image generation: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. You can insert generated images only in the answer. 

In the mode which allows image generation, when images need to be generated, please first output <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]>, and then output a section of image generation parameters enclosed by <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[EOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>. Between <[SOG_never_used_51bce0c785ca2f68081bfa7d91973934]> and <[SOGP_never_used_51bce0c785ca2f68081bfa7d91973934]>, you may describe the planned image’s content, style, composition, details, and any other relevant attributes if necessary, or you may omit such descriptions and proceed directly to the parameter block.

Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.

Provided Mode(s):
No think mode with image generation""",
        custom_ug_params = {
            "is_plain_mode": True,
            "infer_mode": "force_gen", # [force_und, force_gen, interleave] 若没有指定system_prompt，则使用对应模式的默认system prompt
            "cfg_mode": "naive_text", 
            "diffusion_configs": {"height": 200, "width": 200, "cfg_scale": (1.5, 5.0)},
            "training_mode": "sft"
        },
        vlm_vae_eval=True, # 评测理解任务时候是否要加入vae embedding
    ),
    'seeddream5.0m-ct-160k_PSM': partial(
        UGPSMAPI,
        psm='seed.ray.server_a21kxom7f56954a5b5.service.wlby',
        verbose=False,
        retry=3,
        temperature=0,
        timeout=1200,
        max_tokens=2**14,
        convert_to_seedquery=True,
        custom_ug_params = {
            "is_plain_mode": True,
            "infer_mode": "force_gen", # [force_und, force_gen, interleave] 若没有指定system_prompt，则使用对应模式的默认system prompt
            "cfg_mode": "naive_text", 
            "diffusion_configs": {"height": 200, "width": 200, "cfg_scale": (1.5, 5.0)},
            "training_mode": "sft"
        },
        vlm_vae_eval=True, # 评测理解任务时候是否要加入vae embedding
    ),
}


UG_API_GROUPs = [ug_apis, psm_apis]
