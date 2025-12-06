from __future__ import annotations

import math
import os

import decord
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

from ..base import BaseModel
from .constants import (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN,
                        DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,
                        IMAGE_TOKEN_INDEX)
from .conversation import SeparatorStyle, conv_templates

from .llava_qwen import LlavaQwenForCausalLM
from .mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                       tokenizer_image_token)


class VLM3R(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path,
        attn_implementation='flash_attention_2',
        device_map="auto",
        torch_dtype="float16",
        use_cache=True,
        delay_load=False,
        tie_weights=True,
        model_name=None,
        model_base='lmms-lab/LLaVA-NeXT-Video-7B-Qwen2',
        conv_template='qwen_1_5',
        # video params
        mm_resampler_type='spatial_pool',
        mm_spatial_pool_stride=2,
        mm_spatial_pool_out_channels=1024,
        mm_spatial_pool_mode="bilinear",
        mm_newline_position="grid",
        mm_pooling_position="after",
        video_max_frames=32,
        video_fps=1,
        video_force_sample=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained = model_path
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = get_model_name_from_path(model_path)
        
        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)

        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.max_frames_num = int(video_max_frames)
        self.mm_resampler_location = mm_pooling_position
        self.mm_newline_position = mm_newline_position
        self.delay_load = delay_load
        
        # Currently only overwrite=True is supported.
        # if self.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = self.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = self.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        overwrite_config["mm_pooling_position"] = self.mm_resampler_location
        overwrite_config["mm_newline_position"] = self.mm_newline_position
        overwrite_config["add_faster_video"] = False
        overwrite_config["delay_load"] = self.delay_load

        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        if cfg_pretrained.architectures[0] == "LlavaLlamaForCausalLM":  # Ugly code, only used in  vicuna that needs ROPE
            if "224" in cfg_pretrained.mm_vision_tower:
                least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
            else:
                least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

            scaling_factor = math.ceil(least_token_number / 4096)
            if scaling_factor >= 2:
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
        
        # load model
        kwargs["device_map"] = device_map
        if torch_dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        if "multimodal" in kwargs:
            if kwargs["multimodal"] is True:
                is_multimodal = True
                kwargs.pop("multimodal")
        else:
            is_multimodal = False
            
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            
        del kwargs["device_map"]
        additional_config = {
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 152064
        }
        from .language_model.llava_qwen import LlavaQwenConfig
        lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)

        if overwrite_config is not None:
            overwrite_config.update(additional_config)
            for k, v in overwrite_config.items():
                setattr(lora_cfg_pretrained, k, v)
            self.model = LlavaQwenForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=False,
                attn_implementation=attn_implementation,
                config=lora_cfg_pretrained, **kwargs)
        else:
            overwrite_config = additional_config
            for k, v in overwrite_config.items():
                setattr(lora_cfg_pretrained, k, v)
            self.model = LlavaQwenForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=False,
                attn_implementation=attn_implementation,
                config=lora_cfg_pretrained, **kwargs)
        # self.model.to(device="cuda", dtype=torch.float16)
        token_num, token_dim = self.model.lm_head.out_features, self.model.lm_head.in_features


        if self.model.lm_head.weight.shape[0] != token_num:
            self.model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=self.model.device, dtype=self.model.dtype))
            self.model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=self.model.device, dtype=self.model.dtype))


        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download

            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                return torch.load(cache_file, map_location="cpu")

            non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
        non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        self.model.load_state_dict(non_lora_trainables, strict=False)



        self.model = PeftModel.from_pretrained(self.model, model_path)
        print("Merging LoRA weights...")
        self.model = self.model.merge_and_unload()

        if "llava" in model_base.lower() or is_multimodal:
            mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

            new_vocab_size = len(self.tokenizer)
            self.model.resize_token_embeddings(new_vocab_size)

            self.vision_tower = self.model.get_vision_tower()
            if not self.vision_tower.is_loaded:
                self.vision_tower.load_model(device_map=device_map)
            if device_map != "auto":
                self.vision_tower.to(device="cuda", dtype=torch.float16)
            self.image_processor = self.vision_tower.image_processor
        
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.conv_template = conv_template
        self.use_cache = use_cache
        
        self.model.config.video_max_frames = video_max_frames
        self.model.config.video_fps = video_fps
        self.model.config.video_force_sample = video_force_sample
        self.nframe = video_max_frames
        self.fps = video_fps
        self.model.to(device="cuda")


    def generate_inner(self, message, dataset=None):
        videos = []
        images = []
        question = ''
        image_sizes = []
        for s in message:
            if s['type'] == 'image':
                image_obj = Image.open(s['value']).convert("RGB")
                image = self.image_processor.preprocess(image_obj, return_tensors="pt")["pixel_values"].half().cuda()
                image_sizes.append(image_obj.size)  # (width, height)
                images.append(image[0])
            elif s['type'] == 'text':
                question += s['value']
            elif s['type'] == 'video':
                video_path = s['value']
                video_frames, _, _ = self.get_video_frames(video_path)
                video = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
                videos.append(video)
        
        if len(images) > 0:
            image_tensor = images
        elif len(videos) > 0:
            image_tensor = videos
        else:
            assert False, "No images or videos found in the message."
            
        if self.model.config.mm_use_im_start_end:
            image_placeholders = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" ) * len(image_tensor)
        else:
            image_placeholders =  DEFAULT_IMAGE_TOKEN * len(image_tensor) + "\n"

        question = image_placeholders + question
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        
        if self.tokenizer.pad_token_id is None and "qwen" in self.tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            self.tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,  # Pass the list of image tensors
                attention_mask=attention_masks,
                modalities= ["video" for _ in videos] if len(videos) > 0 else ["image" for _ in images],
                image_sizes = image_sizes if len(images) > 0 else None,
                do_sample= False, # True if gen_kwargs["temperature"] > 0 else False,
                temperature=0.0,
                max_new_tokens=16,
                top_p=1.0,
                num_beams=1,
                use_cache=self.use_cache,
                stopping_criteria=[stopping_criteria]
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


    def get_video_frames(self, vid_path):

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        # Align with offical video frame sampling strategy
        indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()

        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]

        return images, indices, video_info
    
    
