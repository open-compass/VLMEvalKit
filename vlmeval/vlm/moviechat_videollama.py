import torch
import warnings
import logging
import copy as cp
from PIL import Image
import pandas as pd
import string
import re
import av
from av.codec.context import CodecContext
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import os
import os.path as osp
from huggingface_hub import snapshot_download
from .base import BaseModel
from ..smp import isimg, listinstr, cn_string
from ..dataset import DATASET_TYPE, DATASET_MODALITY
from transformers import StoppingCriteria, StoppingCriteriaList
from torchvision import transforms

try:
    from .moviechat.common.registry import registry
except ImportError as e:
    logging.debug(
        f"MovieChat is not installed. First, install MovieChat by 'https://github.com/rese1f/MovieChat.git' and link `VLMEval_MovieChat` to `vlmeval/vlm/moviechat`. Change the torch version with `python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118` and `python -m pip install flash-attn==2.3.6 --no-build-isolation`"
    )

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False

class MovieChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(
        self,
        model_path: str = 'Enxin/MovieChat-vicuna',
        pretrained_llama_proj_model: str = "Enxin/MovieChat-proj",
        short_memory_length: int = 18,
        long_memory_length: int = 256,
        sliding_window_length: int = 8,
        merge_frame_length: int = 2,
        tmp_folder: str = "tmp/",
        use_cache: bool = True,
        truncate_context: bool = False,
        truncation: bool = True,
        **kwargs
    ):
        assert model_path is not None

        self.VIDEO_LLM = True
        self.model_path = model_path
        llama_model = snapshot_download(repo_id=model_path) if not osp.isdir(model_path) else model_path
        llama_proj_pth = snapshot_download(repo_id=pretrained_llama_proj_model) if not osp.isdir(pretrained_llama_proj_model) else pretrained_llama_proj_model
        llama_proj = osp.join(llama_proj_pth, "finetune-vicuna7b-v2.pth")
        model_config = {
            "arch": "moviechat",
            "model_type": "pretrain_vicuna",
            "freeze_vit": True,
            "freeze_qformer": True,
            "max_txt_len": 256,
            "end_sym": "###",
            "low_resource": False,
            "frozen_llama_proj": False,
            "llama_model": llama_model,
            "llama_proj_model": llama_proj,
            "image_size": 224,
            "num_query_token": 32,
        }

        model_cls = registry.get_model_class(model_config["arch"])
        
        self.model = model_cls.from_config(model_config).eval()
        self.model = self.model.half().cuda()
        self.model.visual_encoder = self.model.visual_encoder.half().cuda()
        self.model.ln_vision = self.model.ln_vision.cuda()
        self.model.Qformer = self.model.Qformer.half().cuda()
        self.model.llama_model = self.model.llama_model.half().cuda()
        if hasattr(self.model.visual_encoder, 'patch_embed'):
            self.model.visual_encoder.patch_embed = self.model.visual_encoder.patch_embed.cuda()
        self.device = self.model.device

        vis_processor_cfg = {
            "name": "alpro_video_eval",
            "n_frms": 8,
        }
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]  # Resize to 224x224  # Convert PIL Image to Tensor with shape [C, H, W]  # Normalize
        )
        self.processor = registry.get_processor_class(vis_processor_cfg["name"]).from_config(vis_processor_cfg)

        self.model.short_memory_length = short_memory_length
        self.model.long_memory_length = long_memory_length
        self.merge_frame_length = merge_frame_length
        self.sliding_window_length = sliding_window_length
        self.num_clips = (self.model.long_memory_length // self.merge_frame_length) * ((self.model.short_memory_length - self.merge_frame_length) // self.sliding_window_length)
        self.tmp_folder = tmp_folder

        self._tokenizer = self.model.llama_tokenizer
        stop_words_ids = [torch.tensor([835]).to(self.device), torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.model.eval()
        self.truncation = truncation
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        self.kwargs = kwargs

        torch.cuda.empty_cache()

    def get_context_emb(self, input_text, img_list):
        prompt_1 = "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your answers in details.###Human: <Video><ImageHere></Video>"
        prompt_2 = input_text
        prompt_3 = "###Assistant:"

        prompt = prompt_1 + " " + prompt_2 + prompt_3

        prompt_segs = prompt.split("<ImageHere>")
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def answer(self, img_list, input_text, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        embs = self.get_context_emb(input_text, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print("Warning: The number of tokens in current conversation exceeds the max length. " "The model will not see the contexts outside the range.")
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token  at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split("###")[0]  # remove the stop sign '###'
        output_text = output_text.split("Assistant:")[-1].strip()
        return output_text, output_token.cpu().numpy()

    def generate_inner(self, message, dataset=None):

        if DATASET_MODALITY(dataset) == 'VIDEO':
            context, visuals = self.message_to_promptvideo(message)
        elif DATASET_MODALITY(dataset) == 'IMAGE':
            logging.warning('Image modality is not supported for MovieChat')

        if visuals is not None:
            self.model.short_memory_buffer = []
            self.model.long_memory_buffer = []
            img_list = []

            os.makedirs(self.tmp_folder, exist_ok=True)

            video = VideoFileClip(visuals)
            clip_duration = video.duration / self.num_clips

            cur_frame = 0
            for i in range(self.num_clips):
                preprocess_frames = []
                start_time = i * clip_duration
                end_time = start_time + clip_duration
                frames = list(video.subclipped(start_time, end_time).iter_frames(fps=self.sliding_window_length / clip_duration))[: self.sliding_window_length]
                for frame in frames:
                    frame = Image.fromarray(frame)
                    frame_tensor = self.transform(frame) 
                    frame_tensor = frame_tensor.unsqueeze(1)  
                    frame_tensor = self.processor.transform(frame_tensor)
                    frame_tensor = frame_tensor.half().cuda()
                    preprocess_frames.append(frame_tensor[:, 0])  

                frames_tensor = torch.stack(preprocess_frames, dim=0)
                frames_tensor = frames_tensor.half().cuda()

                visual_output = self.model.visual_encoder(frames_tensor)
                visual_output = visual_output.float()
                image_embeds = self.model.ln_vision(visual_output)
                image_embeds = image_embeds.half()
                
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
                query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                encoded_window = query_output.last_hidden_state

                for frame in encoded_window:
                    if cur_frame < (self.model.short_memory_length - self.merge_frame_length):
                        if len(self.model.short_memory_buffer) == self.model.short_memory_length:
                            self.model.short_memory_buffer.pop(0)
                        self.model.short_memory_buffer.append(frame)
                    cur_frame += 1

                if cur_frame == (self.model.short_memory_length - self.merge_frame_length):
                    cur_frame = 0

                    # merge short_memory_frames
                    similar_list = []
                    for frame_i in range(len(self.model.short_memory_buffer) - 1):
                        scores = self.model.short_memory_buffer[frame_i] @ self.model.short_memory_buffer[frame_i + 1].transpose(-1, -2)
                        frame_silimar = torch.mean(scores)
                        similar_list.append(frame_silimar)

                    while len(self.model.short_memory_buffer) > self.merge_frame_length:
                        max_value = max(similar_list)
                        max_index = similar_list.index(max_value)
                        new_frame_feature = (self.model.short_memory_buffer[max_index].cpu() + self.model.short_memory_buffer[max_index + 1].cpu()) / 2
                        self.model.short_memory_buffer[max_index] = new_frame_feature.cuda()
                        del self.model.short_memory_buffer[max_index + 1]
                        similar_list = []
                        for frame_i in range(len(self.model.short_memory_buffer) - 1):
                            scores = self.model.short_memory_buffer[frame_i] @ self.model.short_memory_buffer[frame_i + 1].transpose(-1, -2)
                            frame_silimar = torch.mean(scores)
                            similar_list.append(frame_silimar)

                    for frame in self.model.short_memory_buffer:
                        self.model.long_memory_buffer.append(frame)

            cur_image = self.model.encode_image(
                preprocess_frames[-1].unsqueeze(0).unsqueeze(2).cuda()
            )
            try:
                video_emb, _ = self.model.encode_long_video(cur_image.half(), device=self.device, middle_video=False)
                img_list.append(video_emb)
                answer = self.answer(img_list=img_list, input_text=context, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
            except Exception as e:
                return None
        return answer
