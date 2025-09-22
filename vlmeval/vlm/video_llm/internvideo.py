import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import transformers
from transformers import AutoModel, AutoTokenizer

import warnings
import copy as cp

import sys
import os
import logging
from ..base import BaseModel
from ...smp import isimg, listinstr, version_cmp
from ...dataset import DATASET_TYPE
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def get_indices_by_fps(bound, src_fps, max_frame, target_fps, first_idx=0):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = 0.0, max_frame / src_fps
    if target_fps is None or target_fps <= 0:
        raise ValueError("target_fps must be a positive number")
    times = np.arange(start, end, 1.0 / target_fps)
    frame_indices = np.round(times * src_fps).astype(int)
    frame_indices = np.clip(frame_indices, first_idx, max_frame)
    frame_indices = np.unique(frame_indices)
    return frame_indices


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
    local_num_frames = 4        
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments
    
    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)

    return num_frames

def load_video(
    video_path, bound=None, input_size=448, max_num=1, num_segments=32, target_fps=-1, get_frame_by_duration = False
):  
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    if target_fps is not None and target_fps > 0:
        frame_indices = get_indices_by_fps(bound, fps, max_frame, target_fps, first_idx=0)
    else:
        if get_frame_by_duration:
            duration = max_frame / fps
            num_segments = get_num_frames_by_duration(duration)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def load_image(image_path, input_size=448, max_num=6):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, pixel_values.shape[0]


class InternVideo(BaseModel):
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path='OpenGVLab/InternVideo2_5_Chat_8B', **kwargs):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)
        self.generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1,
            num_beams=1
        )
        self.generation_config.update(kwargs)

        self.nframe = 128
        self.fps = 1

        # recommend using transformers==4.40.0
        # full env recommendation in https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B
        self.transformers_version = '4.40.1'
        assert version_cmp(transformers.__version__, self.transformers_version, 'eq')


    def generate_inner(self, message, dataset=None):
        text_content, videos, images = "", [], []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            elif msg["type"] == "image":
                # images.append(msg["value"])
                continue
            else:
                videos.append(msg["value"])

        if len(videos) > 1:
            raise ValueError(
                "InternVideo does not support multiple videos as input."
            )

        with torch.no_grad():
            video_pixel_values, video_num_patches_list = load_video(videos[0], num_segments=self.nframe, target_fps=self.fps, max_num=1, get_frame_by_duration=False)
            video_pixel_values = video_pixel_values.to(torch.bfloat16).to(self.model.device)
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(video_num_patches_list))])

            img_pixel_values_list, img_num_patches_list = [], []
            img_prefix = ""
            for img_pth in images:
                img_pixel_values, num_patches = load_image(img_pth, max_num=1)
                img_pixel_values_list.append(img_pixel_values)
                img_num_patches_list.append(num_patches)
                img_prefix += f"<image>\n"

            if len(img_pixel_values_list):
                img_pixel_values = torch.cat(img_pixel_values_list, dim=0).to(torch.bfloat16).to(self.model.device)
                pixel_values = torch.cat((img_pixel_values, video_pixel_values), dim=0)
            else:
                pixel_values = video_pixel_values

            # 顺序：图片在前、视频在后（与 question 和 pixel_values 一致）
            num_patches_list_all = img_num_patches_list + video_num_patches_list

            # 对于pixel_values，每张图片重复四遍，使得后面可以merge
            pixel_values = pixel_values.repeat_interleave(4, dim=0)
            # num_patches_list_all = [x for x in num_patches_list_all for _ in range(4)]
            num_patches_list_all = [x * 4 for x in num_patches_list_all]
            question = img_prefix + video_prefix + text_content
            # assert pixel_values.shape[0] % 4 == 0, "pixel_values.shape[0] must be divisible by 4"
            output, chat_history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, num_patches_list=num_patches_list_all, history=None, return_history=True)

        return output
