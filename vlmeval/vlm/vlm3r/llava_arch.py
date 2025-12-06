#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .vision_encoder.builder import build_vision_tower
from .spatial_encoder.builder import build_spatial_tower
from .fusion_block.builder import build_multimodal_fusion_block
from .resampler.builder import build_vision_resampler
from .projector.builder import build_vision_projector
from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from .mm_utils import get_anyres_image_grid_shape
from .utils import rank0_print, rank_print
import random
from einops import rearrange

import torch.nn.functional as F
import numpy as np
import cv2  # OpenCV for resizing and writing images
import matplotlib.cm as cm # For colormaps
import os
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)

            # create spatial tower and fusion block
            if hasattr(config, "spatial_tower"):
                self.spatial_tower = build_spatial_tower(config, delay_load=True)
            if hasattr(config, "fusion_block"):
                self.fusion_block = build_multimodal_fusion_block(config, delay_load=delay_load)

            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_spatial_tower(self):
        spatial_tower = getattr(self, "spatial_tower", None)
        if type(spatial_tower) is list:
            spatial_tower = spatial_tower[0]
        return spatial_tower

    def get_fusion_block(self):
        fusion_block = getattr(self, "fusion_block", None)
        if type(fusion_block) is list:
            fusion_block = fusion_block[0]
        return fusion_block

    def initialize_spatial_tower(self, model_args, fsdp=None):
        spatial_tower = model_args.spatial_tower
        self.config.mm_spatial_tower = spatial_tower

        if self.get_spatial_tower() is None:
            spatial_tower = build_spatial_tower(model_args)
            for k, v in spatial_tower.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.spatial_tower = [spatial_tower]
            else:
                self.spatial_tower = spatial_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                spatial_tower = self.spatial_tower[0]
            else:
                spatial_tower = self.spatial_tower
            spatial_tower.load_model()

    def initialize_fusion_block(self, model_args, fsdp=None):
        # initialize the fusion block
        if getattr(self, "fusion_block", None) is None:
            self.fusion_block = build_multimodal_fusion_block(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.fusion_block.parameters():
                p.requires_grad = True

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.fusion_block.load_state_dict(get_w(mm_projector_weights, "fusion_block"), strict=False)
            rank0_print(f"Loaded fusion block weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type


        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_spatial_tower(self):
        return self.get_model().get_spatial_tower()

    def get_fusion_block(self):
        return self.get_model().get_fusion_block()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    # def encode_images(self, images):
    #     # vision features
    #     image_features = self.get_model().get_vision_tower()(images)
    #     # set brance
    #     if self.get_model().get_spatial_tower() is not None and self.get_model().get_fusion_block() is not None:
    #         # spatial features
    #         spatial_encoder_type = self.get_model().config.spatial_tower
    #         if spatial_encoder_type == "cut3r":
    #             # Scale up image by 16/14 before passing to spatial tower
    #             images_scaled = nn.functional.interpolate(images, size=(432, 432), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         elif spatial_encoder_type == "vggt":
    #             images_scaled = nn.functional.interpolate(images, size=(378, 378), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         elif spatial_encoder_type == "cut3r_points":
    #             images_scaled = nn.functional.interpolate(images, size=(432, 432), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         else:
    #             raise ValueError(f"Unexpected spatial encoder type: {spatial_encoder_type}")

    #         fusion_block_type = self.get_model().config.fusion_block

    #         # Handle special case for mlp2x_gelu_cat first
    #         if fusion_block_type == "mlp2x_gelu_cat":
    #             image_features = torch.cat((image_features, image_spatial_features), dim=-1)
    #             image_features = self.get_model().get_fusion_block()(image_features)
    #         # Handle special case for mlp2x_gelu
    #         elif fusion_block_type == "mlp2x_gelu":
    #             image_features = self.get_model().get_fusion_block()(image_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         # Handle all other fusion types that follow the same pattern
    #         elif fusion_block_type in ["cross_attention", "mlp", "transformer"]:
    #             image_features = self.get_model().get_fusion_block()(image_features, image_spatial_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         elif fusion_block_type == "llava_3d_fusion_block":
    #             image_features = self.get_model().get_fusion_block()(image_features, image_spatial_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         else:
    #             raise ValueError(f"Unexpected fusion block type: {fusion_block_type}")
    #     else:
    #         # project features
    #         image_features = self.get_model().mm_projector(image_features)

    #     return image_features

    def encode_images(self, images, spatial_features=None, point_maps=None):
        # vision features
        image_features = self.get_model().get_vision_tower()(images)
        # fuse with spatial features
        if self.get_model().get_spatial_tower() is not None and self.get_model().get_fusion_block() is not None:
            spatial_encoder_type = self.get_model().config.spatial_tower
            fusion_block_type = self.get_model().config.fusion_block

            if spatial_encoder_type.endswith("points"):
                points = self.get_model().get_spatial_tower()(images)
                image_features = self.get_model().get_fusion_block()(image_features, points)
                image_features = self.get_model().mm_projector(image_features)

            else:
                if spatial_features is not None and 'cut3r' in spatial_encoder_type:
                    camera_tokens, patch_tokens = spatial_features[0]["camera_tokens"], spatial_features[0]["patch_tokens"]
                else:
                    camera_tokens, patch_tokens = self.get_model().get_spatial_tower()(images)

                if fusion_block_type == 'cross_attention':
                    # fuse with spatial features
                    spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
                    spatial_tower_select_feature_list = spatial_tower_select_feature.split(",")
                    final_image_features = []
                    for spatial_tower_select_feature in spatial_tower_select_feature_list:
                        if spatial_tower_select_feature == "camera_tokens":
                            final_image_features.append(camera_tokens)
                        elif spatial_tower_select_feature == "patch_tokens":
                            final_image_features.append(patch_tokens)
                        elif spatial_tower_select_feature == "all":
                            final_image_features = [camera_tokens, patch_tokens]
                        else:
                            raise ValueError(f"Unexpected spatial_tower_select_feature: {spatial_tower_select_feature}")
                    final_image_features = torch.cat(final_image_features, dim=1).to(self.dtype)
                    image_features, attn_weights = self.get_model().get_fusion_block()(image_features, final_image_features)
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'cross_attention_with_mlp':
                    image_features, attn_weights = self.get_model().get_fusion_block()(image_features, patch_tokens)
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'transformer':
                    spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
                    if spatial_tower_select_feature == "all":
                        final_image_features = torch.cat((camera_tokens, patch_tokens), dim=1).to(self.dtype)
                        image_features = self.get_model().get_fusion_block()(image_features, final_image_features)
                        image_features = self.get_model().mm_projector(image_features)

                elif (fusion_block_type == 'mlp_after_clip_proj'
                      or fusion_block_type == 'concat_mlp'
                      or fusion_block_type == 'concat_self_attention'):

                    image_features = self.get_model().mm_projector(image_features)
                    image_features = self.get_model().get_fusion_block()(image_features, patch_tokens)

        elif self.get_model().get_spatial_tower() is None and self.get_model().get_fusion_block() is not None:
            assert point_maps is not None
            image_features = self.get_model().mm_projector(image_features)
            image_features = self.get_model().get_fusion_block()(image_features, point_maps[0]) # FIXME: point_maps is a list of tensors, each tensor is a point map for one image

        else:
            image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):

            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, spatial_features=None, point_maps=None, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images, spatial_features, point_maps)
            # if self.get_model().get_spatial_tower() is not None:
            #     if spatial_features is None:
            #         camera_tokens, patch_tokens = self.encode_spatial_features(concat_images)
            #     else:
            #         camera_tokens, patch_tokens = spatial_features[0]["camera_tokens"], spatial_features[0]["patch_tokens"]
            #     # fuse with spatial features
            #     spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
            #     spatial_tower_select_feature_list = spatial_tower_select_feature.split(",")
            #     final_image_features = []
            #     for spatial_tower_select_feature in spatial_tower_select_feature_list:
            #         if spatial_tower_select_feature == "camera_tokens":
            #             final_image_features.append(camera_tokens)
            #         elif spatial_tower_select_feature == "patch_tokens":
            #             final_image_features.append(patch_tokens)
            #     final_image_features = torch.cat(final_image_features, dim=1)
            #     encoded_image_features = self.get_model().get_fusion_block()(encoded_image_features, final_image_features)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)

            # if self.get_model().get_spatial_tower() is not None:
            #     if spatial_features is not None:
            #         encoded_camera_tokens = spatial_features[0]["camera_tokens"] ## FIXME: spatial_features is a list of dicts, each dict contains camera_tokens and patch_tokens
            #         encoded_patch_tokens = spatial_features[0]["patch_tokens"]
            #         # fusion block
            #         encoded_camera_tokens, encoded_patch_tokens = self.get_model().get_fusion_block()(encoded_camera_tokens, encoded_patch_tokens)
            #     else:
            #         encoded_camera_tokens, encoded_patch_tokens = self.encode_spatial_features(concat_images)
            #     camera_tokens = torch.split(encoded_camera_tokens, split_sizes)
            #     encoded_patch_tokens = torch.split(encoded_patch_tokens, split_sizes)
            #     # split and merge
            #     patch_tokens = []
            #     # pool patch tokens
            #     for idx, patch_token in enumerate(encoded_patch_tokens):
            #         if idx in video_idx_in_batch:
            #             patch_tokens.append(self.get_2dPool(patch_token))
            #         else:
            #             patch_tokens.append(patch_token)

            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")

                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))

                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        # For single images, apply the same grid-wise newline logic
                        # as used for video frames to maintain consistency.
                        image_feature = self.add_token_per_grid(image_feature)
                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                # Handle cases with no image tokens if necessary.
                # Original code appends empty features, adapt if needed.
                cur_image_features = image_features[cur_image_idx]
                # Also get corresponding spatial features
                # if self.get_model().get_spatial_tower() is not None:
                #     cur_camera_tokens = camera_tokens[cur_image_idx]
                #     cur_patch_tokens = patch_tokens[cur_image_idx]

                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)

                # Concatenate text embeds, visual embeds [0:0], and spatial embeds [0:0]?
                # This part of original code seems odd (using [0:0]), clarify its purpose.
                # Assuming you want to append actual features if available, otherwise skip.
                embeds_to_concat = [cur_input_embeds_1]
                # if cur_image_features is not None and cur_image_features.numel() > 0:
                #     embeds_to_concat.append(cur_image_features[0:0]) # Original behavior

                cur_input_embeds = torch.cat(embeds_to_concat, dim=0)

                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1 # Increment even if no image token? Check original logic intent.
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                # Append text embeddings and labels
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                # If this segment was followed by an image token, insert features
                if i < num_images:
                    try:
                        # Get the visual/fused features for the current image
                        cur_image_features = image_features[cur_image_idx]
                        # Get the spatial features for the current image
                        # if self.get_model().get_spatial_tower() is not None:
                        #     cur_camera_tokens = camera_tokens[cur_image_idx]
                        #     cur_patch_tokens = patch_tokens[cur_image_idx]
                    except IndexError:
                         # Fallback logic from original code
                        cur_image_features = image_features[cur_image_idx - 1]
                        # if self.get_model().get_spatial_tower() is not None:
                        #     cur_camera_tokens = camera_tokens[cur_image_idx - 1]
                        #     cur_patch_tokens = patch_tokens[cur_image_idx - 1]

                    cur_image_idx += 1

                    # Prepare combined features (visual + spatial)
                    features_to_insert = []
                    if cur_image_features is not None and cur_image_features.shape[0] > 0:
                        features_to_insert.append(cur_image_features)
                    # spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", None)
                    # if self.get_model().get_spatial_tower() is not None and spatial_tower_select_feature is not None:
                    #     spatial_feature_flags = spatial_tower_select_feature.split(",")

                    #     if cur_camera_tokens is not None and cur_camera_tokens.shape[0] > 0 and "camera_tokens" in spatial_feature_flags:
                    #         features_to_insert.append(cur_camera_tokens.flatten(0, 1))
                    #     if cur_patch_tokens is not None and cur_patch_tokens.shape[0] > 0 and "patch_tokens" in spatial_feature_flags:
                    #         features_to_insert.append(cur_patch_tokens.flatten(0, 1))

                    if features_to_insert:
                        combined_features = torch.cat(features_to_insert, dim=0)
                        cur_new_input_embeds.append(combined_features)
                        # Add IGNORE_INDEX labels for the entire combined feature length
                        cur_new_labels.append(torch.full((combined_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
