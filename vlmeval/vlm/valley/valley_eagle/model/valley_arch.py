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
import torch
from torch import nn
import numpy as np
from ..util.config import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_VI_END_TOKEN,
    DEFAULT_VI_START_TOKEN,
    GANDALF_TOKEN_INDEX,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    COR_START_TOKEN,
    COR_END_TOKEN
)

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .token_compressor.builder import build_token_compressor
from ..util.mm_utils import get_anyres_image_grid_shape, unpad_image


class ValleyMetaModel:
    def __init__(self, config):
        super(ValleyMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower, self.qwen2vl_vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

        if hasattr(config, "token_compressor_config") and config.token_compressor_config is not None:
            self.token_compressor = build_token_compressor(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        qwen2vl_vision_tower = getattr(self, "qwen2vl_vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower, qwen2vl_vision_tower

    def get_token_compressor(self):
        token_compressor = getattr(self, "token_compressor", None)
        return token_compressor

    def initialize_token_compressor(self, model_args, logger):
        self.config.token_compressor_config = model_args.token_compressor_config
        if getattr(self, "token_compressor", None) is None and model_args.token_compressor_config is not None:
            logger.warning("initializing token compressor weights...")
            self.token_compressor = build_token_compressor(self.config)

    def initialize_vision_modules(self, model_args, logger):
        """ Initialize thevision modules and save the model config args
        when first train multimodal model. The function should after model init
        in train script.

        Args:
            model_args (_type_): model arguments from train config.

        """

        self.config.mm_vision_tower = model_args.vision_tower  # model_args.vision_tower is string
        self.config.eagle_vision_tower = model_args.eagle_vision_tower

        self.vision_tower, self.qwen2vl_vision_tower = build_vision_tower(model_args)
        self.vision_tower.load_model()  # vision_tower is an instance
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.pool_out_size = model_args.pool_out_size
        self.config.mm_hidden_size = self.vision_tower.hidden_size
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
        self.config.pixelshuffle_downsample_ratio = model_args.pixelshuffle_downsample_ratio
        self.config.mlp_hidden_dim = model_args.mlp_hidden_dim
        self.config.tokenize_function = model_args.tokenize_function

        # valley-video projector has no mm_projector_type attribute

        if (getattr(self, "mm_projector", None) is None) or (
            getattr(self.mm_projector, "mm_projector_type", None) != self.config.mm_projector_type
        ):
            logger.warning("initializing projector weights...")
            self.mm_projector = build_vision_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
            weight_keys = list(mm_projector_weights.keys())

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            try:
                logger.warning('Loading projector weight, and projector weight keys have prefix "mm_projector". ')
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            except:
                assert "mm_projector" not in weight_keys[0]
                self.mm_projector.load_state_dict(mm_projector_weights)


class ValleyMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_token_compressor(self):
        return self.get_model().get_token_compressor()

    def split_by_instance(self, original_list, split_sizes):
        start = 0
        sub_lists = []
        for size in split_sizes:
            end = start + size
            sub_list = original_list[start:end]
            # sub_lists.append(torch.stack(sub_list, dim=0))
            sub_lists.append([x.to(self.device) for x in sub_list])
            start = end
        return sub_lists

    def encode_images(self, images=None, split_sizes=None, pixel_values=None, grid_thw=None):
        """
        images: (if not anyres) images.shape = [n,3,336,336] , n = number of images + (number of video) * 8
        images: (if anyres) images.shape = [n,3,336,336] , n = number of tiles * number of images
        """
        siglip_vision_tower, qwen2vl_vision_tower = self.get_model().get_vision_tower()

        if images is not None:
            image_features = siglip_vision_tower(images)
            image_features = self.get_model().mm_projector(image_features)

        qwen2vl_image_features = None
        if pixel_values is not None:
            qwen2vl_image_features = qwen2vl_vision_tower(pixel_values, grid_thw)
            qwen2vl_image_split_sizes = torch.prod(grid_thw[:, 1:3] // 2, dim=1)
            qwen2vl_image_features = torch.split(qwen2vl_image_features, qwen2vl_image_split_sizes.tolist(), dim=0)
            qwen2vl_image_features = self.split_by_instance(qwen2vl_image_features, split_sizes)

            if images is None:
                return qwen2vl_image_features

        if getattr(self.config,'anyres', False) and getattr(self.config, 'max_vision_token', None) is not None:
            assert split_sizes is not None
            image_features = list(torch.split(image_features, split_sizes, dim=0))
            for i,image_feature in enumerate(image_features):
                hidden_dim = image_feature.shape[-1]
                image_tokens = image_feature.shape[0] * image_feature.shape[1]
                # the max_vision_token will be processed in the unpad image token part
                if False:
                    if image_tokens > self.config.max_vision_token:
                        intput_shape = int((image_feature.shape[1])**0.5)
                        output_shape = int((self.config.max_vision_token / image_feature.shape[0])**0.5)
                        image_feature = image_feature.view(image_feature.shape[0],intput_shape, intput_shape, -1) \
                                                     .permute(0,3,1,2)
                        # different from roi pooling, but in square image, it seems the same
                        m = nn.AdaptiveAvgPool2d(output_shape)
                        pooling_feature = m(image_feature).permute(0,2,3,1)
                        image_features[i] = pooling_feature.view(image_feature.shape[0], -1, hidden_dim)
                split_sizes = None  # have already split, set the flag
        if getattr(self.config, 'model_class', None) in ['valley-video','valley_video']:
            # since we mix video data and image data in a batch, and in valley video structure,
            # both have same dimention, we need to split them to process
            if split_sizes is not None:
                image_features = torch.split(image_features, split_sizes, dim=0)
            if getattr(self.config, 'mm_use_im_start_end', False):
                video_start_end_image_features = []
                for feature in image_features:
                    temporal_features = feature[:,0,:]
                    video_features = torch.mean(feature[:,1:,:],dim=0)
                    special_token_ids = torch.tensor(
                        self.tokenizer.convert_tokens_to_ids(
                            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN]
                        )
                    ).to(video_features.device)
                    special_token_feature = self.get_model().embed_tokens(special_token_ids)
                    # add special sep feature as [<im_start><video_feature><im_end><vi_start><temporal_feature><vi_end>]
                    new_image_feature = torch.cat([
                        special_token_feature[0].unsqueeze(0),
                        video_features,
                        special_token_feature[1].unsqueeze(0),
                        special_token_feature[2].unsqueeze(0),
                        temporal_features,
                        special_token_feature[2].unsqueeze(0)
                    ])
                    video_start_end_image_features.append(new_image_feature.unsqueeze(0))
                return video_start_end_image_features, qwen2vl_image_features
            else:
                image_features_new = []
                for feature in image_features:
                    temporal_features = feature[:,0,:]
                    video_features = torch.mean(feature[:,1:,:],dim=0)
                    new_image_feature = torch.cat([video_features, temporal_features])
                    image_features_new.append(new_image_feature.unsqueeze(0))  # increase batch dim
                return image_features_new, qwen2vl_image_features
        elif getattr(self.config, 'model_class', None) in ['valley-product','valley_product', 'tinyvalley']:
            if getattr(self.config, 'mm_use_im_start_end', False):
                raise ValueError('mm_use_im_start is not support in valley_product')
            if split_sizes is not None:
                image_features = torch.split(image_features, split_sizes, dim=0)
            return image_features, qwen2vl_image_features
        elif getattr(self.config, 'model_class', None) == 'valley-product-gandalf':
            raise ValueError('valley-product-gandalf is not support in this version.')
        else:
            raise ValueError('No model class specified')

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images,
            image_sizes, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, pack_ids):

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and \
                    images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if not getattr(self.config,'anyres', False):
                concat_images = torch.cat([image for image in images], dim=0)  # to do batch compute
                split_sizes = [image.shape[0] for image in images]
                if pixel_values is not None:
                    image_features, qwen2vl_image_features = self.encode_images(
                        concat_images,
                        split_sizes,
                        pixel_values,
                        image_grid_thw
                    )
                    image_features = [x.to(self.device) for x in image_features]
                elif pixel_values_videos is not None:
                    image_features, qwen2vl_image_features = self.encode_images(
                        concat_images,
                        split_sizes,
                        pixel_values_videos,
                        video_grid_thw
                    )
                    image_features = [x.to(self.device) for x in image_features]
                else:
                    image_features, _ = self.encode_images(concat_images, split_sizes)

                # image_features = [x.flatten(0, 1).to(self.device) for x in image_features]

                # token compress
                if self.get_token_compressor() is not None:
                    image_features = [self.get_token_compressor()(x) for x in image_features]
            else:
                # if do anyres, each image become some sub_images, so need to add a
                # images = [
                #           [image1_tiles(n1,3,336,336), image2_tiles(n2,3,336,336), ...],
                #           [image1_tiles(n1,3,336,336), image2_tiles(n2,3,336,336), ...], ...
                #          ]
                split_sizes = [len(image) for image in images]
                # get qwen2vl features
                qwen2vl_image_features = self.encode_images(None, split_sizes, pixel_values, image_grid_thw)

                image_features = []
                for batch_images in images:
                    concat_images = torch.cat([image for image in batch_images], dim=0)  # to do batch compute
                    split_sizes = [image.shape[0] for image in batch_images]
                    batch_image_features, _ = self.encode_images(
                        concat_images,
                        split_sizes,
                        pixel_values,
                        image_grid_thw
                    )
                    # token compress
                    if self.get_token_compressor() is not None:
                        # x is tensor(n_tiles, T, d) or [tensor(T1, d), tensor(T2, d), ...]
                        batch_image_features = [self.get_token_compressor()(x) for x in batch_image_features]

                    if type(batch_image_features[0]) is list:
                        batch_image_features = [torch.cat(x).to(self.device) for x in batch_image_features]
                    else:
                        # tiles feature need to flatten in token dimention, [n_tiles, T, d] -> [n_tiles * T, d]
                        batch_image_features = [x.view(-1,x.shape[-1]).to(self.device) for x in batch_image_features]

                    image_features.append(batch_image_features)

                # unpad image tokens
                height = width = self.config.num_patches_per_side
                new_image_features = []
                for batch_image_features, batch_image_sizes in zip(image_features, image_sizes):
                    batch_image_features_list = []
                    for cur_image_feature, cur_image_size in zip(batch_image_features, batch_image_sizes):
                        base_image_feature = cur_image_feature[:width * height, :]
                        image_feature = cur_image_feature[width * height:, :]
                        if image_feature.shape[0] != 0:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                cur_image_size,
                                self.config.grid_pinpoints,
                                self.config.vit_crop_size
                            )
                            # (num_patch_H, num_patch_W, H, W, C)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            # (C, num_patch_H, H, num_patch_W, W)
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            # (C, num_token_H, num_token_W)
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            # (C, num_token_H_unpad, num_token_W_unpad)
                            image_feature = unpad_image(image_feature, cur_image_size)
                            input_shape = (image_feature.shape[-2], image_feature.shape[-1])
                            subimage_tokens = np.prod(input_shape)
                            # adaptive avg 2d pool for reducing token num
                            max_subimage_tokens = self.config.max_vision_token - width * height
                            if subimage_tokens > max_subimage_tokens:
                                aspect_ratio = input_shape[0] / input_shape[1]
                                output_shape = (
                                    int((max_subimage_tokens / aspect_ratio) ** 0.5 * aspect_ratio),
                                    int((max_subimage_tokens / aspect_ratio) ** 0.5)
                                )
                                m = nn.AdaptiveAvgPool2d(output_shape)
                                image_feature = m(image_feature)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = cur_image_feature
                        batch_image_features_list.append(image_feature)
                    new_image_features.append(batch_image_features_list)

                image_features = new_image_features

        else:
            image_features = self.encode_images(images).to(self.device)
            # token compress
            if self.get_token_compressor() is not None:
                image_features = self.get_token_compressor()(image_features)

        # TODO: image start / end is not implemented here to support pretraining.
        # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        #     raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif getattr(self, "use_pack", False) is False:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask.bool())
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask.bool())
        ]
        attention_mask = [cur_attention_mask[cur_attention_mask.bool()] for cur_attention_mask in attention_mask]

        new_input_embeds = []
        new_labels = []
        new_attention_mask = []

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_batch_image_idx = 0
            # for iamge
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if getattr(self.config, 'model_class', None) in ['valley-video','valley_video']:
                assert num_images <= 1, 'valley video is not support for multi image input'

            if num_images == 0:
                # if this piece of data is pure text,
                # then concat a dummy image to ensure the whole compute graph is same on all device
                # cur_image_features = image_features[batch_idx][cur_batch_image_idx]
                siglip_feat = image_features[batch_idx][cur_batch_image_idx]
                try:
                    qwen2vl_feat = qwen2vl_image_features[batch_idx][cur_batch_image_idx]
                    cur_image_features = torch.cat((siglip_feat, qwen2vl_feat), dim=0)
                except:
                    print("only siglip feature:", siglip_feat.shape)
                    cur_image_features = siglip_feat
                # print("num_images = 0: ", siglip_feat.shape, qwen2vl_feat.shape, cur_image_features.shape)

                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                if getattr(self.config, "use_special_start_end_token", False) \
                        and getattr(self.config, "training_stage", None) == 'stage1':
                    cur_input_embeds_1 = cur_input_embeds_1.detach()
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features.squeeze(0)[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_attention_mask.append(attention_mask[batch_idx])
                cur_batch_image_idx += 1
                continue

            image_token_indices = \
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []  # this list is to keep text input_ids
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_attention_mask = attention_mask[batch_idx]
            cur_img_attention_mask = [
                attention_mask[batch_idx][i].item()
                for i in torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            ]
            cur_attention_mask_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]])
                cur_attention_mask_noim.append(
                    cur_attention_mask[image_token_indices[i] + 1: image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = list(torch.split(cur_input_embeds, split_sizes, dim=0))  # get text features
            if getattr(self.config, "use_special_start_end_token", False) and \
                    getattr(self.config, "training_stage", None) == 'stage1':
                # for all sequence without image token,
                # the first sequence's last token(<im_start> or <vi_start>) need to update embeds weight,
                # the last sequences's first token(<im_end> or <vi_end>) need to update embeds weight,
                # other sequence's first and last token need to update weight.
                cur_input_embeds_no_im[0] = torch.cat(
                    [cur_input_embeds_no_im[0][:-1,:].detach(),cur_input_embeds_no_im[0][-1,:].unsqueeze(0)],
                    dim=0
                )
                cur_input_embeds_no_im[-1] = torch.cat(
                    [cur_input_embeds_no_im[-1][0,:].unsqueeze(0), cur_input_embeds_no_im[-1][1:,:].detach()],
                    dim=0
                )
                for i in range(1,len(cur_input_embeds_no_im) - 1):
                    # in this branch <image> token should not be placed in succession
                    cur_input_embeds_no_im[i] = torch.cat(
                        [
                            cur_input_embeds_no_im[i][0,:].unsqueeze(0),  # for im_end token
                            cur_input_embeds_no_im[i][1:-1,:].detach(),  # for text token
                            cur_input_embeds_no_im[i][-1,:].unsqueeze(0)  # for im_start token
                        ], dim=0
                    )
            elif getattr(self.config, "training_stage", None) == 'special-token-sft':
                for i in range(len(cur_input_embeds_no_im)):
                    special_token_idx = torch.where(cur_input_ids_noim[i] > self.config.eos_token_id)[0].tolist()
                    cur_input_embeds_no_im[i] = torch.cat([
                        cur_input_embeds_no_im[i][j,:].unsqueeze(0) if j in special_token_idx
                        else cur_input_embeds_no_im[i][j,:].detach().unsqueeze(0)
                        for j in range(len(cur_input_embeds_no_im[i]))
                    ], dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_attention_mask = []
            for i in range(num_images + 1):  # to add multimodal feature internal the text feature
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_attention_mask.append(cur_attention_mask_noim[i])
                if i < num_images:
                    # print(num_images, f"({len(image_features)}, {len(image_features[batch_idx])})", \
                    # f"({len(qwen2vl_image_features)}, {len(qwen2vl_image_features[batch_idx])})", \
                    # f"({batch_idx}, {cur_batch_image_idx})")
                    siglip_feat = image_features[batch_idx][cur_batch_image_idx]
                    try:
                        qwen2vl_feat = qwen2vl_image_features[batch_idx][cur_batch_image_idx]
                        cur_image_features = torch.cat((siglip_feat, qwen2vl_feat), dim=0)
                        # print(siglip_feat.shape, qwen2vl_feat.shape, cur_image_features.shape)
                    except:
                        print("only siglip feature:", siglip_feat.shape)
                        cur_image_features = siglip_feat
                    # cur_image_features = torch.cat((siglip_feat, qwen2vl_feat), dim=0)
                    # print(siglip_feat.shape, qwen2vl_feat.shape, cur_image_features.shape)
                    cur_batch_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype
                        )
                    )
                    # build attention_mask for pack
                    if getattr(self, "use_pack", False) is False:
                        cur_new_attention_mask.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                True,
                                device=cur_attention_mask.device,
                                dtype=cur_attention_mask.dtype
                            )
                        )
                    else:
                        cur_new_attention_mask.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                cur_img_attention_mask[i],
                                device=cur_attention_mask.device,
                                dtype=cur_attention_mask.dtype
                            )
                        )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_attention_mask = torch.cat(cur_new_attention_mask)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_attention_mask.append(cur_new_attention_mask)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_attention_mask = [x[:tokenizer_model_max_length] for x in new_attention_mask]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device
        )
        new_attention_mask_padded = torch.zeros(
            (batch_size, max_len),
            dtype=new_attention_mask[0].dtype,
            device=new_attention_mask[0].device
        )
        # attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_attention_mask) \
                in enumerate(zip(new_input_embeds, new_labels, new_attention_mask)):
            cur_len = cur_new_embed.shape[0]
            if not self.training:  # for inference
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device
                    ),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_attention_mask_padded[i, -cur_len:] = cur_attention_mask
                    # attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device
                    )
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_attention_mask_padded[i, :cur_len] = cur_attention_mask
                    # attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            new_attention_mask = None
        else:
            new_attention_mask = new_attention_mask_padded

        if _position_ids is None:
            position_ids = None

        if getattr(self, "use_pack", False) is True:
            # new_attention_mask = new_attention_mask.bool()
            new_attention_mask = self._prepare_4d_causal_attention_mask_for_pack(
                new_attention_mask,
                dtype=new_input_embeds.dtype
            )  # only for pack

        return None, position_ids, new_attention_mask, past_key_values, new_input_embeds, new_labels

    def _prepare_4d_causal_attention_mask_for_pack(self, attention_mask, dtype):
        """
        Prepares a 4D causal attention mask for packed sequences.

        This function generates a 4D causal attention mask for sequences that are packed together.
        The mask ensures that each token can only attend to previous tokens within the same sequence
        and not across different sequences.

        Args:
            attention_mask (torch.Tensor): A 1D tensor where each element,
                indicating whether the corresponding token is valid (non-zero) or not (zero).
                Tokens with the same non-zero value belong to the same sequence.
                e.g. [1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0], 0 is the padding token.
            dtype (torch.dtype): The data type to use for the resulting mask.

        Returns:
            torch.Tensor: A 4D tensor of shape (bs, 1, max_len, max_len) representing the causal attention mask.
                The mask is filled with `torch.finfo(dtype).min` where tokens cannot attend and 0 where they can.
        """
        batch_size, max_len = attention_mask.shape
        tril_mask = torch.tril(
            torch.ones(
                (batch_size, 1, max_len, max_len),
                dtype=torch.bool,
                device=attention_mask.device
            )
        )
        tril_mask = tril_mask \
            & (attention_mask[:, None, None, :] == attention_mask[:, None, :, None]) \
            & (attention_mask[:, None, None, :] != 0)
        tril_mask = tril_mask.to(dtype=dtype)
        tril_mask[tril_mask == 0] = torch.finfo(dtype).min
        tril_mask[tril_mask == 1] = 0
        return tril_mask

    def initialize_vision_tokenizer(self, model_args, tokenizer, logger):
        if model_args.mm_use_im_patch_token:
            logger.info('Model is using image patch token placeholder. Adding <im_patch> to tokenizer...')
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            logger.info(
                'Model is using im_start and im_end token placeholder. Adding <im_start> and <im_end> to tokenizer...'
            )
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN],
                special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 4
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        elif getattr(model_args, "use_special_start_end_token", False):
            logger.info(
                'Model is using special token for video frame, image and grounding box.'
                'Adding <im_start>/<im_end>/<vi_start>/<vi_end>/<cor>/</cor> to tokenizer...'
            )
            num_new_tokens = tokenizer.add_tokens(
                [
                    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN,
                    DEFAULT_VI_END_TOKEN, COR_START_TOKEN, COR_END_TOKEN
                ],
                special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter and self.config.training_stage == 'stage1':
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if model's word embedding is tied with lm head, then do not freeze lm head(word embed)
                if not getattr(self.config, "tie_word_embeddings", True):
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 6
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
