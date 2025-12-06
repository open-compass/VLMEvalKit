# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from dust3r.heads.postprocess import (
    postprocess,
    postprocess_desc,
    postprocess_rgb,
    postprocess_pose_conf,
    postprocess_pose,
    reg_dense_conf,
)
import dust3r.utils.path_to_croco  # noqa: F401
from models.dpt_block import DPTOutputAdapter  # noqa
from dust3r.utils.camera import pose_encoding_to_camera, PoseDecoder
from dust3r.blocks import ConditionModulationBlock
from torch.utils.checkpoint import checkpoint


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)

        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert (
            self.dim_tokens_enc is not None
        ), "Need to call init(dim_tokens_enc) function first"

        image_size = self.image_size if image_size is None else image_size
        H, W = image_size

        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        layers = [encoder_tokens[hook] for hook in self.hooks]

        layers = [self.adapt_tokens(l) for l in layers]

        layers = [
            rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers
        ]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]

        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        path_4 = self.scratch.refinenet4(layers[3])[
            :, :, : layers[2].shape[2], : layers[2].shape[3]
        ]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        out = self.head(path_1)

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(
        self,
        *,
        n_cls_token=0,
        hooks_idx=None,
        dim_tokens=None,
        output_width_ratio=1,
        num_channels=1,
        postprocess=None,
        depth_mode=None,
        conf_mode=None,
        **kwargs
    ):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(
            output_width_ratio=output_width_ratio, num_channels=num_channels, **kwargs
        )
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(net, has_conf=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim // 2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(
        num_channels=out_nchan + has_conf,
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
        dim_tokens=[ed, dd, dd, dd],
        postprocess=postprocess,
        depth_mode=net.depth_mode,
        conf_mode=net.conf_mode,
        head_type="regression",
    )


class DPTPts3dPose(nn.Module):
    def __init__(self, net, has_conf=False, has_rgb=False, has_pose=False):
        super(DPTPts3dPose, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode

        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose

        pts_channels = 3 + has_conf
        rgb_channels = has_rgb * 3
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        hooks_idx = [0, 1, 2, 3]
        dim_tokens = [ed, dd, dd, dd]
        head_type = "regression"
        output_width_ratio = 1

        pts_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=pts_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        rgb_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=rgb_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        if hooks_idx is not None:
            pts_dpt_args.update(hooks=hooks_idx)
            rgb_dpt_args.update(hooks=hooks_idx)

        self.dpt_self = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_self.init(**dpt_init_args)

        self.final_transform = nn.ModuleList(
            [
                ConditionModulationBlock(
                    net.dec_embed_dim,
                    net.dec_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    rope=net.rope,
                )
                for _ in range(2)
            ]
        )

        self.dpt_cross = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_cross.init(**dpt_init_args)

        if has_rgb:
            self.dpt_rgb = DPTOutputAdapter_fix(**rgb_dpt_args)
            dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
            self.dpt_rgb.init(**dpt_init_args)

        if has_pose:
            in_dim = net.dec_embed_dim
            self.pose_head = PoseDecoder(hidden_size=in_dim)

    def forward(self, x, img_info, **kwargs):
        if self.has_pose:
            pose_token = x[-1][:, 0].clone()
            token = x[-1][:, 1:]
            with torch.cuda.amp.autocast(enabled=True):
                pose = self.pose_head(pose_token)

            token_cross = token.clone()
            for blk in self.final_transform:
                token_cross = blk(token_cross, pose_token, kwargs.get("pos"))
            x = x[:-1] + [token]
            x_cross = x[:-1] + [token_cross]

        with torch.cuda.amp.autocast(enabled=True):
            self_out = checkpoint(
                self.dpt_self,
                x,
                image_size=(img_info[0], img_info[1]),
                use_reentrant=False,
            )

            final_output = postprocess(self_out, self.depth_mode, self.conf_mode)
            final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            final_output["conf_self"] = final_output.pop("conf")

            if self.has_rgb:
                rgb_out = checkpoint(
                    self.dpt_rgb,
                    x,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                rgb_output = postprocess_rgb(rgb_out)
                final_output.update(rgb_output)

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7
                cross_out = checkpoint(
                    self.dpt_cross,
                    x_cross,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                tmp = postprocess(cross_out, self.depth_mode, self.conf_mode)
                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                final_output["conf"] = tmp.pop("conf")
        return final_output
