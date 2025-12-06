# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import torch
import torch.nn as nn
import torch.nn.functional as F
from dust3r.heads.postprocess import (
    postprocess,
    postprocess_desc,
    postprocess_rgb,
    postprocess_pose_conf,
    postprocess_pose,
    reg_dense_conf,
)
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa
from dust3r.utils.geometry import geotrf
from dust3r.utils.camera import pose_encoding_to_camera, PoseDecoder
from dust3r.blocks import ConditionModulationBlock


class LinearPts3d(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(
        self, net, has_conf=False, has_depth=False, has_rgb=False, has_pose_conf=False
    ):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose_conf = has_pose_conf
        self.has_depth = has_depth
        self.proj = Mlp(
            net.dec_embed_dim, out_features=(3 + has_conf) * self.patch_size**2
        )
        if has_depth:
            self.self_proj = Mlp(
                net.dec_embed_dim, out_features=(3 + has_conf) * self.patch_size**2
            )
        if has_rgb:
            self.rgb_proj = Mlp(net.dec_embed_dim, out_features=3 * self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(
            B, -1, H // self.patch_size, W // self.patch_size
        )
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        final_output = postprocess(feat, self.depth_mode, self.conf_mode)
        final_output["pts3d_in_other_view"] = final_output.pop("pts3d")

        if self.has_depth:
            self_feat = self.self_proj(tokens)  # B,S,D
            self_feat = self_feat.transpose(-1, -2).view(
                B, -1, H // self.patch_size, W // self.patch_size
            )
            self_feat = F.pixel_shuffle(self_feat, self.patch_size)  # B,3,H,W
            self_3d_output = postprocess(self_feat, self.depth_mode, self.conf_mode)
            self_3d_output["pts3d_in_self_view"] = self_3d_output.pop("pts3d")
            self_3d_output["conf_self"] = self_3d_output.pop("conf")
            final_output.update(self_3d_output)

        if self.has_rgb:
            rgb_feat = self.rgb_proj(tokens)
            rgb_feat = rgb_feat.transpose(-1, -2).view(
                B, -1, H // self.patch_size, W // self.patch_size
            )
            rgb_feat = F.pixel_shuffle(rgb_feat, self.patch_size)  # B,3,H,W
            rgb_output = postprocess_rgb(rgb_feat)
            final_output.update(rgb_output)

        if self.has_pose_conf:
            pose_conf = self.pose_conf_proj(tokens)
            pose_conf = pose_conf.transpose(-1, -2).view(
                B, -1, H // self.patch_size, W // self.patch_size
            )
            pose_conf = F.pixel_shuffle(pose_conf, self.patch_size)
            pose_conf_output = postprocess_pose_conf(pose_conf)
            final_output.update(pose_conf_output)

        return final_output


class LinearPts3d_Desc(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(
        self,
        net,
        has_conf=False,
        has_depth=False,
        local_feat_dim=24,
        hidden_dim_factor=4.0,
    ):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf
        self.double_channel = has_depth
        self.local_feat_dim = local_feat_dim

        if not has_depth:
            self.proj = nn.Linear(
                net.dec_embed_dim, (3 + has_conf) * self.patch_size**2
            )
        else:
            self.proj = nn.Linear(
                net.dec_embed_dim, (3 + has_conf) * 2 * self.patch_size**2
            )
        idim = net.enc_embed_dim + net.dec_embed_dim
        self.head_local_features = Mlp(
            in_features=idim,
            hidden_features=int(hidden_dim_factor * idim),
            out_features=(self.local_feat_dim + 1) * self.patch_size**2,
        )

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(
            B, -1, H // self.patch_size, W // self.patch_size
        )
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(
            B, -1, H // self.patch_size, W // self.patch_size
        )
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W
        feat = torch.cat([feat, local_features], dim=1)

        return postprocess_desc(
            feat,
            self.depth_mode,
            self.conf_mode,
            self.local_feat_dim,
            self.double_channel,
        )


class LinearPts3dPoseDirect(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, has_conf=False, has_rgb=False, has_pose=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode
        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose

        self.proj = Mlp(
            net.dec_embed_dim, out_features=(3 + has_conf) * self.patch_size**2
        )
        if has_rgb:
            self.rgb_proj = Mlp(net.dec_embed_dim, out_features=3 * self.patch_size**2)
        if has_pose:
            self.pose_head = PoseDecoder(hidden_size=net.dec_embed_dim)
        if has_conf:
            self.cross_conf_proj = Mlp(
                net.dec_embed_dim, out_features=self.patch_size**2
            )

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        if self.has_pose:
            pose_token = tokens[:, 0]
            tokens = tokens[:, 1:]
        B, S, D = tokens.shape

        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(
            B, -1, H // self.patch_size, W // self.patch_size
        )
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W
        final_output = postprocess(feat, self.depth_mode, self.conf_mode)
        final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
        final_output["conf_self"] = final_output.pop("conf")

        if self.has_rgb:
            rgb_feat = self.rgb_proj(tokens)
            rgb_feat = rgb_feat.transpose(-1, -2).view(
                B, -1, H // self.patch_size, W // self.patch_size
            )
            rgb_feat = F.pixel_shuffle(rgb_feat, self.patch_size)  # B,3,H,W
            rgb_output = postprocess_rgb(rgb_feat)
            final_output.update(rgb_output)

        if self.has_pose:
            pose = self.pose_head(pose_token)
            pose = postprocess_pose(pose, self.pose_mode)
            final_output["camera_pose"] = pose  # B,7
            final_output["pts3d_in_other_view"] = geotrf(
                pose_encoding_to_camera(final_output["camera_pose"]),
                final_output["pts3d_in_self_view"],
            )

        if self.has_conf:
            cross_conf = self.cross_conf_proj(tokens)
            cross_conf = cross_conf.transpose(-1, -2).view(
                B, -1, H // self.patch_size, W // self.patch_size
            )
            cross_conf = F.pixel_shuffle(cross_conf, self.patch_size)[:, 0]
            final_output["conf"] = reg_dense_conf(cross_conf, mode=self.conf_mode)
        return final_output


class LinearPts3dPose(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(
        self, net, has_conf=False, has_rgb=False, has_pose=False, mlp_ratio=4.0
    ):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode
        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose

        self.proj = Mlp(
            net.dec_embed_dim,
            hidden_features=int(mlp_ratio * net.dec_embed_dim),
            out_features=(3 + has_conf) * self.patch_size**2,
        )
        if has_rgb:
            self.rgb_proj = Mlp(
                net.dec_embed_dim,
                hidden_features=int(mlp_ratio * net.dec_embed_dim),
                out_features=3 * self.patch_size**2,
            )
        if has_pose:
            self.pose_head = PoseDecoder(hidden_size=net.dec_embed_dim)
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
            self.cross_proj = Mlp(
                net.dec_embed_dim,
                hidden_features=int(mlp_ratio * net.dec_embed_dim),
                out_features=(3 + has_conf) * self.patch_size**2,
            )

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape, **kwargs):
        H, W = img_shape
        tokens = decout[-1]
        if self.has_pose:
            pose_token = tokens[:, 0]
            tokens = tokens[:, 1:]
            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)
            cross_tokens = tokens
            for blk in self.final_transform:
                cross_tokens = blk(cross_tokens, pose_token, kwargs.get("pos"))

        with torch.cuda.amp.autocast(enabled=False):
            B, S, D = tokens.shape

            feat = self.proj(tokens)  # B,S,D
            feat = feat.transpose(-1, -2).view(
                B, -1, H // self.patch_size, W // self.patch_size
            )
            feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W
            final_output = postprocess(
                feat, self.depth_mode, self.conf_mode, pos_z=True
            )
            final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            final_output["conf_self"] = final_output.pop("conf")

            if self.has_rgb:
                rgb_feat = self.rgb_proj(tokens)
                rgb_feat = rgb_feat.transpose(-1, -2).view(
                    B, -1, H // self.patch_size, W // self.patch_size
                )
                rgb_feat = F.pixel_shuffle(rgb_feat, self.patch_size)  # B,3,H,W
                rgb_output = postprocess_rgb(rgb_feat)
                final_output.update(rgb_output)

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7

                cross_feat = self.cross_proj(cross_tokens)  # B,S,D
                cross_feat = cross_feat.transpose(-1, -2).view(
                    B, -1, H // self.patch_size, W // self.patch_size
                )
                cross_feat = F.pixel_shuffle(cross_feat, self.patch_size)  # B,3,H,W
                tmp = postprocess(cross_feat, self.depth_mode, self.conf_mode)
                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                final_output["conf"] = tmp.pop("conf")

            return final_output
