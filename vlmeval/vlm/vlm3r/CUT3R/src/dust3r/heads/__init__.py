# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

from .linear_head import LinearPts3d, LinearPts3d_Desc, LinearPts3dPose
from .dpt_head import DPTPts3dPose


def head_factory(
    head_type,
    output_mode,
    net,
    has_conf=False,
    has_depth=False,
    has_rgb=False,
    has_pose_conf=False,
    has_pose=False,
):
    """ " build a prediction head for the decoder"""
    if head_type == "linear" and output_mode == "pts3d":
        return LinearPts3d(net, has_conf, has_depth, has_rgb, has_pose_conf)
    elif head_type == "linear" and output_mode == "pts3d+pose":
        return LinearPts3dPose(net, has_conf, has_rgb, has_pose)
    elif head_type == "linear" and output_mode.startswith("pts3d+desc"):
        local_feat_dim = int(output_mode[10:])
        return LinearPts3d_Desc(net, has_conf, has_depth, local_feat_dim)
    elif head_type == "dpt" and output_mode == "pts3d":
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
        return create_dpt_head(net, has_conf=has_conf)
    elif head_type == "dpt" and output_mode == "pts3d+pose":
        return DPTPts3dPose(net, has_conf, has_rgb, has_pose)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
