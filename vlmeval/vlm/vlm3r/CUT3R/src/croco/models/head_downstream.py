# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Heads for downstream tasks
# --------------------------------------------------------

"""
A head is a module where the __init__ defines only the head hyperparameters.
A method setup(croconet) takes a CroCoNet and set all layers according to the head and croconet attributes.
The forward takes the features as well as a dictionary img_info containing the keys 'width' and 'height'
"""

import torch
import torch.nn as nn
from .dpt_block import DPTOutputAdapter


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for CroCo.
    by default, hooks_idx will be equal to:
    * for encoder-only: 4 equally spread layers
    * for encoder+decoder: last encoder + 3 equally spread layers of the decoder
    """

    def __init__(
        self,
        *,
        hooks_idx=None,
        layer_dims=[96, 192, 384, 768],
        output_width_ratio=1,
        num_channels=1,
        postprocess=None,
        **kwargs,
    ):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_blocks = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.output_width_ratio = output_width_ratio
        self.num_channels = num_channels
        self.hooks_idx = hooks_idx
        self.layer_dims = layer_dims

    def setup(self, croconet):
        dpt_args = {
            "output_width_ratio": self.output_width_ratio,
            "num_channels": self.num_channels,
        }
        if self.hooks_idx is None:
            if hasattr(croconet, "dec_blocks"):  # encoder + decoder
                step = {8: 3, 12: 4, 24: 8}[croconet.dec_depth]
                hooks_idx = [
                    croconet.dec_depth + croconet.enc_depth - 1 - i * step
                    for i in range(3, -1, -1)
                ]
            else:  # encoder only
                step = croconet.enc_depth // 4
                hooks_idx = [
                    croconet.enc_depth - 1 - i * step for i in range(3, -1, -1)
                ]
            self.hooks_idx = hooks_idx
            print(
                f"  PixelwiseTaskWithDPT: automatically setting hook_idxs={self.hooks_idx}"
            )
        dpt_args["hooks"] = self.hooks_idx
        dpt_args["layer_dims"] = self.layer_dims
        self.dpt = DPTOutputAdapter(**dpt_args)
        dim_tokens = [
            (
                croconet.enc_embed_dim
                if hook < croconet.enc_depth
                else croconet.dec_embed_dim
            )
            for hook in self.hooks_idx
        ]
        dpt_init_args = {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info["height"], img_info["width"]))
        if self.postprocess:
            out = self.postprocess(out)
        return out
