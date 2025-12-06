# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import torch
import torch.nn as nn

from itertools import repeat
import collections.abc
from torch.nn.functional import scaled_dot_product_attention
from functools import partial


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Attention(nn.Module):

    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope.float() if rope is not None else None

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(1, 3)
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]

        q_type = q.dtype
        k_type = k.dtype
        if self.rope is not None:
            q = q.float()
            k = k.float()
            with torch.autocast(device_type="cuda", enabled=False):
                q = self.rope(q, xpos)
                k = self.rope(k, xpos)
            q = q.to(q_type)
            k = k.to(k_type)

        x = (
            scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=self.attn_drop.p, scale=self.scale
            )
            .transpose(1, 2)
            .reshape(B, N, C)
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):

    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope.float() if rope is not None else None

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = (
            self.projq(query)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.projk(key)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.projv(value)
            .reshape(B, Nv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q_type = q.dtype
        k_type = k.dtype
        if self.rope is not None:
            if qpos is not None:
                q = q.float()
                with torch.autocast(device_type="cuda", enabled=False):
                    q = self.rope(q, qpos)
                q = q.to(q_type)

            if kpos is not None:
                k = k.float()
                with torch.autocast(device_type="cuda", enabled=False):
                    k = self.rope(k, kpos)
                k = k.to(k_type)

        x = (
            scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=self.attn_drop.p, scale=self.scale
            )
            .transpose(1, 2)
            .reshape(B, Nq, C)
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.cross_attn = CrossAttention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


class CustomDecoderBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.cross_attn = CrossAttention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        self.norm_z = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, z, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        z_ = self.norm_z(z)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, z_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


class ModLN(nn.Module):
    """
    Modulation with adaLN.

    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """

    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):

        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mlp(mod).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]


class ConditionModulationBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(ModLN, eps=1e-6),
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mod, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x, mod), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x, mod)))
        return x


class PositionGetter(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos


class PatchEmbed(nn.Module):
    """just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.position_getter = PositionGetter()

    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        torch._assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import dust3r.utils.path_to_croco
    from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
    from functools import partial
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)

    enc_blocks_ray_map = (
        nn.ModuleList(
            [
                Block(
                    768,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=RoPE2D(100),
                )
                for _ in range(2)
            ]
        )
        .cuda()
        .train()
    )

    x = torch.randn(2, 196, 768, requires_grad=True).cuda()
    xpos = torch.arange(0, 196).unsqueeze(0).unsqueeze(-1).repeat(2, 1, 2).cuda().long()
    enc_blocks_ray_map.zero_grad()
    for blk in enc_blocks_ray_map:

        x = checkpoint(blk, x, xpos)
    enc_blocks_ray_map.zero_grad()
    x.sum().backward()

    grad_not_checkpointed = {}
    for name, param in enc_blocks_ray_map.named_parameters():
        grad_not_checkpointed[name] = param.grad.data.clone()
        print(name, grad_not_checkpointed[name])
        break
