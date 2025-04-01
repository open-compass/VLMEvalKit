import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import (
        AttentionPoolLatent,
        DropPath,
        LayerType,
        Mlp,
        PatchDropout,
        PatchEmbed,
        resample_abs_pos_embed,
    )
    from timm.models._manipulate import checkpoint_seq, named_apply
except:
    print("Wrong timm version")

from flash_attn import flash_attn_func, flash_attn_varlen_func

from typing import Optional

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed
import os

if "LOAD_VISION_EARLY" in os.environ:
    print("LOAD_VISION_EARLY is set")
    LOAD_VISION_EARLY = True
else:
    LOAD_VISION_EARLY = False

if "FORCE_NO_DOWNSAMPLE" in os.environ:
    print("FORCE_NO_DOWNSAMPLE is set")
    FORCE_NO_DOWNSAMPLE = True
else:
    FORCE_NO_DOWNSAMPLE = False


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""The original timm.models.layers.weight_init.trunc_normal_ can not handle bfloat16 yet, here we first
    convert the tensor to float32, apply the trunc_normal_() in float32, and then convert it back to its orignal dtype.
    Fills the input Tensor with values drawn from a truncated normal distribution. The values are effectively drawn
    from the normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    with torch.no_grad():
        dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        tensor_fp32 = _no_grad_trunc_normal_(tensor_fp32, mean, std, a, b)
        tensor_dtype = tensor_fp32.to(dtype=dtype)
        tensor.copy_(tensor_dtype)


def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
    trunc_normal_(self.latent, std=self.latent_dim**-0.5)


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        # self.fused_attn = use_fused_attn()
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if cu_slens is not None:
            q = q.permute(0, 2, 1, 3)  # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            max_seqlen = torch.max(cu_slens[1:] - cu_slens[:-1]).item()
            x = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_slens,
                cu_seqlens_k=cu_slens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.scale,
                causal=False,
            )

            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

        else:
            q = q.permute(0, 2, 1, 3)  # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(q, k, v, softmax_scale=self.scale)  # -> b, n, h, c

            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q,
        #         k,
        #         v,
        #         dropout_p=self.attn_drop.p if self.training else 0.0,
        #     )
        # else:
        #     q = q * self.scale
        #     attn = q @ k.transpose(-2, -1)
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     x = attn @ v

        # x = x.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_slens=cu_slens)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        strict_img_size: bool = False,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        ignore_head: bool = False,
        add_patch2x2: bool = False,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        # norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        # act_layer = get_act_layer(act_layer) or nn.GELU
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            strict_img_size=strict_img_size,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)

        # deepspeed.zero.register_external_parameter(self, self.pos_embed)
        # deepspeed.zero.register_external_parameter(self, self.patch_embed.proj.weight)
        # deepspeed.zero.register_external_parameter(self, self.patch_embed.proj.bias)
        # print(self.patch_embed.state_dict().keys())

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )

        if add_patch2x2:
            if add_patch2x2 == "v2":
                self.downsample = nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.Conv2d(embed_dim * 2, embed_dim * 4, 1),
                )
            else:
                mid_dim = embed_dim * 2
                self.downsample = nn.Sequential(
                    nn.Conv2d(embed_dim, mid_dim, kernel_size=2, stride=2), nn.GELU(), nn.Conv2d(mid_dim, mid_dim, 1)
                )

        else:
            self.downsample = None

        # self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # # Classifier Head
        # if global_pool == "map":
        #     AttentionPoolLatent.init_weights = init_weights
        #     self.attn_pool = AttentionPoolLatent(
        #         self.embed_dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         norm_layer=norm_layer,
        #     )
        # else:
        #     self.attn_pool = None
        # self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        # self.head_drop = nn.Dropout(drop_rate)
        # self.head = (
        #     nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # )

        # if weight_init != "skip":
        #     self.init_weights(weight_init)

    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""] = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        # head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map " and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def rescale_positional_embedding(self, out_size):
        h, w = out_size
        pos_embed_shape = int((self.pos_embed.shape[1]) ** 0.5)
        if (h, w) == (pos_embed_shape, pos_embed_shape):
            return self.pos_embed
        rescaled_positional_embedding = self.pos_embed.new_zeros(1, h * w, self.pos_embed.shape[2])
        pe_2d = self.pos_embed[0].T.contiguous().view(1, -1, pos_embed_shape, pos_embed_shape)
        pe_2d = F.interpolate(pe_2d, out_size, mode="bilinear", align_corners=False).view(-1, h * w)
        rescaled_positional_embedding[0] = pe_2d.T.contiguous()
        return rescaled_positional_embedding

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features_list(self, x_list):
        x_all = []
        image_sizes = []
        for x in x_list:
            bs, _, h, w = x.shape

            # fix patch size=14 in datasets
            pad_h = (self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]) % self.patch_embed.patch_size[
                0
            ]
            pad_w = (self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]) % self.patch_embed.patch_size[
                1
            ]
            x = F.pad(x, (0, pad_w, 0, pad_h))

            bs, _, h, w = x.shape

            h = h // self.patch_embed.patch_size[0]
            w = w // self.patch_embed.patch_size[1]

            x = self.patch_embed(x)
            # x = self._pos_embed(x)
            x = x + self.rescale_positional_embedding(out_size=(h, w))
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            x_all.append(x)
            image_sizes.append((h, w))

        slen = [xi.size(1) for xi in x_all]
        x = torch.cat(x_all, dim=1)

        cu_indices = [
            0,
        ]
        for i in slen:
            cu_indices.append(cu_indices[-1] + i)

        cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(x.device)
        for idx, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, cu_slens, use_reentrant=True)
            else:
                x = blk(x, cu_slens=cu_slens)
        feats = x.split(slen, dim=1)  # [(1, slen, c)]

        if self.downsample is not None:
            new_feats = []
            new_sizes = []
            for f, s in zip(feats, image_sizes):
                h, w = s
                b, n, c = f.size()
                f = f.reshape(b, h, w, c).permute(0, 3, 1, 2)
                f = self.downsample(f)
                b, c, h, w = f.size()
                f = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
                new_feats.append(f)
                new_sizes.append((h, w))
            return new_feats, new_sizes

        return feats, image_sizes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]

        x = self.patch_embed(x)
        # x = self._pos_embed(x)
        x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        if self.downsample is not None:
            b, n, c = x.size()
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            x = self.downsample(x)
            b, c, h, w = x.size()
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            new_feats = x
            new_sizes = (h, w)
            return new_feats, new_sizes

        return x, (h, w)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm(x)
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, cal_attn_pool=False):
        if type(x) is list:
            x, image_sizes = self.forward_features_list(x)
            return x, image_sizes, None
        else:
            x, image_sizes = self.forward_features(x)
            return x, image_sizes, None


@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False


SigLIP_MODEL_CONFIG = {
    "siglip_so400m_patch14_384": {
        "image_size": 384,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_so400m_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_so400m_patch14_224": {
        "image_size": 224,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_large_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4,
        "global_pool": "map",
        "use_checkpoint": False,
    },
}


def resize_evaclip_pos_embed(model: VisionTransformer, interpolation: str = "bicubic"):
    # interpolate position embedding
    orig_size = 24
    new_size = 128
    pos_tokens = model.pos_embed
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, model.embed_dim).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode=interpolation, align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    model.pos_embed = nn.Parameter(pos_tokens, requires_grad=True)
    return model


def create_siglip_vit(
    model_name: str = "siglip_so400m_patch14_384",
    image_size: int = 384,
    select_layer: int = -1,
    path: str = "",
    gradient_checkpointing: bool = False,
    **kwargs,
):
    assert model_name in SigLIP_MODEL_CONFIG.keys(), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"

    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])

    if select_layer <= 0:
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:
        layers = min(vision_cfg.layers, select_layer)

    if "patch2x2" or "patch4x4" in path:
        add_patch2x2 = True
    else:
        add_patch2x2 = False

    if "patch4x4pool" in path or "patch2x2from4x4" in path:
        add_patch2x2 = "v2"

    if FORCE_NO_DOWNSAMPLE:
        add_patch2x2 = False

    model = VisionTransformer(
        img_size=2048,
        patch_size=16,
        embed_dim=vision_cfg.width,
        depth=layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        dynamic_img_pad=False,
        strict_img_size=False,
        ignore_head=kwargs.get("ignore_head", False),
        weight_init=kwargs.get("weight_init", "skip"),
        num_classes=0,
        add_patch2x2=add_patch2x2,
    )

    print("#### Skip loading vision backbone")

    if gradient_checkpointing:
        model.set_grad_checkpointing(True)
    return model


from transformers import CLIPImageProcessor
import torch.distributed as dist


class SigLIPViTAnysizeWrapper(nn.Module):
    def __init__(self, vision_tower, path, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.args = args
        self.path = path

        self.select_layer = -1
        if self.select_layer < -1:
            self.select_layer += 1
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.output_dim = 1152

        if not delay_load or LOAD_VISION_EARLY:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if (
            self.args.mm_projector_type == "conv_mlp"
            or self.args.mm_projector_type == "multipath_conv_mlp"
            or self.args.mm_projector_type == "multipath_conv_mlp_woconv"
        ):
            self.image_processor.crop_size["height"] = 384
            self.image_processor.crop_size["width"] = 384
            self.image_processor.size["shortest_edge"] = 384
            print("Resizeing clip processor to 384...")
        self.image_processor.image_mean = [0.5, 0.5, 0.5]
        self.image_processor.image_std = [0.5, 0.5, 0.5]
        print("Loading vision model...")

        self.vision_tower = create_siglip_vit(
            path=self.path, model_name="siglip_so400m_patch16_384", gradient_checkpointing=False
        )
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.vision_tower.eval()
        self.is_loaded = True

    def train(self, mode=True):
        self.training = mode

        if self.is_loaded:
            self.vision_tower.eval()

    def split_images(self, images, split_res=512, base_size=32):
        split_images = []
        sub_images_info = []
        for image in images:
            now_sub_images = []
            _, c, h, w = image.shape
            if h * w <= split_res * split_res:
                split_images.append(image)
                sub_images_info.append(
                    (1, 1, 1, h // base_size, w // base_size, [(0, h // base_size, 0, w // base_size)])
                )
                continue
            nsplit_h = math.ceil(h / split_res)
            nsplit_w = math.ceil(w / split_res)
            sub_h = int(h / nsplit_h / base_size) * base_size
            sub_w = int(w / nsplit_w / base_size) * base_size
            crop_infos = []
            for i in range(nsplit_h):
                for j in range(nsplit_w):
                    begin_h = i * sub_h
                    begin_w = j * sub_w

                    if i == nsplit_h - 1:
                        end_h = h
                    else:
                        end_h = (i + 1) * sub_h

                    if j == nsplit_w - 1:
                        end_w = w
                    else:
                        end_w = (j + 1) * sub_w

                    assert (end_h - begin_h) % base_size == 0 and (end_w - begin_w) % base_size == 0

                    sub_image = image[:, :, begin_h:end_h, begin_w:end_w]
                    now_sub_images.append(sub_image)
                    crop_infos.append(
                        (begin_h // base_size, end_h // base_size, begin_w // base_size, end_w // base_size)
                    )

            split_images += now_sub_images
            sub_images_info.append(
                (len(now_sub_images), nsplit_h, nsplit_w, h // base_size, w // base_size, crop_infos)
            )

        return split_images, sub_images_info

    def unsplit_images(self, features, sizes, sub_images_info):
        new_features = []
        for feature, size in zip(features, sizes):
            h, w = size
            new_features.append(feature.reshape(1, h, w, -1))

        fused_images = []
        images_sizes = []
        sub_count = 0
        for n_split, nsplit_h, nsplit_w, total_h, total_w, crop_infos in sub_images_info:
            sub_features = new_features[sub_count : sub_count + n_split]
            sub_count += n_split

            total_feature = new_features[0].new_zeros(1, total_h, total_w, self.hidden_size)
            for feature, (begin_h, end_h, begin_w, end_w) in zip(sub_features, crop_infos):
                total_feature[:, begin_h:end_h, begin_w:end_w] += feature

            fused_images.append(total_feature.reshape(1, total_h * total_w, self.hidden_size))
            images_sizes.append((total_h, total_w))

        return fused_images, images_sizes

    def forward_func(self, images, force_fix_size=False, cal_attn_pool=False):
        if type(images) is list:
            xs = [x.to(self.dtype) for x in images]
            image_features, img_size, cls_token = self.vision_tower(xs, cal_attn_pool=cal_attn_pool)
            image_features = [x.to(images[0].dtype) for x in image_features]

        else:
            image_forward_outs, img_size, cls_token = self.vision_tower(
                images.to(self.dtype), cal_attn_pool=cal_attn_pool
            )
            image_features = image_forward_outs.to(images.dtype)

        return image_features, img_size, cls_token

    def forward(self, images, cal_attn_pool=False):
        with torch.no_grad():
            image_features, img_size, cls_token = self.forward_func(images, cal_attn_pool=cal_attn_pool)
            return image_features, img_size

    @property
    def dummy_feature(self):
        return torch.zeros(1, 1152, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.pos_embed.dtype

    @property
    def device(self):
        return self.vision_tower.pos_embed.device

    @property
    def hidden_size(self):
        return self.output_dim

    @property
    def config(self):
        return type(
            "LLaVAConfigWrapper",
            (),
            {
                # 'image_size': 224,
                "patch_size": 16,
            },
        )()
