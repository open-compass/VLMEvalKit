import torch
from torch import nn
import torch.nn.functional as F


class EVOTokenCompressor(nn.Module):
    """
    A PyTorch module for compressing tokens using EVO.
    Reference: https://github.com/YifanXu74/Evo-ViT/blob/main/deit/evo_deit.py

    This module compresses input tokens by reducing their spatial dimensions according to a specified prune ratio.
    It includes normalization, a 2-layer MLP, and a pruning mechanism.

    Attributes:
        embed_dim (int): The input tensor's embedding dimension. Default is 2048.
        inner_dim (int): The inner dimension for the 2-layer MLP. Default is 64.
        prune_ratio (float): The ratio of tokens to prune. Default is 0.25.

    Example:
    >>> compressor = EVOTokenCompressor(prune_ratio=0.25)
    >>> input_tensor =torch.randn(1, 256, 4096) # Shape: [B, N, dim]
    >>> output_tensor = compressor(input_tensor)
    >>> print(output_tensor.shape) # Expected shape: [1, 64, 4096]
    """

    def __init__(self, embed_dim=2048, inner_dim=64, prune_ratio=0.25, **kwargs):
        super(EVOTokenCompressor, self).__init__()
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim

        if type(prune_ratio) is str:
            prune_ratio = eval(prune_ratio)
        self.prune_ratio = prune_ratio

        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.norm(x)
        x = self.out_conv(x)
        return F.softmax(x.squeeze(-1), dim=-1)

    def easy_gather(self, x, indices):
        # x: B,N,C; indices: B,N
        B, N, C = x.shape
        N_new = indices.shape[1]
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        indices = indices + offset
        out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
        return out

    def _inner_forward(self, x):
        B, N, C = x.shape
        N_prune = int(N * self.prune_ratio)
        pred_score = self.forward_features(x)
        _, indices = torch.sort(pred_score, dim=1, descending=True)  # torch.sort is derivable
        x = self.easy_gather(x, indices)

        image_embeds = x[:, :N_prune]

        return image_embeds

    def forward(self, x):
        if type(x) is list:
            x = [self._inner_forward(item.unsqueeze(0)).squeeze(0) for item in x]
        else:
            x = self._inner_forward(x)
        return x
