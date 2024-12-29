import torch
from torch import nn
import torch.nn.functional as F


class LavitTokenCompressor(nn.Module):
    """
    A PyTorch module for compressing tokens using LaVIT.
    Reference: https://github.com/jy0205/LaVIT/blob/main/LaVIT/models/modeling_visual_tokenzier.py

    This module compresses input tokens by reducing their spatial dimensions.
    It uses Gumbel-Softmax sampling to select the tokens to keep.
    The number of tokens to keep in each image is UNCERTAIN.

    Attributes:
        embed_dim (int): The input tensor's embedding dimension. Default is 2048.
        inner_dim (int): The inner dimension for the 2-layer MLP. Default is 64.

    Example:
    >>> compressor = LavitTokenCompressor(embed_dim=4096, inner_dim=64)
    >>> input_tensor = torch.randn(2, 256, 4096)  # Shape: [B, N, dim]
    >>> output_tokens = compressor(input_tensor)
    >>> print([token.shape for token in output_tokens])  # Example output: [(114, 4096), (98, 4096))]
    """

    def __init__(self, embed_dim=2048, inner_dim=64, **kwargs):
        super(LavitTokenCompressor, self).__init__()
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim

        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, 2),
            nn.LogSoftmax(dim=-1)
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

    def forward_features(self, x, policy):
        x = self.norm(x)
        B, N, C = x.size()
        local_x = x[:,:, :C // 2]
        global_x = (x[:,:, C // 2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C // 2)], dim=-1)
        return self.out_conv(x)

    def _inner_forward(self, x):
        B, N, C = x.shape
        mask = torch.ones((B, N, 1), dtype=x.dtype, device=x.device)
        pred_score = self.forward_features(x, mask).reshape(B, -1, 2)
        # Sample from the score distribution
        hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0]   # [N, num_patches]
        token_num = hard_keep_decision.long().sum(dim=-1)
        index_select = hard_keep_decision.bool()

        # get remained token list
        remained_token = torch.masked_select(x, index_select[:,:,None])
        remained_token = remained_token.reshape(-1, C)  # (sum_n, dim)
        remained_token_list = torch.split(remained_token, token_num.tolist())  # [(n1, dim), (n2, dim), ...]
        remained_token_list = list(remained_token_list)

        return remained_token_list

    def forward(self, x):
        if type(x) is list:
            x = [self._inner_forward(item.unsqueeze(0)).squeeze(0) for item in x]
        else:
            x = self._inner_forward(x)
        return x
