from torch import nn


class AvgPoolTokenCompressor(nn.Module):
    """
    A PyTorch module for compressing tokens using average pooling.

    This module performs average pooling on the input tensor to reduce its spatial dimensions
    by a specified scale factor.

    Attributes:
        scale (int): The scale factor for downsampling.

    Example:
    >>> compressor = AvgPoolTokenCompressor(scale=2)
    >>> input_tensor = torch.randn(1, 256, 4096) # Shape: [B, N, dim]
    >>> output_tensor = compressor(input_tensor)
    >>> print(output_tensor.shape) # Expected shape: [1, 64, 4096]
    """

    def __init__(self, scale) -> None:
        super(AvgPoolTokenCompressor, self).__init__()
        self.scale = scale

    def _inner_forward(self, x):
        scale = self.scale
        B, N, dim = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, dim)

        return x.view(B, H // scale, scale, W // scale, scale, dim) \
            .permute(0, 1, 3, 5, 2, 4) \
            .reshape(B, H // scale, W // scale, dim, scale * scale) \
            .mean(dim=-1) \
            .squeeze(dim=-1) \
            .reshape(B, -1, dim)

    def forward(self, x):
        if type(x) is list:
            x = [self._inner_forward(item.unsqueeze(0)).squeeze(0) for item in x]
        else:
            x = self._inner_forward(x)
        return x
