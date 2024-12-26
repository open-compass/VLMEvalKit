from torch import nn


class ROIPoolTokenCompressor(nn.Module):
    """
    A Pytorch module for compressing tokens using RoI Pooling.

    This module performs RoI Pooling on the input tensor to reduce its spatial dimensions
    by specified max_vision_token and mode.

    Attributes:
        max_vision_token (int): The max vision token number.
        mode (str): The mode for RoI Pooling. Default is "single". Options: "single" or "multiple".

    Note:
        When mode is "single", max_vision_token means the max vision token number of
        one image (no cropping) or one tile (cropping).
        When mode is "multiple", max_vision_token means the max vision token number of
        all tiles (only for cropping).

    Example:
    >>> compressor = ROIPoolTokenCompressor(max_vision_token=64, mode="single")
    >>> input_tensor = torch.randn(1, 256, 4096) # Shape: [B, N, dim], B means the number of images
    >>> output_tensor = compressor(input_tensor)
    >>> print(output_tensor.shape) # Expected shape: [1, 64, 4096]

    >>> compressor = ROIPoolTokenCompressor(max_vision_token="4*64", mode="multiple")
    >>> input_tensor = torch.randn(4, 256, 4096) # Shape: [B, N, dim], B means the number of tiles of one image
    >>> output_tensor = compressor(input_tensor)
    >>> print(output_tensor.shape) # Expected shape: [4, 64, 4096]
    """

    def __init__(self, max_vision_token, mode="single") -> None:
        super(ROIPoolTokenCompressor, self).__init__()
        assert mode in ["single", "multiple"], "Unspported mode for ROIPoolTokenCompressor"
        if type(max_vision_token) is str:
            max_vision_token = eval(max_vision_token)
        self.max_vision_token = max_vision_token
        self.mode = mode

    def _inner_forward(self, x):
        B, N, dim = x.shape
        H = W = int(N ** 0.5)

        if self.mode == "single" and N > self.max_vision_token:
            H_new = W_new = int(self.max_vision_token ** 0.5)
            x = x.view(B, H, W, dim).permute(0, 3, 1, 2)
            # different from roi pooling, but in square image, it seems the same
            x = nn.AdaptiveAvgPool2d((H_new, W_new))(x)
            x = x.permute(0, 2, 3, 1).view(B, -1, dim)

        elif self.mode == "multiple" and (B * N) > self.max_vision_token:
            H_new = W_new = int((self.max_vision_token / B) ** 0.5)
            x = x.view(B, H, W, dim).permute(0, 3, 1, 2)
            # different from roi pooling, but in square image, it seems the same
            x = nn.AdaptiveAvgPool2d((H_new, W_new))(x)
            x = x.permute(0, 2, 3, 1).view(B, -1, dim)

        return x

    def forward(self, x):
        if type(x) is list:
            x = [self._inner_forward(item.unsqueeze(0)).squeeze(0) for item in x]
        else:
            x = self._inner_forward(x)
        return x
