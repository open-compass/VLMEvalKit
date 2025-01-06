import torch
import torch.nn as nn
import re
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class IdentityPatchMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        """
        It is used to remove the first token (cls token) in the image feature.
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced (n = v - 1)
        """
        return x[:, 1:, :]

    @property
    def config(self):
        return {"mm_projector_type": 'identity_patch'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[2, 3, 5, 8], pool_mode='max'):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes
        self.pooling_method = {'max': nn.AdaptiveMaxPool2d, 'mean': nn.AdaptiveAvgPool2d}[pool_mode]
        self.layers = [self.pooling_method(i) for i in pool_sizes]

    def forward(self, x):
        b, c, h, W = x.size()
        pooled = []
        for p in self.layers:
            pooled.append(p(x).view(b, c, -1))
        return torch.cat(pooled, -1)


class LinearAdapter(nn.Linear):
    def __init__(self, mm_hidden_size, hidden_size):
        super(LinearAdapter, self).__init__(mm_hidden_size, hidden_size)
        self.mm_projector_type = 'linear'


class ConvAdapter(nn.Module):
    def __init__(self, dim_in, dim_out, mlp_hidden_dim=None):
        super().__init__()
        self.mm_projector_type = 'conv_adapter'
        if mlp_hidden_dim is None:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
                nn.Linear(dim_out, dim_out)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, dim_out)
            )
        self.conv = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        x = self.mlp(x)

        f, v, d = x.shape
        s = int(math.sqrt(v - 1))
        x = x[:, 1:, :]  # remove cls_token
        x = x.reshape(f, s, s, d).permute([0, 3, 1, 2])
        x = self.conv(x)
        x = x.permute([0, 2, 3, 1]).reshape(f, -1, d)
        return x


class PoolAdapter(nn.Module):
    def __init__(self, dim_in, dim_out, pool_out_size=4):
        super().__init__()
        self.mm_projector_type = 'pool_adapter'
        self.pool_h, self.pool_w = pool_out_size, pool_out_size

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        f, v, d = x.shape
        # print(x.shape)  # torch.Size([image_num, vit_token_num, dim_in])  [8, 257, 1024]        f, v, d = x.shape
        s = int(math.sqrt(v - 1))
        x = x[:, 1:, :]  # remove cls_token
        x = x.reshape(f, s, s, d)

        if s % self.pool_h == 0 and s % self.pool_w == 0:
            x = x.reshape(f, self.pool_h, s // self.pool_h, self.pool_w, s // self.pool_w, d)
            x = x.permute([0, 1, 3, 5, 2, 4]).reshape(f, self.pool_h * self.pool_w, d, -1).mean(-1)
            x = self.mlp(x)  # [f, h*w, d]
        else:
            raise ValueError()

        return x


class PoolAdapterCLS(nn.Module):
    def __init__(self, dim_in, dim_out, pool_out_size=4):
        super().__init__()
        self.mm_projector_type = 'pool_adapter_w_cls'
        self.pool_h, self.pool_w = pool_out_size, pool_out_size

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        f, v, d = x.shape
        # print(x.shape)  # torch.Size([image_num, vit_token_num, dim_in])  [8, 257, 1024]        f, v, d = x.shape
        s = int(math.sqrt(v - 1))

        cls = x[:, 0, :]
        feature = x[:, 1:, :]  # remove cls_token

        feature = feature.reshape(f, s, s, d)

        if s % self.pool_h == 0 and s % self.pool_w == 0:
            feature = feature.reshape(f, self.pool_h, s // self.pool_h, self.pool_w, s // self.pool_w, d)
            feature = feature.permute([0, 1, 3, 5, 2, 4]).reshape(f, self.pool_h * self.pool_w, d, -1).mean(-1)
            feature = torch.concat([cls.unsqueeze(1), feature], dim=1)
            feature = self.mlp(feature)  # [f, h*w, d]
        else:
            raise ValueError()

        return feature


class AdaptPooler(nn.Module):
    def __init__(self, dim_in, dim_out, pool_out_size=4):
        super().__init__()
        self.mm_projector_type = 'adapt_pooler'
        self.pool_h, self.pool_w = pool_out_size, pool_out_size
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        x = self.mlp(x)

        f, v, d = x.shape
        s = int(math.sqrt(v - 1))
        x = x[:, 1:, :]  # remove cls_token
        x = x.reshape(f, s, s, d)

        x = x.reshape(f, self.pool_h, s // self.pool_h, self.pool_w, s // self.pool_w, d)
        x = x.permute([0, 1, 3, 5, 2, 4]).reshape(f, self.pool_h * self.pool_w, d, -1).mean(-1)
        return x


class AdaptPoolerCLS(nn.Module):
    def __init__(self, dim_in, dim_out, pool_out_size=4):
        super().__init__()
        self.mm_projector_type = 'adapt_pooler_w_cls'
        self.pool_h, self.pool_w = pool_out_size, pool_out_size
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        x = self.mlp(x)

        f, v, d = x.shape
        s = int(math.sqrt(v - 1))
        cls = x[:, 0, :]
        feature = x[:, 1:, :]  # remove cls_token
        feature = feature.reshape(f, s, s, d)

        feature = feature.reshape(f, self.pool_h, s // self.pool_h, self.pool_w, s // self.pool_w, d)
        feature = feature.permute([0, 1, 3, 5, 2, 4]).reshape(f, self.pool_h * self.pool_w, d, -1).mean(-1)
        return torch.concat([cls.unsqueeze(1), feature], dim=1)


class AdaptPyraPooler(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mm_projector_type = 'adapt_pyrapooler'
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )
        self.pool = SpatialPyramidPooling([2, 3, 5, 8], pool_mode='max')

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        x = self.mlp(x)
        f, v, d = x.shape
        s = int(math.sqrt(v - 1))
        x = x[:, 1:, :]  # remove cls_token
        x = x.reshape(f, s, s, d).permute([0, 3, 1, 2])
        x = self.pool(x).permute([0, 2, 1])
        return x


class MlpPixelShuffle(nn.Module):
    def __init__(self, dim_in, dim_out, pixelshuffle_downsample_ratio, mlp_hidden_dim=None):
        super().__init__()
        self.mm_projector_type = 'mlp_pixel_shuffle'
        if mlp_hidden_dim is None:
            self.mlp = nn.Sequential(
                nn.Linear(int(dim_in * (pixelshuffle_downsample_ratio ** 2)), dim_out),
                nn.GELU(),
                nn.Linear(dim_out, dim_out)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(int(dim_in * (pixelshuffle_downsample_ratio ** 2)), mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, dim_out)
            )
        self.scale_factor = pixelshuffle_downsample_ratio

    def pixel_shuffle(self, x, scale_factor=2):
        # change scale_factor from float to int

        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H / scale, C * scale
        x = x.view(n, w, int(h / scale_factor), int(c * scale_factor))
        # N, W, H / scale, C * scale --> N, H / scale, W, C * scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H / scale, W, C * scale --> N, H / scale, W / scale, C * (scale ** 2)
        x = x.view(n, int(h / scale_factor), int(w / scale_factor),
                   int(c * (scale_factor * scale_factor)))

        x = x.permute(0, 2, 1, 3).contiguous()

        return x

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        x = x[:, 1:, :]  # remove cls_token
        h = w = int(x.shape[1] ** 0.5)
        x = x.view(x.shape[0], h, w, -1)
        x = self.pixel_shuffle(x, self.scale_factor)
        x = self.mlp(x)
        x = x.view(x.shape[0],-1,x.shape[-1])
        return x


class OvisConvAdapter(nn.Module):
    def __init__(self, dim_in, dim_out, vocab_size, tokenize_function="softmax"):
        super().__init__()
        self.mm_projector_type = 'ovis_conv_adapter'
        self.conv = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_in, vocab_size, bias=False),
            torch.nn.LayerNorm(vocab_size)
        )
        self.embedding = torch.nn.Embedding(vocab_size, dim_out)
        self.tokenize_function = tokenize_function

    def tokenize(self, logits):
        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.tokenize_function == 'softmax':
            tokens = torch.nn.functional.softmax(logits, dim=-1)
        elif self.tokenize_function == 'gumbel_argmax':
            tokens = torch.nn.functional.gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                'Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax,'
                f' but got {self.config.tokenize_function}'
            )
        return tokens

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (F, v, D)
        Returns:
            shape (F, n, D) where n is token_num that has been reduced
        """
        # conv
        f, v, d = x.shape
        s = int(math.sqrt(v - 1))
        x = x[:, 1:, :]  # remove cls_token
        x = x.reshape(f, s, s, d).permute([0, 3, 1, 2])
        x = self.conv(x)
        x = x.permute([0, 2, 3, 1]).reshape(f, -1, d)

        # tokenize
        logits = self.mlp(x)
        visual_tokens = self.tokenize(logits)

        # get embeddings
        out = torch.matmul(visual_tokens, self.embedding.weight)

        return out


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return LinearAdapter(config.mm_hidden_size, config.hidden_size)
    elif projector_type == 'pool_adapter':
        return PoolAdapter(config.mm_hidden_size, config.hidden_size, config.pool_out_size)
    elif projector_type == 'adapt_pooler':
        return AdaptPooler(config.mm_hidden_size, config.hidden_size, config.pool_out_size)
    elif projector_type == 'adapt_pyrapooler':
        return AdaptPyraPooler(config.mm_hidden_size, config.hidden_size)
    elif projector_type == 'adapt_pooler_w_cls':
        return AdaptPoolerCLS(config.mm_hidden_size, config.hidden_size, config.pool_out_size)
    elif projector_type == 'pool_adapter_w_cls':
        return PoolAdapterCLS(config.mm_hidden_size, config.hidden_size, config.pool_out_size)
    elif projector_type == 'conv_adapter':
        return ConvAdapter(config.mm_hidden_size, config.hidden_size, getattr(config, "mlp_hidden_dim", None))
    elif projector_type == 'mlp_pixel_shuffle':
        return MlpPixelShuffle(config.mm_hidden_size, config.hidden_size,
                               config.pixelshuffle_downsample_ratio, getattr(config, "mlp_hidden_dim", None))
    elif projector_type == 'ovis_conv_adapter':
        return OvisConvAdapter(config.mm_hidden_size, config.hidden_size, getattr(config, "mlp_hidden_dim", 32000),
                               getattr(config, "tokenize_function", "softmax"))

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            mm_projector = nn.Sequential(*modules)
            # this line is for fixing bug in valley/model/valley_arch.py line 72.
            # If the projector is 2 layer mlp, projector has no attr named mm_projector_type.
            mm_projector.mm_projector_type = projector_type
        return mm_projector

    if projector_type == 'identity':
        return IdentityMap()

    if projector_type == 'identity_patch':
        return IdentityPatchMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
