from .lavit import LavitTokenCompressor
from .evo import EVOTokenCompressor
from .avgpool import AvgPoolTokenCompressor
from .roipool import ROIPoolTokenCompressor
from .minicpm_resampler import MiniCPMResampler
from torch import nn


class TokenCompressorStream(nn.Module):
    def __init__(self, compressor_list, compressor_type_list) -> None:
        super(TokenCompressorStream, self).__init__()
        self.compressor_list = nn.ModuleList(compressor_list)
        self.compressor_type_list = compressor_type_list

    def has_type(self, target):
        return target in self.compressor_type_list

    def forward(self, x):
        # x can be tensor(B, N, C) or [tensor(N1, C), tensor(N2, C), ...]
        for type, compressor in zip(self.compressor_type_list, self.compressor_list):
            x = compressor(x)

        return x


def build_token_compressor(config) -> nn.Sequential:
    token_compressor_config = config.token_compressor_config

    compressor_list = []
    compressor_type_list = []
    for item in token_compressor_config:
        print(item)
        compressor_type = item["type"]
        compressor_params = item["params"]

        # build token compressor by compressor type
        if compressor_type == "lavit":
            compressor = LavitTokenCompressor(embed_dim=config.hidden_size, **compressor_params)
        elif compressor_type == "evo":
            compressor = EVOTokenCompressor(embed_dim=config.hidden_size, **compressor_params)
        elif compressor_type == "avgpool":
            compressor = AvgPoolTokenCompressor(**compressor_params)
        elif compressor_type == "roipool":
            compressor = ROIPoolTokenCompressor(**compressor_params)
        elif compressor_type == "minicpm_resampler":
            assert config.mm_projector_type == "identity_patch"
            compressor = MiniCPMResampler(embed_dim=config.hidden_size,
                                          num_heads=config.hidden_size // 128,
                                          kv_dim=config.mm_hidden_size,
                                          **compressor_params)
        else:
            raise ValueError("Unspported Compressor type!")

        compressor_list.append(compressor)
        compressor_type_list.append(compressor_type)

    print(f"building token compressor done. using: {compressor_type_list}")
    return TokenCompressorStream(compressor_list, compressor_type_list)
