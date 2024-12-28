from transformers import PretrainedConfig


siglip_config = PretrainedConfig.from_dict(
    {
        "attention_dropout": 0.0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "layer_norm_eps": 1e-06,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }
)

qwen2vl_vit_config = PretrainedConfig.from_dict(
    {
        "depth": 32,
        "embed_dim": 1280,
        "hidden_act": "quick_gelu",
        "hidden_size": 3584,
        "in_channels": 3,
        "in_chans": 3,
        "mlp_ratio": 4,
        "model_type": "qwen2_vl",
        "num_heads": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "temporal_patch_size": 2,
        "_attn_implementation": "flash_attention_2",
        "_attn_implementation_internal": "flash_attention_2"
    }
)
