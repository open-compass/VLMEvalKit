import torch
from ...util.vision_encoder_config import qwen2vl_vit_config


def build_vision_tower(vision_tower_cfg, **kwargs):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if getattr(vision_tower_cfg, "language", None) is None:
        vision_tower_cfg.language = "chinese" if "chinese" in vision_tower else "english"
    print(f"language: {vision_tower_cfg.language}, vision_tower: {vision_tower}")

    if "siglip-so400m-patch14-384" in vision_tower:
        from .siglip_encoder import SigLipVisionTower
        qwen2vl_vision_tower = Qwen2VisionTransformerPretrainedModel._from_config(qwen2vl_vit_config)
        qwen2vl_vision_tower.requires_grad_(False)
        return SigLipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs), qwen2vl_vision_tower
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
