import torch


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if getattr(vision_tower_cfg, "language", None) is None:
        vision_tower_cfg.language = "chinese" if "chinese" in vision_tower else "english"
    print(f"language: {vision_tower_cfg.language}, vision_tower: {vision_tower}")

    if "siglip-so400m-patch14-384" in vision_tower:
        from .siglip_encoder import SigLipVisionTower
        from ..language_model.valley_qwen2vl import ValleyQwen2VLForCausalLM

        qwen2vl_vision_tower = ValleyQwen2VLForCausalLM.from_pretrained(
            vision_tower_cfg.eagle_vision_tower, attn_implementation="flash_attention_2"
        ).visual
        qwen2vl_vision_tower.requires_grad_(False)
        return SigLipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs), qwen2vl_vision_tower
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
