import os
import sys

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up four levels to reach VLM-3R/
vlm_3r_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

def build_spatial_tower(spatial_tower_cfg, **kwargs):
    spatial_tower = getattr(spatial_tower_cfg, "mm_spatial_tower", getattr(spatial_tower_cfg, "spatial_tower", 'spann3r'))
    if spatial_tower == "cut3r":
        cut3r_path = os.path.join(vlm_3r_root, 'CUT3R')
        if cut3r_path not in sys.path:
            sys.path.append(cut3r_path)
        # Use relative import for the encoder wrapper/adapter file
        from .cut3r_spatial_encoder import Cut3rSpatialTower
        return Cut3rSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    raise ValueError(f"Unknown vision tower: {spatial_tower}")
