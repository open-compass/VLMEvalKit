import torch
import torch.nn as nn
from ...util.vision_encoder_config import siglip_config


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, cache_dir="./cache_dir"):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            from transformers import SiglipVisionConfig, SiglipVisionModel

            self.cfg_only = SiglipVisionConfig.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)  # dummy-load

    def load_model(self):
        from transformers import SiglipImageProcessor, SiglipVisionModel

        self.image_processor = SiglipImageProcessor.from_pretrained(self.image_tower_name)
        self.vision_tower = SiglipVisionModel._from_config(siglip_config)
        self.vision_tower.requires_grad_(False)
        self.image_processor.crop_size = self.image_processor.size["height"]

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        assert self.select_feature == "cls_patch"
        image_features = torch.cat([image_forward_outs[:, :1, :], image_forward_outs], dim=1)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    return_dict=True,
                )
                image_feature = self.feature_select(image_forward_out.last_hidden_state).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True,
            )
            image_features = self.feature_select(image_forward_outs.last_hidden_state).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
