import torch
from PIL import Image
from abc import abstractproperty
import os.path as osp
import os 
from vlmeval.smp import *


class InstructBLIP:

    INSTALL_REQ = True

    def __init__(self, name):

        self.name_map = {
            'instructblip_7b': [
                'lmsys/vicuna-7b-v1.1'
            ],
            'instructblip_13b': [
                'lmsys/vicuna-13b-v1.1'
            ]
        }

        self.config_map = {
            'instructblip_7b': f'misc/blip2_instruct_vicuna7b.yaml', 
            'instructblip_13b': f'misc/blip2_instruct_vicuna13b.yaml', 
        }

        self.file_path = __file__
        config_root = osp.dirname(self.file_path)

        model_path = None
        assert name in self.name_map or osp.exists(name)

        if name in self.name_map:
            for pth in self.name_map[name]:
                if osp.exists(pth):
                    model_path = pth
                    break
                elif len(pth.split('/')) == 2:
                    model_path = pth
                    break
        else:
            model_path = name
        assert model_path is not None
            
        try:
            from lavis.models import load_preprocess
            from omegaconf import OmegaConf
            from lavis.common.registry import registry
        except:
            warnings.warn("Please install lavis before using InstructBLIP. ")
            exit(-1)

        assert name in self.config_map
        cfg_path = osp.join(config_root, self.config_map[name])
        cfg = OmegaConf.load(cfg_path)

        model_cfg = cfg.model
        OmegaConf.update(model_cfg, "llm_model", model_path)
        model_cls = registry.get_model_class(name="blip2_vicuna_instruct")
        model = model_cls.from_config(model_cfg)
        model.eval()

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        device = self.device
        model.to(device)
        self.model = model

        preprocess_cfg = cfg.preprocess
        vis_processors, _ = load_preprocess(preprocess_cfg)
        self.vis_processors = vis_processors

    def generate(self, image_path, prompt, dataset=None):
        vis_processors = self.vis_processors
        raw_image = Image.open(image_path).convert('RGB')
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        outputs = self.model.generate(dict(image=image_tensor, prompt=prompt))
        return outputs[0]