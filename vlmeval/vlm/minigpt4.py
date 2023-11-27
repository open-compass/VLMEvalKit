import torch
import sys
from abc import abstractproperty
import os.path as osp
from transformers import StoppingCriteriaList
from PIL import Image

class MiniGPT4:

    INSTALL_REQ = True
    
    def __init__(self, 
                 mode='v2', 
                 root='/mnt/petrelfs/share_data/duanhaodong/MiniGPT-4/', 
                 temperature=0.01, 
                 max_out_len=512):
        if mode == 'v2':
            cfg = 'minigptv2_eval.yaml'
        elif mode == 'v1_7b':
            cfg = 'minigpt4_7b_eval.yaml'
        elif mode == 'v1_13b':
            cfg = 'minigpt4_13b_eval.yaml'
        else:
            raise NotImplementedError
        
        self.mode = mode
        self.temperature = temperature 
        self.max_out_len = max_out_len
        self.root = root 
        this_dir = osp.dirname(__file__)

        self.cfg = osp.join(this_dir, 'misc', cfg)
        sys.path.append(self.root)
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry

        device = torch.cuda.current_device()
        self.device = device
        
        args = abstractproperty()
        args.cfg_path = self.cfg
        args.options = []
        cfg = Config(args)
        model_cfg = cfg.model_cfg
        model_cfg.device_8bit = device 
        model_cls = registry.get_model_class(model_cfg.arch)
        model = model_cls.from_config(model_cfg).to(device)
        model.eval()
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model = model
        self.vis_processor = vis_processor  

        
    def generate(self, image_path, prompt):
        raw_image = Image.open(image_path).convert('RGB')
        images = self.vis_processor(raw_image).unsqueeze(0)
        img_prompt = '###Human: <Img><ImageHere></Img> '
        prompt = img_prompt + ' ' + prompt
        texts = [prompt]
        outputs = self.model.generate(
            images,
            texts,
            num_beams=1,
            max_new_tokens=20,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1,
            length_penalty=1,
            temperature=1,
            do_sample=False,
            stop_words_ids=[2])
        return outputs[0]
