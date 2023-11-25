import torch
import sys
from abc import abstractproperty
import os.path as osp
from transformers import StoppingCriteriaList

class MiniGPT4:
    
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
        self.cfg = osp.join(root, cfg)
        sys.path.append(self.root)
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import StoppingCriteriaSub, CONV_VISION_Vicuna0, CONV_VISION_LLama2

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
        self.CONV_VISION = CONV_VISION_LLama2 if self.mode == 'v2' else CONV_VISION_Vicuna0

        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device) for ids in stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        
    def generate(self, image_path, prompt):
        from minigpt4.conversation.conversation import Chat
        if self.mode == 'v2':
            chat = Chat(self.model, self.vis_processor, device=self.device)
        else:
            chat = Chat(self.model, self.vis_processor, device=self.device, stopping_criteria=self.stopping_criteria)

        chat_state = self.CONV_VISION.copy()
        img_list = []
        _ = chat.upload_img(image_path, chat_state, img_list)
        chat.encode_img(img_list)
        chat.ask(prompt, chat_state)
        msg = chat.answer(conv=chat_state, img_list=img_list, temperature=0.01, 
                          max_new_tokens=500, max_length=2000)[0]
        return msg