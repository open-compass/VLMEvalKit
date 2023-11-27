import sys
import torch
import os.path as osp


class PandaGPT:

    INSTALL_REQ = True

    def __init__(self, name, root='/mnt/petrelfs/share_data/duanhaodong/PandaGPT/'):
        assert name == 'PandaGPT_13B'
        self.name = name
        sys.path.append(osp.join(root, 'code'))
        try:
            from model.openllama import OpenLLAMAPEFTModel
        except:
            raise ImportError('Please first install PandaGPT and set the root path to use PandaGPT. ')
        self.args = {
            'model': 'openllama_peft',
            'imagebind_ckpt_path': osp.join(root, 'pretrained_ckpt/imagebind_ckpt'),
            'vicuna_ckpt_path': osp.join(root, 'pretrained_ckpt/vicuna_ckpt/13b_v0'),
            'delta_ckpt_path': osp.join(root, 'pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt'),
            'stage': 2,
            'max_tgt_len': 256,
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
        }
        model = OpenLLAMAPEFTModel(**self.args)
        delta_ckpt = torch.load(self.args['delta_ckpt_path'], map_location=torch.device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
        torch.cuda.empty_cache()
        self.model = model.eval().half().cuda()
        
    def generate(self, image_path, prompt, dataset=None):
        struct = {
            'prompt': prompt, 
            'image_paths': [image_path], 
            'audio_paths': [],
            'video_paths': [],
            'thermal_paths': [],
            'top_p': 0.9, 
            'temperature': 0.001, 
            'max_tgt_len': 256, 
            'modality_embeds': []
        }
        resp = self.model.generate(struct)
        return resp