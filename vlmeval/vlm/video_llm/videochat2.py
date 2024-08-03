import torch
import warnings
import copy as cp
import numpy as np
import sys
import os.path as osp
import os
import requests
import shutil
import huggingface_hub
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms import PILToTensor
from torchvision import transforms
from peft import get_peft_model, LoraConfig, TaskType
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE

class VideoChat2_HD(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True

    SYS = """
Carefully watch the video and pay attention to the cause and sequence of events, \
the detail and movement of objects, and the action and pose of persons. \
Based on your observations, select the best option that accurately addresses the question.\n
"""

    def __init__(self, model_path='OpenGVLab/VideoChat2_HD_stage4_Mistral_7B', root='./Ask-Anything', config_file='./configs/videochat2_hd.json', **kwargs):
        self.config_file = config_file
        self.root = root
        self.model_path = model_path

        if root is None:
            warnings.warn('Please set `root` to Ask-Anything directory, which is cloned from here: https://github.com/OpenGVLab/Ask-Anything')
            sys.exit(-1)

        sys.path.append(osp.join(root,'video_chat2'))
        try:
            from utils.config import Config
            from models import VideoChat2_it_hd_mistral
            from dataset.hd_utils import HD_transform_padding, HD_transform_no_padding
        except:
            raise ImportError(
                'Please first install VideoChat2 and set the root path to use VideoChat2, '
                'which is cloned from here: https://github.com/OpenGVLab/Ask-Anything '
            )

        cfg = Config.from_file(self.config_file)

        def download_file(url, pth):
            destination_folder = pth
            # 确保目标文件夹存在
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # 获取文件名
            filename = os.path.basename(url)
            destination_path = os.path.join(destination_folder, filename)
            if os.path.exists(destination_path):
                print(f'File downloaded! No repeat download needed. Saved in {destination_path}')
                return

            # 下载文件
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(destination_path, 'wb') as file:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, file)
                print(f"文件已下载并保存到 {destination_path}")
            else:
                print(f"下载失败，状态码: {response.status_code}")

        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        huggingface_hub.login(hf_token)
        videochat2_model_path = snapshot_download(repo_id=cfg.model.videochat2_model_path, repo_type='model')
        cfg.model.videochat2_model_path = osp.join(videochat2_model_path, 'videochat2_mistral_7b_stage2.pth')
        mistral_model_path = snapshot_download(repo_id=cfg.model.mistral_model_path, repo_type='model')
        cfg.model.mistral_model_path = mistral_model_path
        vit_blip_model_path = os.path.join(LMUDataRoot(), 'models')
        cfg.model.vit_blip_model_path = download_file(cfg.model.vit_blip_model_path, vit_blip_model_path)
        model = VideoChat2_it_hd_mistral(config=cfg.model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "lm_head"
            ]
        )
        model.mistral_model = get_peft_model(model.mistral_model, peft_config)
        stage4_model_path = snapshot_download(repo_id=model_path, repo_type='model')
        state_dict = torch.load(osp.join(stage4_model_path, 'videochat2_hd_mistral_7b_stage4.pth'), "cuda")

        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)

        model = model.to(torch.device('cuda'))
        model = model.eval()
        self.model = model

        #  position embedding
        self.nframe = 16
        self.resolution = 224
        self.hd_num = 6
        new_pos_emb = self.get_sinusoid_encoding_table(
            n_position=(self.resolution // 16) ** 2 * self.nframe,
            cur_frame=self.nframe
        )
        self.model.vision_encoder.encoder.pos_embed = new_pos_emb

        self.hd_transform = HD_transform_no_padding

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize(mean, std)
        ])

    def get_sinusoid_encoding_table(self, n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # generate checkpoint position embedding
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1
        sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

        print(f"n_position: {n_position}")
        print(f"pre_n_position: {pre_n_position}")

        if n_position != pre_n_position:
            T = ckpt_num_frame  # checkpoint frame
            P = 14  # checkpoint size
            C = d_hid
            new_P = int((n_position // cur_frame) ** 0.5) # testing size
            if new_P != 14:
                print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
                print(f'Interpolate the position embedding')
                sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
                sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
                sinusoid_table = torch.nn.functional.interpolate(
                    sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
                sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

        if cur_frame != ckpt_num_frame:
            print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
            print(f'Interpolate the position embedding')
            T = ckpt_num_frame  # checkpoint frame
            new_T = cur_frame  # testing frame
            # interpolate
            P = int((n_position // cur_frame) ** 0.5) # testing size
            C = d_hid
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
            sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
            sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

        return sinusoid_table

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.nframe
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.nframe)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)

        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        frames = self.hd_transform(frames.float(), image_size=self.resolution, hd_num=self.hd_num)
        torch_imgs = self.transform(frames)
        return torch_imgs

    def ask(self, text, conv):
        conv.messages.append([conv.roles[0], text])

    def infer_mme(
        self, data_sample, system="",
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        add_subtitle=False,
    ):
        assert system_q == False, "do not support system_q now"
        video = data_sample["video"]
        T_, C, H, W = video.shape
        video = video.reshape(1, T_, C, H, W).to("cuda:0")

        video_list = []
        with torch.no_grad():
            if system_q:
                raise NotImplementedError
            else:
                video_emb, _, _ = self.model.encode_img(video, system)
        video_list.append(video_emb[0])

        pred_list = []
        gt_list = []
        for idx, qa in enumerate(data_sample['qa_list']):
            print(f"----------qa_{idx}---------", flush=True)
            chat = EasyDict({
                "system": system,
                "roles": ("[INST]", "[/INST]"),
                "messages": [],
                "sep": ""
            })

            if add_subtitle:
                if data_sample['subtitle'] != '':
                    subtitle = f"This video's subtitles are listed below: {data_sample['subtitle']}"
                    chat.messages.append([chat.roles[0], f"{subtitle}\n<Video><VideoHere></Video> [/INST]"])
                else:
                    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])
            else:
                chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])

            if system_llm:
                prompt = system + qa[0] + question_prompt
            else:
                prompt = qa[0] + question_prompt

            self.ask(prompt, chat)

            llm_message = answer(
                conv=chat, model=self.model, do_sample=False,
                img_list=video_list, max_new_tokens=100,
                answer_prompt=answer_prompt, print_res=print_res
            )[0]
            # remove potential explanation
            llm_message = return_prompt + llm_message.strip().split('\n')[0]
            print(f"Pred: {llm_message}", flush=True)
            print(f"GT: {qa[1]}", flush=True)
            pred_list.append(llm_message[1])
            gt_list.append(qa[1][1])
        return pred_list, gt_list

    def generate_inner(self, message, dataset=None):
        question, video = self.message_to_promptvideo(message)
        imgs = self.read_video(video)

