import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .aria import Aria
from .base import BaseModel
from .cogvlm import CogVlm, GLM4v
from .emu import Emu
from .eagle_x import Eagle
from .idefics import IDEFICS, IDEFICS2
from .instructblip import InstructBLIP
from .kosmos import Kosmos2
from .llava import LLaVA, LLaVA_Next, LLaVA_XTuner, LLaVA_Next2, LLaVA_OneVision, LLaVA_OneVision_HF
from .minicpm_v import MiniCPM_V, MiniCPM_Llama3_V, MiniCPM_V_2_6
from .minigpt4 import MiniGPT4
from .mmalaya import MMAlaya, MMAlaya2
from .monkey import Monkey, MonkeyChat
from .moondream import Moondream1, Moondream2
from .minimonkey import MiniMonkey
from .mplug_owl2 import mPLUG_Owl2
from .omnilmm import OmniLMM12B
from .open_flamingo import OpenFlamingo
from .pandagpt import PandaGPT
from .qwen_vl import QwenVL, QwenVLChat
from .qwen2_vl import Qwen2VLChat
from .transcore_m import TransCoreM
from .visualglm import VisualGLM
from .xcomposer import ShareCaptioner, XComposer, XComposer2, XComposer2_4KHD, XComposer2d5
from .yi_vl import Yi_VL
from .internvl import InternVLChat
from .deepseek_vl import DeepSeekVL
from .janus import Janus
from .mgm import Mini_Gemini
from .bunnyllama3 import BunnyLLama3
from .vxverse import VXVERSE
from .paligemma import PaliGemma
from .qh_360vl import QH_360VL
from .phi3_vision import Phi3Vision, Phi3_5Vision
from .wemm import WeMM
from .cambrian import Cambrian
from .chameleon import Chameleon
from .video_llm import VideoLLaVA, VideoLLaVA_HF, Chatunivi, VideoChatGPT, LLaMAVID, VideoChat2_HD, PLLaVA
from .vila import VILA
from .ovis import Ovis, Ovis1_6, Ovis1_6_Plus
from .mantis import Mantis
from .mixsense import LLama3Mixsense
from .parrot import Parrot
from .omchat import OmChat
from .rbdash import RBDash
from .xgen_mm import XGenMM
from .slime import SliME
from .mplug_owl3 import mPLUG_Owl3
from .pixtral import Pixtral
from .llama_vision import llama_vision
from .molmo import molmo
from .points import POINTS, POINTSV15
from .nvlm import NVLM
from .vintern_chat import VinternChat
from .h2ovl_mississippi import H2OVLChat
from .falcon_vlm import Falcon2VLM
from .smolvlm import SmolVLM
