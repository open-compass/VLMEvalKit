import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .cogvlm import CogVlm, GLM4v
from .emu import Emu
from .idefics import IDEFICS, IDEFICS2
from .instructblip import InstructBLIP
from .llava import LLaVA, LLaVA_Next, LLaVA_XTuner, LLaVA_Next2, LLaVA_OneVision
from .minicpm_v import MiniCPM_V, MiniCPM_Llama3_V, MiniCPM_V_2_6
from .minigpt4 import MiniGPT4
from .mmalaya import MMAlaya
from .monkey import Monkey, MonkeyChat
from .mplug_owl2 import mPLUG_Owl2
from .omnilmm import OmniLMM12B
from .open_flamingo import OpenFlamingo
from .pandagpt import PandaGPT
from .qwen_vl import QwenVL, QwenVLChat
from .transcore_m import TransCoreM
from .visualglm import VisualGLM
from .xcomposer import ShareCaptioner, XComposer, XComposer2, XComposer2_4KHD, XComposer2d5
from .yi_vl import Yi_VL
from .internvl_chat import InternVLChat
from .deepseek_vl import DeepSeekVL
from .mgm import Mini_Gemini
from .bunnyllama3 import BunnyLLama3
from .vxverse import VXVERSE
from .paligemma import PaliGemma
from .qh_360vl import QH_360VL
from .phi3_vision import Phi3Vision
from .wemm import WeMM
from .cambrian import Cambrian
from .chameleon import Chameleon
from .video_llm.video_llava import VideoLLaVA, VideoLLaVA_HF
from .vila import VILA
from .ovis import Ovis
from .mantis import Mantis
from .mixsense import LLama3Mixsense
from .parrot import Parrot
from .omchat import OmChat
