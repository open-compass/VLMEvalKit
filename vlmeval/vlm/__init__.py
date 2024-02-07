import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .cogvlm import CogVlm
from .emu import Emu
from .idefics import IDEFICS
from .instructblip import InstructBLIP
from .llava import LLaVA
from .llava_xtuner import LLaVA_XTuner
from .minicpm_v import MiniCPM_V
from .minigpt4 import MiniGPT4
from .mmalaya import MMAlaya
from .monkey import Monkey, MonkeyChat
from .mplug_owl2 import mPLUG_Owl2
from .omnilmm import OmniLMM12B
from .open_flamingo import OpenFlamingo
from .pandagpt import PandaGPT
from .qwen_vl import QwenVL, QwenVLChat
from .sharedcaptioner import SharedCaptioner
from .transcore_m import TransCoreM
from .visualglm import VisualGLM
from .xcomposer import XComposer
from .xcomposer2 import XComposer2
from .yi_vl import Yi_VL
