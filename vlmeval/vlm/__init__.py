import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .glm import GLM4v, GLMThinking
from .llava import LLaVA_Next, LLaVA_OneVision, LLaVA_OneVision_HF
from .minicpm_v import MiniCPM_V_2_6, MiniCPM_o_2_6, MiniCPM_V_4, MiniCPM_V_4_5
from .moondream import Moondream1, Moondream2, Moondream3
from .qwen import Qwen2VLChat, Qwen2VLChatAguvis, Qwen3VLChat
from .intern import InternS1Chat, InternVLChat, XComposer2_4KHD, XComposer2d5
from .deepseek import DeepSeekVL2, DeepSeekOCR
from .gemma import PaliGemma, Gemma3
from .phi import Phi3Vision, Phi3_5Vision, Phi4Multimodal
from .cambrian import Cambrian, CambrianS
from .ovis import Ovis, Ovis1_6, Ovis1_6_Plus, Ovis2, OvisU1, Ovis2_5
from .pixtral import Pixtral
from .llama import llama_vision, llama4
from .molmo import molmo
from .smolvlm import SmolVLM, SmolVLM2
from .sail_vl import SailVL
from .keye_vlm import KeyeChat
from .nvidia import Cosmos, VILA, NVILA
