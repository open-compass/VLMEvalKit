from vlmeval.ulm import *
from functools import partial
import os

BAGEL_CODE_ROOT = '/mnt/bn/ic-vlm/kenny/Codes/BAGEL'
OMNIGEN_CODE_ROOT = '/mnt/bn/ic-vlm/kenny/Codes/OmniGen2'

bagel_series = {
    'Bagel': partial(
        Bagel, 
        model_path='ByteDance-Seed/BAGEL-7B-MoT', 
        bagel_code_root=BAGEL_CODE_ROOT
    )
}

janus_series = {
    "Janus-1.3B": partial(JanusGeneration, model_path="deepseek-ai/Janus-1.3B"),
    "Janus-Pro-1B": partial(JanusPro, model_path="deepseek-ai/Janus-Pro-1B"),
    "Janus-Pro-7B": partial(JanusPro, model_path="deepseek-ai/Janus-Pro-7B"),
}

omnigen_series = {
    "OmniGen2": partial(
        OmniGen2, 
        model_path="OmniGen2/OmniGen2", 
        omnigen_code_root=OMNIGEN_CODE_ROOT
    ),
}


UG_OPENSOURCE_GROUPS = [bagel_series, janus_series, omnigen_series]
