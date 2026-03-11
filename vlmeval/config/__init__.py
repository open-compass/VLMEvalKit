from .vlm_api import VLM_API_GROUPS
from .vlm_opensource import VLM_OPENSOURCE_GROUPS
from .ug_opensource import UG_OPENSOURCE_GROUPS
from .dataset_groups import DATASET_GROUPS

supported_VLM = {}
supported_ULM = {}
supported_APIs = {}

for grp in VLM_API_GROUPS:
    supported_APIs.update(grp)

for grp in VLM_OPENSOURCE_GROUPS + VLM_API_GROUPS:
    supported_VLM.update(grp)

for grp in UG_OPENSOURCE_GROUPS:
    supported_ULM.update(grp)

def build_model(model_name, **kwargs):
    if model_name in supported_VLM:
        return supported_VLM[model_name](**kwargs)
    elif model_name in supported_ULM:
        return supported_ULM[model_name](**kwargs)
    else:
        raise NotImplementedError
