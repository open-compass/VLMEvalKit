from vlmeval.api import *
from functools import partial

SEED20_MINI = {
    'Seed2.0-Mini-V1-MM1280-P95': partial(
        PSMAPI,
        psm=[
            'seed.ray.server_sz1bq3957d69832098.service.wlby',
        ],
        prob=[1],
        system_prompt='seed1.8-thinking',
        verbose=False,
        retry=3,
        temperature=1,
        top_p=0.95,
        timeout=3600,
        max_tokens=2**16,
        comment="rlvjoe8opr69820c7a",
        image_pixel_limit={'min_pixels': 1280 * 42 * 42, 'max_pixels': 1280 * 42 * 42},
    ), 
}

LITE_V2_PSM = [
    'seed.ray.server_it766ar0mu69a12005.service.wlby', 
    'seed.ray.server_cgd71llvdp69a12019.service.wlby'
]
LITE_MODELCARD = 'qi3zou9wxy698972a1'

SEED20_LITE = {
    'Seed2.0-Lite-PSM-xhigh': partial(
        PSMAPI,
        psm=LITE_V2_PSM,
        prob=[1],
        system_prompt='seed1.8-thinking',
        verbose=False,
        retry=3,
        temperature=1,
        top_p=0.95,
        timeout=3600,
        max_tokens=2**16,
        image_pixel_limit={'min_pixels': 1280 * 42 * 42, 'max_pixels': 5120 * 42 * 42},
        comment='qi3zou9wxy698972a1',
    ),
    'Seed2.0-Lite-PSM': partial(
        PSMAPI,
        psm=LITE_V2_PSM,
        prob=[1],
        system_prompt='seed1.8-thinking',
        verbose=False,
        retry=3,
        temperature=1,
        top_p=0.95,
        timeout=3600,
        max_tokens=2**16,
        image_pixel_limit={'min_pixels': 1280 * 42 * 42, 'max_pixels': 1280 * 42 * 42},
        comment='qi3zou9wxy698972a1',
    ),
}

SEED20_PRO = {}
seed20_pro_default = dict(
    system_prompt='seed1.8-thinking',
    verbose=False,
    retry=3,
    temperature=1, 
    top_p=0.95,
    timeout=7200, 
    max_tokens=128000,
    image_pixel_limit={'min_pixels': 1280 * 42 * 42, 'max_pixels': 1280 * 42 * 42})

SEED20_PRO['Seed2.0-Pro-Exp28-S35'] = partial(
    PSMAPI, 
    psm=[
        'seed.ray.server_5z0dk6ez0069a119e1.service.wlby',
    ], 
    **seed20_pro_default)


SEED20_GROUPS = [SEED20_MINI, SEED20_LITE, SEED20_PRO]
