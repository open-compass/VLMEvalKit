import sys
from vlmeval.config import *
from vlmeval.smp import *

# Define valid modes
MODES = ('dlist', 'mlist', 'missing')

CLI_HELP_MSG = \
    f"""
    Arguments received: {str(['vlmutil'] + sys.argv[1:])}. vlmutil commands use the following syntax:

        vlmutil MODE MODE_ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode

    Some usages for xtuner commands: (See more by using -h for specific command!)

        1. List all the dataset by levels: l1, l2, l3, etc.:
            vlmutil dlist l1
        2. List all the models by categories: 4.33.0, 4.37.0, api, etc.:
            vlmutil mlist 4.33.0 [all/small/large]
        3. Report missing results:
            vlmutil report l1

    GitHub: https://github.com/open-compass/VLMEvalKit
    """  # noqa: E501


dataset_levels = {
    'l1': [
        ('MMVet', 'gpt-4-turbo_score.csv'), ('MMMU_DEV_VAL', 'acc.csv'),
        ('MathVista_MINI', 'gpt-4-turbo_score.csv'), ('HallusionBench', 'score.csv'),
        ('OCRBench', 'score.json'), ('AI2D_TEST', 'acc.csv'), ('MMStar', 'acc.csv'),
        ('MMBench_V11', 'acc.csv'), ('MMBench_CN_V11', 'acc.csv')
    ],
    'l2': [
        ('MME', 'score.csv'), ('LLaVABench', 'score.csv'), ('RealWorldQA', 'acc.csv'),
        ('MMBench', 'acc.csv'), ('MMBench_CN', 'acc.csv'), ('CCBench', 'acc.csv'),
        ('SEEDBench_IMG', 'acc.csv'), ('COCO_VAL', 'score.json'), ('POPE', 'score.csv'),
        ('ScienceQA_VAL', 'acc.csv'), ('ScienceQA_TEST', 'acc.csv'),
    ],
    'l3': [
        ('OCRVQA_TESTCORE', 'acc.csv'), ('TextVQA_VAL', 'acc.csv'),
        ('ChartQA_TEST', 'acc.csv'), ('DocVQA_VAL', 'acc.csv'), ('InfoVQA_VAL', 'acc.csv'),
    ]
}

dataset_levels['l12'] = dataset_levels['l1'] + dataset_levels['l2']
dataset_levels['l23'] = dataset_levels['l2'] + dataset_levels['l3']
dataset_levels['l123'] = dataset_levels['l12'] + dataset_levels['l3']

models = {
    '4.33.0': list(qwen_series) + list(internvl_series) + list(xcomposer_series) + [
        'mPLUG-Owl2', 'flamingov2', 'VisualGLM_6b', 'MMAlaya', 'PandaGPT_13B', 'VXVERSE'
    ] + list(idefics_series) + list(minigpt4_series) + list(instructblip_series),
    '4.37.0': [x for x in llava_series if 'next' not in x] + [
        'TransCore_M', 'cogvlm-chat', 'cogvlm-grounding-generalist', 'emu2_chat',
        'MiniCPM-V', 'MiniCPM-V-2', 'OmniLMM_12B', 'InternVL-Chat-V1-5'
    ] + list(xtuner_series) + list(yivl_series) + list(deepseekvl_series),
    '4.40.0': [
        'idefics2_8b', 'Bunny-llama3-8B', 'MiniCPM-Llama3-V-2_5', '360VL-70B', 'paligemma-3b-mix-448'
    ] + [x for x in llava_series if 'next' in x],
    'api': list(api_models)
}

SKIP_MODELS = [
    'InternVL-Chat-V1-1', 'InternVL-Chat-V1-2', 'InternVL-Chat-V1-2-Plus',
    'MiniGPT-4-v1-13B', 'instructblip_13b', 'MGM_7B', 'GPT4V_HIGH',
]

LARGE_MODELS = [
    'InternVL-Chat-V1-2', 'InternVL-Chat-V1-2-Plus', 'idefics_80b_instruct',
    '360VL-70B', 'emu2_chat'
]


def completed(m, d, suf):
    score_file = f'{m}/{m}_{d}_{suf}'
    if osp.exists(score_file):
        return True
    if d == 'MMBench':
        s1, s2 = f'{m}/{m}_MMBench_DEV_EN_{suf}', f'{m}/{m}_MMBench_TEST_EN_{suf}'
        return osp.exists(s1) and osp.exists(s2)
    elif d == 'MMBench_CN':
        s1, s2 = f'{m}/{m}_MMBench_DEV_CN_{suf}', f'{m}/{m}_MMBench_TEST_CN_{suf}'
        return osp.exists(s1) and osp.exists(s2)
    return False


def DLIST(lvl):
    lst = dataset_levels[lvl]
    return lst


def MLIST(lvl, size='all'):
    model_list = models[lvl]
    if size == 'small':
        model_list = [m for m in model_list if m not in LARGE_MODELS]
    elif size == 'large':
        model_list = [m for m in model_list if m in LARGE_MODELS]
    return model_list


def REPORT(lvl):
    from vlmeval.config import supported_VLM
    logger = get_logger('REPORT')
    logger.info(colored('Essential Datasets: ', 'red'))
    models = list(supported_VLM)
    models = [m for m in models if m not in SKIP_MODELS and osp.exists(m)]

    data_list = DLIST(lvl)
    MISSING = []
    for f in models:
        for D, suff in data_list:
            if not completed(f, D, suff):
                logger.info(colored(f'Model {f} x Dataset {D} Not Found. ', 'red'))
                MISSING.append(f'--model {f} --data {D}')
    MISSING.append('')
    mwlines(MISSING, f'missing_{lvl}.txt')


def cli():
    logger = get_logger('VLMEvalKit Tools')
    args = sys.argv[1:]
    if not args:  # no arguments passed
        logger.info(CLI_HELP_MSG)
        return
    if args[0].lower() in MODES:
        if args[0].lower() == 'dlist':
            assert len(args) >= 2
            lst = DLIST(args[1])
            print(' '.join(lst))
        elif args[0].lower() == 'mlist':
            assert len(args) >= 2
            size = 'all'
            if len(args) > 2:
                size = args[2].lower()
            lst = MLIST(args[1], size)
            print(' '.join(lst))
        elif args[0].lower() == 'report':
            assert len(args) >= 2
            REPORT(args[1])
    else:
        logger.error('WARNING: command error!')
        logger.info(CLI_HELP_MSG)
        return
