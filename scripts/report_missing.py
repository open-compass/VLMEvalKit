from vlmeval.smp import *
from vlmeval.config import supported_VLM

logger = get_logger('Report Missing')
SKIP_MODELS = [
    'InternVL-Chat-V1-1', 'InternVL-Chat-V1-2', 'InternVL-Chat-V1-2-Plus', 
    'MiniGPT-4-v1-13B', 'instructblip_13b'
]

ESSENTIAL = [
    ('MME', 'score.csv'), ('SEEDBench_IMG', 'acc.csv'), ('MMBench', 'acc.csv'), 
    ('CCBench', 'acc.csv'), ('MMBench_CN', 'acc.csv'), ('MMVet', 'gpt-4-turbo_score.csv'),
    ('MMMU_DEV_VAL', 'acc.csv'), ('MathVista_MINI', 'gpt-4-turbo_score.csv'), ('HallusionBench', 'score.csv'),
    ('AI2D_TEST', 'acc.csv'), ('LLaVABench', 'score.csv'), ('OCRBench', 'score.json'),
    ('MMStar', 'acc.csv'), ('RealWorldQA', 'acc.csv')
]
OPTIONAL = [
    ('OCRVQA_TESTCORE', 'acc.csv'), ('TextVQA_VAL', 'acc.csv'), ('ChartQA_VALTEST_HUMAN', 'acc.csv'), 
    ('COCO_VAL', 'score.json'), ('ScienceQA_VAL', 'acc.csv'), ('ScienceQA_TEST', 'acc.csv'),
]

models = list(supported_VLM)

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

for f in models:
    if not osp.exists(f):
        logger.info(f'{f} not evaluated. ')
models = [x for x in models if osp.exists(x) and x not in SKIP_MODELS]

logger.info(colored('Essential Datasets: ', 'red'))

ESS_MISSING = []
for f in models:
    files = ls(f, mode='file')
    for D, suff in ESSENTIAL:
        if not completed(f, D, suff):
            logger.info(colored(f'Model {f} x Dataset {D} Not Found. ', 'red'))
            ESS_MISSING.append(f'--model {f} --data {D}')
ESS_MISSING.append('')
mwlines(ESS_MISSING, 'missing_essential.txt')

logger.info(colored('Optional Datasets: ', 'magenta'))

OPT_MISSING = []
for f in models:
    files = ls(f, mode='file')
    for D, suff in OPTIONAL:
        if not completed(f, D, suff):
            logger.info(colored(f'Model {f} x Dataset {D} Not Found. ', 'magenta'))
            OPT_MISSING.append(f'--model {f} --data {D}')
OPT_MISSING.append('')
mwlines(OPT_MISSING, 'missing_optional.txt')