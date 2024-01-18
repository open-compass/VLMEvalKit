from vlmeval.smp import *
from vlmeval.config import supported_VLM

logger = get_logger('Report Missing')

dataset = [
    'MME', 'SEEDBench_IMG', 'MMBench', 'CCBench', 'MMBench_CN',
    'MMVet', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'COCO_VAL', 'MMMU_DEV_VAL',
    'ChartQA_VALTEST_HUMAN', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MathVista_MINI', 'HallusionBench',
    'AI2D', 'LLaVABench'
]
suffix = [
    'score.csv', 'acc.csv', 'acc.csv', 'acc.csv', 'acc.csv',
    'gpt-4-turbo_score.csv', 'acc.csv', 'acc.csv', 'score.json', 'acc.csv',
    'acc.csv', 'acc.csv', 'acc.csv', 'gpt-4-turbo_score.csv', 'score.csv',
    'acc.csv', 'score.csv'
]

N = len(dataset)
assert N == len(suffix)
models = list(supported_VLM)

def missing(m, d, suf):
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
        continue
    files = ls(f, mode='file')
    for i in range(N):
        D = dataset[i]
        suff = suffix[i]
        if missing(f, D, suff):
            logger.info(colored(f'Model {f} x Dataset {D} Not Found. ', '#FF0000'))