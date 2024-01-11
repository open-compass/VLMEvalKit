from vlmeval.smp import *
from vlmeval.config import supported_VLM

logger = get_logger('Report Missing')

dataset = [
    'MME', 'SEEDBench_IMG', 'MMBench', 'CCBench', 'MMBench_CN',
    'MMVet', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'COCO_VAL', 'MMMU_DEV_VAL',
    'ChartQA_VALTEST_HUMAN', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MathVista_MINI', 'HallusionBench'
]
suffix = [
    'score.csv', 'acc.csv', 'acc.csv', 'acc.csv', 'acc.csv',
    'gpt-4-turbo_score.csv', 'acc.csv', 'acc.csv', 'score.json', 'acc.csv',
    'acc.csv', 'acc.csv', 'acc.csv', 'gpt-4-turbo_score.csv', 'score.csv'
]

N = len(dataset)
assert N == len(suffix)
models = list(supported_VLM)

for f in models:
    if not osp.exists(f):
        logger.info(f'{f} not evaluated. ')
        continue
    files = ls(f, mode='file')
    for i in range(N):
        D = dataset[i]
        suff = suffix[i]
        pred_file = f'{f}/{f}_{D}.xlsx'
        score_file = f'{f}/{f}_{D}_{suff}'
        if not osp.exists(score_file):
            logger.info(colored(f'Model {f} x Dataset {D}: Not Found. ', '#FF0000'))