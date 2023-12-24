from vlmeval.smp import *
from vlmeval.config import supported_VLM
from termcolor import colored

logger = get_logger('Report Missing')

dataset = [
    'MME', 'SEEDBench_IMG', 'MMBench', 'CCBench', 'MMBench_CN', 
    'MMVet', 'OCRVQA_TESTCORE', 'TextVQA_VAL'
]
suffix = [
    'score.csv', 'acc.csv', 'acc.csv', 'acc.csv', 'acc.csv', 
    'gpt-4-turbo_score.csv', 'acc.csv', 'acc.csv'
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
        if osp.exists(pred_file) and not osp.exists(score_file):
            logger.info(colored(f'Model {f} x Dataset {D}: Not Found. ', 'red'))