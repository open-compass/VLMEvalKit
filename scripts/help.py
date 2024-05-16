import sys
import os.path as osp
from vlmeval.smp import *


L1 = [
    ('MMVet', 'gpt-4-turbo_score.csv'), ('MMMU_DEV_VAL', 'acc.csv'),
    ('MathVista_MINI', 'gpt-4-turbo_score.csv'), ('HallusionBench', 'score.csv'),
    ('OCRBench', 'score.json'), ('AI2D_TEST', 'acc.csv'), ('MMStar', 'acc.csv'), 
    ('MMBench_V11', 'acc.csv'), ('MMBench_CN_V11', 'acc.csv')
]
L2 = [
    ('MME', 'score.csv'), ('LLaVABench', 'score.csv'), ('RealWorldQA', 'acc.csv'),
    ('MMBench', 'acc.csv'), ('MMBench_CN', 'acc.csv'), ('CCBench', 'acc.csv'),
    ('SEEDBench_IMG', 'acc.csv'), ('COCO_VAL', 'score.json'), ('POPE', 'score.csv'),
    ('ScienceQA_VAL', 'acc.csv'), ('ScienceQA_TEST', 'acc.csv'),
]
L3 = [
    ('OCRVQA_TESTCORE', 'acc.csv'), ('TextVQA_VAL', 'acc.csv'), 
    ('ChartQA_TEST', 'acc.csv'), ('DocVQA_VAL', 'acc.csv'), ('InfoVQA_VAL', 'acc.csv'),    
]

SKIP_MODELS = [
    'InternVL-Chat-V1-1', 'InternVL-Chat-V1-2', 'InternVL-Chat-V1-2-Plus', 
    'MiniGPT-4-v1-13B', 'instructblip_13b'
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

def LIST(lvl):
    data_list = {
        'l1': L1, 
        'l2': L1 + L2,
        'l3': L1 + L2 + L3,
    }[lvl]
    data = ' '.join([x[1] for x in data_list])
    print(data)

def REPORT(lvl):
    from vlmeval.config import supported_VLM
    logger = get_logger('REPORT')
    logger.info(colored('Essential Datasets: ', 'red'))
    models = list(supported_VLM)
    models = [m for m in models if m not in SKIP_MODELS and osp.exists(m)]

    data_list = {'l1': L1, 'l2': L2, 'l3': L3}[lvl]
    MISSING = []
    for f in models:
        for D, suff in data_list:
            if not completed(f, D, suff):
                logger.info(colored(f'Model {f} x Dataset {D} Not Found. ', 'red'))
                MISSING.append(f'--model {f} --data {D}')
    MISSING.append('')
    mwlines(MISSING, f'missing_{lvl}.txt')


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        LIST(sys.argv[2])
    elif sys.argv[1] == 'report':
        REPORT(sys.argv[2])
    