from vlmeval.smp import *
from vlmeval.config import supported_VLM

def is_api(x):
    return getattr(supported_VLM[x].func, 'is_api', False)

datasets = ['OCRVQA_TESTCORE', 'TextVQA_VAL']
dataset_str = ' '.join(datasets)
models = list(supported_VLM)
models = [x for x in models if 'fs' not in x]
models = [x for x in models if not is_api(x)]
small_models = [x for x in models if '80b' not in x]
large_models = [x for x in models if '80b' in x]
models = small_models + large_models

for m in models:
    if '80b' in m:
        cmd = f'python run.py --data {dataset_str} --model {m}'
    else:
        cmd = f'bash run.sh --data {dataset_str} --model {m}'
    print(cmd)
    os.system(cmd)