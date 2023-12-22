from vlmeval.smp import *
from vlmeval.config import supported_VLM

def is_api(x):
    return getattr(supported_VLM[x].func, 'is_api', False)

DATASET = 'OCRVQA_TESTCORE'
models = list(supported_VLM)
models = [x for x in models if 'fs' not in x]
models = [x for x in models if not is_api(x)]

for m in models:
    if '80b' in m:
        cmd = f'python run.py --data {DATASET} --model {m} --verbose'
    else:
        cmd = f'torchrun --nproc-per-node=8 run.py --data {DATASET} --model {m} --verbose'
    print(cmd)
    os.system(cmd)