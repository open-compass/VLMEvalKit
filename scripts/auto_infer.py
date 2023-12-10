from vlmeval.smp import *
from vlmeval.config import supported_VLM

DATASET = 'MMVet'
models = list(supported_VLM)
models = [x for x in models if 'fs' not in x]

for m in models:
    if '80b' in m:
        cmd = f'python inference.py --data {DATASET} --model {m} --verbose'
    else:
        cmd = f'torchrun --nproc-per-node=8 inference.py --data {DATASET} --model {m} --verbose'
    print(cmd)
    os.system(cmd)