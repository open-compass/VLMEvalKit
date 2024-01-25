import argparse
from vlmeval.smp import *
from vlmeval.config import supported_VLM

def is_api(x):
    return getattr(supported_VLM[x].func, 'is_api', False)

models = list(supported_VLM)
models = [x for x in models if 'fs' not in x]
models = [x for x in models if not is_api(x)]
small_models = [x for x in models if '80b' not in x]
large_models = [x for x in models if '80b' in x]
models = small_models + large_models

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, nargs='+', required=True)
args = parser.parse_args()

# Skip some models
models = [x for x in models if not listinstr(['MiniGPT', 'grounding-generalist'], x)]

for m in models:
    unknown_datasets = [x for x in args.data if not osp.exists(f'{m}/{m}_{x}.xlsx')]
    if len(unknown_datasets) == 0:
        continue
    dataset_str = ' '.join(unknown_datasets)
    if '80b' in m:
        cmd = f'python run.py --data {dataset_str} --model {m}'
    else:
        cmd = f'bash run.sh --data {dataset_str} --model {m}'
    print(cmd)
    os.system(cmd)