import sys
from vlmeval import *
FAIL_MSG = 'Failed to obtain answer via API.'

root = sys.argv[1]
if root[-1] in '/\\':
    root = root[:-1]

model_name = root.split('/')[-1]
datasets = list(dataset_URLs)

for d in datasets:
    fname = f'{model_name}_{d}.xlsx'
    pth = osp.join(root, fname)
    if osp.exists(pth):
        data = load(pth)
        # Detect Failure
        assert 'prediction' in data
        fail = [FAIL_MSG in x for x in data['prediction']]
        if sum(fail):
            nfail = sum(fail)
            ntot = len(fail)
            print(f'Model {model_name} x Dataset {d}: {nfail} out of {ntot} failed. {nfail / ntot * 100: .2f}%. ')
    