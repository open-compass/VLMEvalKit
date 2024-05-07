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

        eval_files = ls(root, match=f'{model_name}_{d}_')
        eval_files = [x for x in eval_files if listinstr(['openai', 'gpt'], x) and x.endswith('.xlsx')]
        
        assert len(eval_files) == 1
        eval_file = eval_files[0]
        data = load(eval_file)
        
        if listinstr(['MathVista', 'MMVet'], d):
            bad = [x for x in data['log'] if 'All 5 retries failed.' in x]
            if len(bad):
                print(f'Model {model_name} x Dataset {d} Evaluation: {len(bad)} out of {len(data)} failed.')
        elif d == 'LLaVABench':
            sub = data[data['gpt4_score'] == -1]
            sub = sub[sub['gpt4_score'] == -1]
            if len(sub):
                print(f'Model {model_name} x Dataset {d} Evaluation: {len(sub)} out of {len(data)} failed.')
        else:
            bad = [x for x in data['log'] if FAIL_MSG in x]
            if len(bad):
                print(f'Model {model_name} x Dataset {d} Evaluation: {len(bad)} out of {len(data)} failed.')
                