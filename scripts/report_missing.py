from vlmeval.smp import *
import time
from datetime import datetime

dataset = ['MME', 'SEEDBench_IMG', 'MMBench', 'CCBench', 'MMBench_CN']
suffix = ['score.csv', 'acc.csv', 'acc.csv', 'acc.csv', 'acc.csv']
script = ['mme_eval.py', 'multiple_choice.py', 'multiple_choice.py', 'multiple_choice.py', 'multiple_choice.py']

N = len(dataset)
assert N == len(suffix) == len(script)

fs = ls(mode='dir')
for f in fs:
    files = ls(f, mode='file')
    for i in range(N):
        D = dataset[i]
        suff = suffix[i]
        scri = script[i]
        pred_file = f'{f}/{f}_{D}.xlsx'
        score_file = f'{f}/{f}_{D}_{suff}'
        if osp.exists(pred_file) and not osp.exists(score_file):
            print(f, D)