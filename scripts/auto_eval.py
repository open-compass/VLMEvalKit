from vlmeval.smp import *
import time
from datetime import datetime

dataset = ['MME', 'SEEDBench_IMG', 'MMBench', 'CCBench', 'MMBench_CN']
suffix = ['score.csv', 'acc.csv', 'acc.csv', 'acc.csv', 'acc.csv']
script = ['mme_eval.py', 'multiple_choice.py', 'multiple_choice.py', 'multiple_choice.py', 'multiple_choice.py']

N = len(dataset)
assert N == len(suffix) == len(script)

def now():
    return datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

cnt = 0
while True:
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
                cmd = f'python {scri} {pred_file} --verbose'
                if D != 'MME':
                    cmd += f' --dataset {D}'
                print(cmd)
                time.sleep(1)
                os.system(cmd)
                time.sleep(5)
    cnt += 1
    if cnt % 10 == 0:
        print('Looping', now())
    time.sleep(30)