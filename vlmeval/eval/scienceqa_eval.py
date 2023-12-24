from vlmeval.smp import *
import pandas as pd
import random


def ScienceQA_eval(eval_file, nproc=4, verbose=False):
    logger = get_logger('Evaluation')

    data = load(eval_file)

    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    pred = []
    gt = []
    for _, line in enumerate(lines):
        gt.append(line['answer'])
        ref_answer = []
        for c in ['A', 'B', 'C', 'D', 'E']:
            if not pd.isna(line[c]):
                ref_answer.append(c)
        if line['prediction'] not in ref_answer:
            pred.append(random.choice(ref_answer))
        else:
            pred.append(line['prediction'])

    res = [pred[i] == gt[i] for i in range(len(pred))]
    acc = sum(res) / len(res)
    scienceqa_score_dict = {'IMG': acc}
    score_pth = eval_file.replace('.xlsx', '_score.json')
    dump(scienceqa_score_dict, score_pth)
    logger.info(
        f'ScienceQA_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info(f'Score: ')
    for key, value in scienceqa_score_dict.items():
        logger.info('{}:{}'.format(key, value))
