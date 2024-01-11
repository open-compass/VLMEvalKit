from vlmeval.smp import *
from vlmeval.utils.dataset_config import dataset_URLs

def get_score(model, dataset):
    file_name = f'{model}/{model}_{dataset}'
    if listinstr(['CCBench', 'MMBench', 'SEEDBench_IMG', 'MMMU', 'ScienceQA'], dataset):
        file_name += '_acc.csv'
    elif listinstr(['MME', 'Hallusion'], dataset):
        file_name += '_score.csv'
    elif listinstr(['MMVet', 'MathVista'], dataset):
        file_name += '_gpt-4-turbo_score.csv'
    elif listinstr(['COCO'], dataset):
        file_name += '_score.json'
    else:
        raise NotImplementedError
    
    if not osp.exists(file_name):
        return {}
    data = load(file_name)
    ret = {}
    if dataset == 'CCBench':
        ret[dataset] = data['Overall'][0] * 100
    elif dataset == 'MMBench':
        for n, a in zip(data['split'], data['Overall']):
            if n == 'dev':
                ret['MMBench_DEV_EN'] = a * 100
            elif n == 'test':
                ret['MMBench_TEST_EN'] = a * 100
    elif dataset == 'MMBench_CN':
        for n, a in zip(data['split'], data['Overall']):
            if n == 'dev':
                ret['MMBench_DEV_CN'] = a * 100
            elif n == 'test':
                ret['MMBench_TEST_CN'] = a * 100
    elif listinstr(['SEEDBench', 'ScienceQA', 'MMBench'], dataset):
        ret[dataset] = data['Overall'][0] * 100
    elif 'MME' == dataset:
        ret[dataset] = data['perception'][0] + data['reasoning'][0]
    elif 'MMVet' == dataset:
        data = data[data['Category'] == 'Overall']
        ret[dataset] = float(data.iloc[0]['acc'])
    elif 'HallusionBench' == dataset:
        data = data[data['split'] == 'Overall']
        for met in ['aAcc', 'qAcc', 'fAcc']:
            ret[dataset + f' ({met})'] = float(data.iloc[0][met])
    elif 'MMMU' in dataset:
        data = data[data['split'] == 'validation']
        ret['MMMU (val)'] = float(data.iloc[0]['Overall']) * 100
    elif 'MathVista' in dataset:
        data = data[data['Task&Skill'] == 'Overall']
        ret[dataset] = float(data.iloc[0]['acc'])
     
    return ret

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', default=[])
    parser.add_argument("--model", type=str, nargs='+', required=True)
    args = parser.parse_args()
    return args

def gen_table(models, datasets):
    res = defaultdict(dict)
    for m in models:
        for d in datasets:
            try:
                res[m].update(get_score(m, d))
            except:
                pass
    keys = []
    for m in models:
        for d in res[m]:
            keys.append(d)
    keys = list(set(keys))
    keys.sort()
    final = defaultdict(list)
    for m in models:
        final['Model'].append(m)
        for k in keys:
            if k in res[m]:
                final[k].append(res[m][k])
            else:
                final[k].append(None)
    final = pd.DataFrame(final)
    dump(final, 'summ.csv')
    if len(final) >= len(final.iloc[0].keys()):
        print(tabulate(final))
    else:
        print(tabulate(final.T))
    
if __name__ == '__main__':
    args = parse_args()
    if args.data == []:
        args.data = list(dataset_URLs)
    gen_table(args.model, args.data)