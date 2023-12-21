from vlmeval.smp import *

def get_score(model, dataset):
    file_name = f'{model}/{model}_{dataset}'
    if listinstr(['CCBench', 'MMBench', 'SEEDBench_IMG'], dataset):
        file_name += '_acc.csv'
    elif listinstr(['MME'], dataset):
        file_name += '_score.csv'
    elif listinstr(['MMVet'], dataset):
        file_name += '_gpt-4-turbo_score.csv'
    elif listinstr(['COCO'], dataset):
        file_name += '_score.json'
    else:
        raise NotImplementedError
    
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
    elif 'MMBench' in dataset:
        assert dataset in ['MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN']
        ret[dataset] = data['Overall'][0] * 100
    elif 'MME' == dataset:
        ret[dataset] = data['perception'][0] + data['reasoning'][0]
    elif 'MMVet' == dataset:
        for n, a in zip(data['Category'], data['acc']):
            if n == 'Overall':
                ret[dataset] = a
    elif 'SEEDBench_IMG' == dataset:
        ret[dataset] = data['Overall'][0] * 100
    return ret

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    args = parser.parse_args()
    return args

def gen_table(models, datasets):
    res = defaultdict(dict)
    for m in models:
        for d in datasets:
            res[m].update(get_score(m, d))
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
    print(tabulate(final, headers=['Model'] + keys))
    
if __name__ == '__main__':
    args = parse_args()
    gen_table(args.model, args.data)