from vlmeval.smp import *
from vlmeval.dataset import SUPPORTED_DATASETS, build_dataset, build_judge
from vlmeval.config import DATASET_GROUPS

def get_reso_img(img):
    if osp.exists(img):
        im = Image.open(img)
        return im.size[0] * im.size[1]
    else:
        raise NotImplementedError

def get_reso_prompt(msgs):
    resos = []
    for msg in msgs:
        if msg['type'] == 'image':
            resos.append(get_reso_img(msg['value']))
    return resos

def reso_stats(dataset_name):
    dataset = build_dataset(dataset_name)
    func = dataset.build_prompt
    data = dataset.data
    prompts = track_progress_rich(func, [(row, ) for _, row in data.iterrows()], nproc=32)
    resos = track_progress_rich(get_reso_prompt, [(x, ) for x in prompts], nproc=32)
    resos_all = []
    for item in resos:
        resos_all.extend(item)
    dist = distribution(resos_all)
    dist['dataset'] = dataset_name
    print(dist)
    return dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    datasets = []
    for dataset in args.dataset:
        if dataset in DATASET_GROUPS:
            datasets.extend(DATASET_GROUPS[dataset])
        else:
            datasets.append(dataset)
    datasets = list(set(datasets))

    data_all = []
    for dataset in datasets:
        if 'arcagi' in dataset.lower():
            continue
        reso = reso_stats(dataset)
        data_all.append(reso)
        dump(pd.concat(data_all), 'reso_all.csv')
    print(f'The resolution statistics of {len(datasets)} datasets saved in reso_all.csv.')
