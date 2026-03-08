from vlmeval import *
from vlmeval.dataset import build_dataset
from vlmeval.config import DATASET_GROUPS

date, commit_id = timestr('day'), githash(digits=8)
eval_id = f"T{date}_G{commit_id}"

def get_reso_img(img):
    if osp.exists(img):
        im = Image.open(img)
        return im.size[0] * im.size[1]
    else:
        return None

def get_reso_prompt(msgs):
    resos = []
    for msg in msgs:
        if msg['type'] == 'image':
            resos.append(get_reso_img(msg['value']))
    return resos

reso_cache = {}

def reso_map(dataset_name):
    if dataset_name in reso_cache:
        return reso_cache[dataset_name]
    
    dataset = build_dataset(dataset_name)
    func = dataset.build_prompt
    data = dataset.data
    prompts = track_progress_rich(func, [(row, ) for _, row in data.iterrows()], nproc=32)
    resos = track_progress_rich(get_reso_prompt, [(x, ) for x in prompts], nproc=32)
    large_res = [any([x is None or x > 1280 * 42 * 42 for x in reso]) for reso in resos]
    mapp = {x: y for x, y in zip(data['index'], large_res)}
    reso_cache[dataset_name] = mapp
    return mapp

def mkhigh(model_name, dataset_name, root):
    fname = f'{root}/{model_name}/{model_name}_{dataset_name}.tsv'
    data = load(fname)
    mapp = reso_map(dataset_name)
    data['prediction'] = [FAIL_MSG if mapp[idx] else pred for idx, pred in zip(data['index'], data['prediction'])]
    data['raw_prediction'] = [FAIL_MSG if mapp[idx] else pred for idx, pred in zip(data['index'], data['raw_prediction'])]
    data['thinking'] = [[] if mapp[idx] else pred for idx, pred in zip(data['index'], data['thinking'])]
    new_model_name = model_name.replace('MM1280', 'xhigh')
    new_dir = f'{root}/{new_model_name}/{eval_id}/'
    os.makedirs(new_dir, exist_ok=True)
    dump(data, f'{new_dir}/{new_model_name}_{dataset_name}.tsv')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Reuse Low Resolution Results for High Res Eval')
    parser.add_argument('--model', type=str, nargs='+')
    parser.add_argument('--data', type=str, nargs='+')
    parser.add_argument('--work-dir', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    models = args.model
    root = args.work_dir
    datasets = []
    for item in args.data:
        if item in DATASET_GROUPS:
            datasets.extend(DATASET_GROUPS[item])
        else:
            datasets.append(item)
    datasets = list(set(datasets))
    
    if root is None:
        import vlmeval
        pth = vlmeval.__path__[0]
        pth = osp.join(pth, '../outputs')
        root = pth
    for data in datasets:
        for model in models:
            mkhigh(model, data, root)
