import torch 
import torch.distributed as dist
import datetime
from vlmeval.config import supported_VLM
from vlmeval.utils import TSVDataset
from vlmeval.eval import MME_rating, MME_postproc
from vlmeval.smp import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

def infer_data(model_name, dataset, indices, out_file, verbose=False):
    res = {}
    if osp.exists(out_file):
        res = load(out_file)

    lt = len(indices)
    data = dataset.data.iloc[indices]
    dataset_name = dataset.dataset

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        return 

    if isinstance(model_name, str):
        model = supported_VLM[model_name]()
    else:
        model = model_name

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'build_prompt'):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        if dataset_name in ['CORE_MM']:
            assert hasattr(model, 'multi_generate')
            response = model.multi_generate(prompt=struct['text'], image_paths=struct['image'], dataset=dataset_name)
        else:
            response = model.generate(prompt=struct['text'], image_path=struct['image'], dataset=dataset_name)
        torch.cuda.empty_cache()
        
        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    dump(res, out_file)
    return model

def prefetch_acc(result_file):
    data = load(result_file)
    from vlmeval.eval.multiple_choice import build_choices, can_infer
    tot = defaultdict(lambda: 0)
    match = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        tot['Overall'] += 1
        tot[cate] += 1
        choices = build_choices(item)
        matched = can_infer(item['prediction'], choices)
        if matched:
            match['Overall'] += 1
            match[cate] += 1
            if matched == item['answer']:
                hit['Overall'] += 1
                hit[cate] += 1
    res = defaultdict(list)
    for k in tot.keys():
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['match'].append(match[k])
        res['hit'].append(hit[k])
        res['match_rate'].append(match[k] / tot[k] * 100)
        if match[k] == 0:
            res['acc'].append(0)
        else:
            res['acc'].append(hit[k] / match[k] * 100)
    res = pd.DataFrame(res)
    return res
        
def main():
    args = parse_args()
    assert len(args.data), "--data should be a list of data files"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

    for _, model_name in enumerate(args.model):
        model = None

        os.makedirs(model_name, exist_ok=True)
        pred_root = model_name

        for i, dataset_name in enumerate(args.data):
            if local_rank == 0:
                dataset = TSVDataset(dataset_name)
            if world_size > 1:
                dist.barrier()
            dataset = TSVDataset(dataset_name)
            tmpl = f'{pred_root}/' + '{}' + f'{world_size}_{dataset_name}.pkl'

            lt = len(dataset)
            indices = list(range(local_rank, lt, world_size))
            out_file = tmpl.format(local_rank)
            result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'
            
            if model is None:
                model = model_name # which is only a name

            # CHECKER
            if dataset_name == 'CORE_MM':
                MULTI_IMG = getattr(supported_VLM[model_name].func, 'MULTI_IMG', False)
                if not MULTI_IMG:
                    print(f'Model {model_name} does not support the `multi_generate` interface, which is required for testing CORE_MM, skip it. ')
                    continue

            if not osp.exists(result_file):
                model = infer_data(model, dataset=dataset, indices=indices, out_file=out_file, verbose=args.verbose)
                if world_size > 1:
                    dist.barrier()

                if local_rank == 0:
                    data_all = {}
                    for i in range(world_size):
                        data_all.update(load(tmpl.format(i)))

                    data = dataset.data
                    assert len(data_all) == len(data)
                    
                    data['prediction'] = [data_all[x] for x in data['index']]
                    data.pop('image')

                    if dataset_name == 'MME':
                        data = MME_postproc(data)

                    dump(data, result_file)   
                    for i in range(world_size):
                        os.remove(tmpl.format(i))
                         
            if local_rank == 0 and dataset_name not in ['MME', 'CORE_MM', 'MMVet']:
                time.sleep(3)
                res = prefetch_acc(result_file)
                print(model_name, res)
                dump(res, result_file.replace('.xlsx', '_prefetch.xlsx'))
            
            if local_rank == 0 and dataset_name == 'MME':
                time.sleep(3)
                res = MME_rating(result_file)
                print(model_name, res)
                dump(res, result_file.replace('.xlsx', '_prefetch.xlsx'))

if __name__ == '__main__':
    main()