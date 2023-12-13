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

def infer_data(model_name, dataset_name, out_file, verbose=False):
    res = {}
    if osp.exists(out_file):
        res = load(out_file)

    rank, world_size = get_rank_and_world_size()   
    if rank == 0:
        dataset = TSVDataset(dataset_name)
    if world_size > 1:
        dist.barrier()
    dataset = TSVDataset(dataset_name)
    lt = len(dataset)
    indices = list(range(rank, lt, world_size))
    lt = len(indices)
    data = dataset.data.iloc[indices]

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
    logger = get_logger('Inference')

    args = parse_args()
    assert len(args.data), "--data should be a list of data files"

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

    for _, model_name in enumerate(args.model):
        model = None
        os.makedirs(model_name, exist_ok=True)
        pred_root = model_name

        for i, dataset_name in enumerate(args.data):
            tmpl = f'{pred_root}/' + '{}' + f'{world_size}_{dataset_name}.pkl'
            out_file = tmpl.format(rank)
            result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'
            
            if model is None:
                model = model_name # which is only a name

            # CHECKER
            if dataset_name == 'CORE_MM':
                MULTI_IMG = getattr(supported_VLM[model_name].func, 'MULTI_IMG', False)
                if not MULTI_IMG:
                    logger.error(f'Model {model_name} does not support the `multi_generate` interface, which is required for testing CORE_MM, skip it. ')
                    continue

            if not osp.exists(result_file):
                model = infer_data(model, dataset_name=dataset_name, out_file=out_file, verbose=args.verbose)
                if world_size > 1:
                    dist.barrier()

                if rank == 0:

                    data_all = {}
                    for i in range(world_size):
                        data_all.update(load(tmpl.format(i)))

                    data = TSVDataset(dataset_name).data
                    assert len(data_all) == len(data)
                    data['prediction'] = [data_all[x] for x in data['index']]
                    data.pop('image')

                    if dataset_name == 'MME':
                        data = MME_postproc(data)

                    dump(data, result_file)   
                    for i in range(world_size):
                        os.remove(tmpl.format(i))
                         
            if rank == 0 and dataset_name not in ['MME', 'CORE_MM', 'MMVet']:
                time.sleep(3)
                res = prefetch_acc(result_file)
                print(model_name, res)
                dump(res, result_file.replace('.xlsx', '_prefetch.xlsx'))
            
            if rank == 0 and dataset_name == 'MME':
                time.sleep(3)
                res = MME_rating(result_file)
                print(model_name, res)
                dump(res, result_file.replace('.xlsx', '_prefetch.xlsx'))

if __name__ == '__main__':
    main()