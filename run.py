import torch
import torch.distributed as dist
from vlmeval.smp import *
from vlmeval.eval import MME_eval, MMVet_eval, multiple_choice_eval, MME_rating, MME_postproc
from vlmeval.infer import infer_data, prefetch_acc
from vlmeval.utils import TSVDataset
from vlmeval.config import supported_VLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--mode", type=str, default='all', choices=['all', 'infer'])
    parser.add_argument("--nproc", type=str, default=4, help="Parallel API calling")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

def main():
    logger = get_logger('RUN')

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
                if args.mode == 'all':
                    logger.error(f'Dataset {dataset_name} does not support `evaluation` now, will skip the evaluation. ')

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
                         
            if rank == 0:
                time.sleep(3)
                res = None
                if dataset_name == 'MME':
                    res = MME_rating(result_file)
                elif dataset_name not in ['CORE_MM', 'MMVet']:
                    res = prefetch_acc(result_file)
                else:
                    logger.warning(f'{dataset_name} is not handled by prefetch score calculator')
                if res is not None:
                    logger.info(f'{model_name} prefetching: ')
                    logger.info(res)
                    dump(res, result_file.replace('.xlsx', '_prefetch.xlsx'))
                
            if rank == 0 and args.mode == 'all':
                if listinstr(['MMBench', 'CCBench', 'SEEDBench_IMG'], dataset_name):
                    multiple_choice_eval(result_file, dataset=dataset_name, model='chatgpt-0613', nproc=args.nproc, verbose=args.verbose)
                elif dataset_name == 'MME':
                    MME_eval(result_file, model='chatgpt-0613', nproc=args.nproc, verbose=args.verbose)
                elif dataset_name == 'MMVet':
                    MMVet_eval(result_file, model='gpt-4-turbo', nproc=args.nproc, verbose=args.verbose)
                else:
                    logger.error(f'Dataset {dataset_name} is not handled by evaluator, will be skipped. ')
            
if __name__ == '__main__':
    main()