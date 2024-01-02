import torch
import torch.distributed as dist
from vlmeval.smp import *
from vlmeval.evaluate import COCO_eval, YOrN_eval, MMVet_eval, multiple_choice_eval, VQAEval, MathVista_eval
from vlmeval.inference import infer_data_job, prefetch_acc
from vlmeval.config import supported_VLM
from vlmeval.utils import dataset_URLs, abbr2full

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--mode", type=str, default='all', choices=['all', 'infer'])
    parser.add_argument("--nproc", type=int, default=4, help="Parallel API calling")
    parser.add_argument("--ignore", action='store_true', help="Ignore failed indices. ")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--prefetch", action='store_true')
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
            if dataset_name not in dataset_URLs:
                dataset_name = abbr2full(dataset_name)
            
            if dataset_name not in dataset_URLs:
                logger.error(f'Unknown dataset: {dataset_name}. ')
                continue

            result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'
            
            if model is None:
                model = model_name # which is only a name

            # CHECKER
            if dataset_name == 'CORE_MM':
                MULTI_IMG = getattr(supported_VLM[model_name].func, 'multi_generate', None)
                if MULTI_IMG is not None:
                    logger.error(f'Model {model_name} does not support the `multi_generate` interface, which is required for testing CORE_MM, skip it. ')
                    continue
                if args.mode == 'all':
                    logger.error(f'Dataset {dataset_name} does not support `evaluation` now, will skip the evaluation. ')

            model = infer_data_job(model, model_name=model_name, dataset_name=dataset_name, verbose=args.verbose, api_nproc=args.nproc, ignore_failed=args.ignore)                     
            if rank == 0:
                time.sleep(3)
                res = None
                if listinstr(['SEEDBench_IMG', 'MMBench', 'CCBench', 'ScienceQA'], dataset_name):
                    res = prefetch_acc(result_file)
                else:
                    logger.warning(f'{dataset_name} is not handled by prefetch score calculator')
                if res is not None:
                    logger.info(f'{model_name} prefetching: ')
                    logger.info(res)
                    dump(res, result_file.replace('.xlsx', '_prefetch.xlsx'))
                
            if rank == 0 and args.mode == 'all':
                if listinstr(['MMBench', 'CCBench', 'SEEDBench_IMG', 'MMMU', 'ScienceQA'], dataset_name):
                    multiple_choice_eval(result_file, dataset=dataset_name, model='chatgpt-0613', nproc=args.nproc, verbose=args.verbose)
                elif listinstr(['MME', 'Hallusion'], dataset_name):
                    YOrN_eval(result_file, model='chatgpt-0613', nproc=args.nproc, verbose=args.verbose, dataset=dataset_name)
                elif dataset_name == 'MMVet':
                    MMVet_eval(result_file, model='gpt-4-turbo', nproc=args.nproc, verbose=args.verbose)
                elif listinstr(['COCO'], dataset_name):
                    COCO_eval(result_file)
                elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA'], dataset_name):
                    VQAEval(result_file)
                elif listinstr(['MathVista'], dataset_name):
                    MathVista_eval(result_file, model='gpt-4-turbo', nproc=args.nproc, verbose=args.verbose)
                else:
                    logger.error(f'Dataset {dataset_name} is not handled by evaluator, will be skipped. ')
            
if __name__ == '__main__':
    main()