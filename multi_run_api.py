from concurrent.futures import ThreadPoolExecutor
from vlmeval.dataset import build_dataset
from tqdm import tqdm

import os
from vlmeval.smp import *


def parse_args():
    help_msg = "This script will call `run.py` to run the inference of multiple benchmarks in parallel"
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--parallel', type=int, default=4, help='Running the inference of N benchmarks in parallel. ')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dry', action='store_true')
    # Configuration for Resume
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', action='store_true')

    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')
    args = parse_args()
    assert len(args.model), '--model should be a list of models'
    assert len(args.data), '--data should be a list of data files'
    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    from vlmeval.config import DATASET_GROUPS
    from vlmeval.dataset import SUPPORTED_DATASETS
    data_all = []
    for k in args.data:
        if k in DATASET_GROUPS:
            data_all.extend(DATASET_GROUPS[k])
        elif k in SUPPORTED_DATASETS:
            data_all.append(k)
        else:
            logger.warning(f'Unknown dataset {k}, will be skipped. ')
    data_exist = [k for k in data_all if k in SUPPORTED_DATASETS]
    data_all = []
    for x in data_exist:
        if x not in data_all:
            data_all.append(x)

    logger.info(f'[Multi Run API] Will use the following datasets for eval: {data_all}')
    args.data = data_all

    avg_nproc = args.api_nproc // args.parallel
    command_pref = f"python run.py --mode infer --work-dir {args.work_dir} "  # noqa: E501

    if args.retry is not None:
        command_pref += f'--retry {args.retry} '
    if args.judge_args is not None:
        command_pref += f'--judge-args {args.judge_args} '
    if args.judge is not None:
        command_pref += f'--judge {args.judge} '
    if args.verbose:
        command_pref += '--verbose '
    if args.reuse:
        command_pref += '--reuse '
    command_pref += '--reuse-aux 1 ' if args.reuse_aux else '--reuse-aux 0 '
    command_pref += '--model {model_name} --data {dataset_name} --api-nproc {nproc}'

    commands_dict = {}

    data_first = os.environ.get('DATAFIRST', '') == 1
    if data_first:
        for d_name in args.data:
            for m_name in args.model:
                nproc = avg_nproc
                commands_dict[f'{m_name};{d_name}'] = command_pref.format(
                    model_name=m_name, dataset_name=d_name, nproc=nproc)
    else:
        for m_name in args.model:
            for d_name in args.data:
                nproc = avg_nproc
                commands_dict[f'{m_name};{d_name}'] = command_pref.format(
                    model_name=m_name, dataset_name=d_name, nproc=nproc)

    if args.dry:
        mwlines(list(commands_dict.values()), 'multi_run_cmds.txt')
        exit(0)

    eval_executor = ThreadPoolExecutor(max_workers=4)
    eval_nproc = 32
    bad_tuples = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {}
        futures_eval = {}
        for k in commands_dict:
            futures[k] = executor.submit(run_command, cmd=commands_dict[k])

        tot = len(commands_dict)
        pbar = tqdm(total=tot, desc='Benchmark Level Progress Bar')
        unfinished = list(commands_dict.keys())

        while len(unfinished):
            assert len(futures) or len(futures_eval), 'futures or futures_eval should not be empty'
            futures_to_remove = []
            for name in futures.keys():
                if futures[name].done():
                    model_name, dataset_name = name.split(';')
                    root = osp.join(args.work_dir, model_name)
                    preds = ls(root, level=2, match=f'{model_name}_{dataset_name}.tsv')
                    assert len(preds), 'inference job done but the pred_file does not exist'

                    eval_cmd = commands_dict[name].replace(' infer ', ' eval ')
                    if '--reuse' not in eval_cmd:
                        eval_cmd += ' --reuse'
                    eval_cmd = eval_cmd.split('--api-nproc')[0] + f' --api-nproc {eval_nproc}'
                    futures_eval[name] = eval_executor.submit(run_command, cmd=eval_cmd)
                    futures_to_remove.append(name)

            for name in futures_to_remove:
                futures.pop(name)

            futures_eval_to_remove = []
            for name in futures_eval.keys():
                if futures_eval[name].done():
                    unfinished.remove(name)
                    pbar.update(1)
                    model_name, dataset_name = name.split(';')
                    try:
                        dataset = build_dataset(dataset_name)
                        ret = dataset.report(model_name, dataset_name, root=args.work_dir, verbose=1)
                        logger.info(f'Result for {model_name} x {dataset_name}: {ret}')
                    except:
                        logger.warning(f'Failed to obtain result for {model_name} x {dataset_name}')
                        bad_tuples.append((model_name, dataset_name))
                    futures_eval_to_remove.append(name)
            for name in futures_eval_to_remove:
                futures_eval.pop(name)

            time.sleep(5)
        pbar.close()
    if len(bad_tuples):
        print(f'The following {len(bad_tuples)} tuples failed: {bad_tuples}')
        for tup in bad_tuples:
            print(f'Model Name: {tup[0]}, Dataset Name: {tup[1]}')
    else:
        print('All tuples are successfully evaluated. ')


if __name__ == '__main__':
    load_env()
    main()
