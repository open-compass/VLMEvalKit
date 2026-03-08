import os
import json
import click
import pandas as pd
from tabulate import tabulate
import os.path as osp
from vlmeval.smp import get_logger, ls


@click.command()
@click.argument('data_file', type=str)
@click.option('--judge', default=None, type=str)
@click.option('--api-nproc', default=32, type=int)
@click.option('--retry', default=3, type=int)
@click.option('--verbose', is_flag=True)
@click.option('--rerun', is_flag=True)
def EVAL(data_file, judge, api_nproc, retry, verbose, rerun):
    from vlmeval.dataset import build_dataset, SUPPORTED_DATASETS
    logger = get_logger('VLMEvalKit Tool-Eval')

    def extract_model_dataset(file_name):
        fname = osp.splitext(file_name)[0].split('/')[-1]
        parts = fname.split('_')
        for i in range(len(parts)):
            if '_'.join(parts[i:]) in SUPPORTED_DATASETS:
                return '_'.join(parts[:i]), '_'.join(parts[i:])
        return None, None
    model_name, dataset_name = extract_model_dataset(data_file)
    assert model_name is not None and dataset_name is not None, data_file
    dataset = build_dataset(dataset_name)
    prefix_dataset_names = [x for x in SUPPORTED_DATASETS if x.startswith(dataset_name) and x != dataset_name]
    # Set the judge kwargs first before evaluation or dumping
    judge_kwargs = {'nproc': api_nproc, 'verbose': verbose, 'retry': retry}
    if judge is None:
        if getattr(dataset, 'DEFAULT_JUDGE', None):
            judge_kwargs['model'] = dataset.DEFAULT_JUDGE
    else:
        judge_kwargs['model'] = judge
    judge_kwargs['nproc'] = api_nproc
    if rerun:
        assert data_file.endswith('.tsv')
        match_list = [dataset_name + '_'] + ['!' + x for x in prefix_dataset_names]
        files2remove = ls(osp.dirname(data_file), match=match_list)
        for name in files2remove:
            os.system(f'rm {name}')
    eval_results = dataset.evaluate(data_file, **judge_kwargs)
    if eval_results is not None:
        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
        logger.info('Evaluation Results:')
    if isinstance(eval_results, dict):
        logger.info('\n' + json.dumps(eval_results, indent=4))
    elif isinstance(eval_results, pd.DataFrame):
        logger.info('\n')
        logger.info(tabulate(eval_results.T) if len(eval_results) < len(eval_results.columns) else eval_results)
    return eval_results


@click.command()
@click.argument('model_name', type=str)
@click.argument('dataset_name', type=str)
@click.option('--root', default=None, type=str)
def PRINT_ACC(model_name, dataset_name, root=None):
    print_acc(model_name, dataset_name, root=root)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('other_args', nargs=-1, type=click.UNPROCESSED)
def RUN(other_args):
    import sys
    import vlmeval
    from vlmeval.smp import run_command
    run_script = osp.join(vlmeval.__path__[0], '../run.py')
    PYTHON = sys.executable
    run_command([PYTHON, run_script] + list(other_args))


def print_acc(model_name, dataset_name, root=None):
    from vlmeval.dataset import build_dataset, SUPPORTED_DATASETS
    logger = get_logger('VLMEvalKit Tool-Print Acc')
    assert model_name is not None and dataset_name is not None, (model_name, dataset_name)
    from vlmeval.config import DATASET_GROUPS
    import vlmeval
    if root is None:
        root = osp.join(vlmeval.__path__[0], '../outputs')

    datasets = None
    if dataset_name in DATASET_GROUPS:
        datasets = DATASET_GROUPS[dataset_name]
    elif ',' in dataset_name:
        datasets = dataset_name.split(',')

    if datasets is not None:
        for dataset in datasets:
            print_acc(model_name, dataset, root=root)
        exit(0)

    # assert dataset_name in SUPPORTED_DATASETS, dataset_name
    try:
        dataset = build_dataset(dataset_name)
        rating = dataset.report(model_name, dataset_name, root=root, verbose=True)
        logger.info('-' * 40 + dataset_name + '-' * 40)
        logger.info(json.dumps(rating, indent=4))
    except:
        root = osp.join(root, model_name)
        fs = ls(root, match=[f'{model_name}_{dataset_name}'])
        fs = [x for x in fs if x.endswith('.json') or x.endswith('.csv')]
        from vlmeval.smp import mrlines
        logger.info('-' * 40 + dataset_name + '-' * 40)
        for f in fs:
            lines = mrlines(f)
            logger.info('-' * 40 + f + '-' * 40)
            logger.info('\n'.join(lines))
    logger.info('\n\n')
