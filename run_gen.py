import json
import os
import subprocess
import torch


# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK",1))

GPU_LIST = get_gpu_list()
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
    CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    print(
        f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},'
        f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}'
    )


from vlmeval.config import supported_ULM
from vlmeval.dataset import build_dataset
from vlmeval.inference_gen import infer_data_job
from vlmeval.smp import *


# Make WORLD_SIZE invisible when build models
def build_model_from_config(cfg, model_name):
    import vlmeval.api
    import vlmeval.ulm
    config = cp.deepcopy(cfg[model_name])
    if 'class' not in config:
        return supported_ULM[model_name](**config)
    cls_name = config.pop('class')
    if hasattr(vlmeval.api, cls_name):
        model = getattr(vlmeval.api, cls_name)(**config)
    elif hasattr(vlmeval.ulm, cls_name):
        model = getattr(vlmeval.ulm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.ulm`')
    assert getattr(model, 'SUPPORT_GEN', False), f'Model {cfg} does not support generation.'
    return model


def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        dataset = cls(**valid_params)
        assert getattr(dataset, 'SUPPORT_GEN', False), f'Dataset {cfg} does not support generation.'
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')


def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `vlmeval/config.py` file.
    To find all supported dataset names, please refer to the `vlmeval/dataset/__init__.py` file.
    Note that for Gen Eval, model.SUPPORT_GEN and dataset.SUPPORT_GEN must be True.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "Janus-Pro-1B": {
                "class": "JanusPro",
                "model": "Janus-Pro-1B",
                "model_path": "<PATH>/deepseek-ai/Janus-Pro-1B",
                "temperature": 0.5
            },
            "Janus-Pro-7B": {}
        },
        "data": {
            "DPGBench": {
                "class": "DPGBench",
                "dataset": "DPGBench"
            },
            "T2ICompBench_non_Spatial_VAL": {
                "class": "T2ICompBench",
                "dataset": "T2ICompBench_non_Spatial_VAL"
            }
        }
}
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `vlmeval.ulm` or `vlmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_ULM` of `vlmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `vlmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API ulms, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'])
    # API Kwargs, Apply to API ulms and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API ulms')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=int, default=True, help='reuse auxiliary evaluation files')
    parser.add_argument('--num-generations', type=int, default=None, help='number of generations for each prompt')
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')
    args = parse_args()
    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        assert len(args.data), '--data should be a list of data files'

    if RANK == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if not use_config:
        for k, v in supported_ULM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_ULM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_ULM[k] = v

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root_meta = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        ws_bak = os.environ.pop('WORLD_SIZE', None)
        if use_config:
            model = build_model_from_config(cfg['model'], model_name)
        else:
            model = supported_ULM[model_name]()
        assert model is not None, model_name
        if ws_bak:
            os.environ['WORLD_SIZE'] = ws_bak

        for _, dataset_name in enumerate(args.data):
            if WORLD_SIZE > 1:
                dist.barrier()

            try:
                if WORLD_SIZE > 1:
                    if RANK == 0:
                        dataset = build_dataset_from_config(cfg['data'], dataset_name) if use_config else build_dataset(dataset_name)  # noqa: E501
                    dist.barrier()
                dataset = build_dataset_from_config(cfg['data'], dataset_name) if use_config else build_dataset(dataset_name)  # noqa: E501
                if dataset is None:
                    logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                    continue

                assert dataset is not None and hasattr(dataset, 'PRED_FORMAT')
                result_file_base = dataset.PRED_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
                result_file = osp.join(pred_root, result_file_base)
                assert dataset.TYPE in model.EXPERTISE, f'Dataset {dataset_name} is not supported by model {model_name}'

                # Reuse the previous prediction file if exists
                if RANK == 0 and len(prev_pred_roots):
                    prepare_reuse_files(
                        pred_root_meta=pred_root_meta, eval_id=eval_id, model_name=model_name,
                        dataset_name=dataset_name, reuse=args.reuse, reuse_aux=args.reuse_aux
                    )

                if WORLD_SIZE > 1:
                    dist.barrier()

                num_generations = args.num_generations if args.num_generations is not None else -1
                # Set default num_generations by dataset names
                if num_generations == -1:
                    if dataset_name in ['GenEval', 'DPGBench']:
                        num_generations = 4
                    elif listinstr(['T2ICompBench'], dataset_name):
                        num_generations = 10
                    else:
                        num_generations = 1

                if args.mode != 'eval':
                    model = infer_data_job(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc,
                        num_generations=num_generations)
                else:
                    assert osp.exists(result_file), f'Prediction file {result_file} does not exist.'
                    print(f'Skip infer; load results from {result_file}')

                # Set the judge kwargs first before evaluation or dumping
                judge_kwargs = {
                    'nproc': args.api_nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                # What is the judge model? Here we explain the logic
                # 1. By default, use the `DEFAULT_JUDGE` in env, if not set, default to `gpt-4o-1120`
                # 2. Try to get `dataset.judge`, if set, will override 1
                # 3. Try to get `args.judge`, if set, will override 2
                judge_model = os.environ.get('DEFAULT_JUDGE', 'gpt-4o-1120')
                if hasattr(dataset, 'DEFAULT_JUDGE'):
                    judge_model = dataset.DEFAULT_JUDGE
                if args.judge is not None:
                    judge_model = args.judge

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                judge_kwargs['model'] = judge_model

                if RANK == 0:
                    logger.info(judge_kwargs)

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Skip the evaluation part if only infer
                if args.mode == 'infer':
                    continue

                # Setup the proxy for the evaluation
                eval_proxy = os.environ.get('EVAL_PROXY', None)
                old_proxy = os.environ.get('HTTP_PROXY', '')
                if eval_proxy is not None:
                    proxy_set(eval_proxy)

                if WORLD_SIZE > 1:
                    dist.barrier()

                if RANK == 0:
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                        logger.info('Evaluation Results:')
                        if isinstance(eval_results, dict):
                            logger.info('\n' + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            logger.info('\n' + tabulate(eval_results))

                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    # Create the symbolic links for the prediction files
                    files = os.listdir(pred_root)
                    files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        rel_symlink(file_addr, link_addr)

            except Exception as e:
                logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                                 'skipping this combination.')
                continue
        del model

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
