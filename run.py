from vlmeval.smp import *
from vlmeval.config import supported_VLM
from vlmeval.dataset.VideoBench.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video


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


# Make WORLD_SIZE invisible when build models
def build_model_from_config(cfg, model_name, use_vllm=False):
    import vlmeval.api
    import vlmeval.vlm
    ws_bak = os.environ.pop('WORLD_SIZE', None)

    config = cp.deepcopy(cfg[model_name])
    if use_vllm:
        config['use_vllm'] = use_vllm
    if 'class' not in config:
        return supported_VLM[model_name](**config)
    cls_name = config.pop('class')
    if hasattr(vlmeval.api, cls_name):
        model = getattr(vlmeval.api, cls_name)(**config)
    elif hasattr(vlmeval.vlm, cls_name):
        model = getattr(vlmeval.vlm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')

    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak
    return model


def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')


def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `vlmeval/config.py` of check the output of the command \
        `vlmutil mlist all` in the terminal (you should first have vlmeval installed).
    To find all supported dataset names, please refer to the `vlmeval/dataset/__init__.py` file. The python script \
        to print all supported dataset names is as follows:
        ```python
        from vlmeval.dataset import SUPPORTED_DATASETS
        print(SUPPORTED_DATASETS)
        ```
        or you can check the output of the command `vlmutil dlist all` in the terminal.
    To find all supported video dataset default settings, please refer to the \
        `vlmeval/dataset/video_dataset_config.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "GPT4o_20240806_T00_HIGH": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0,
                "img_detail": "high"
            },
            "GPT4o_20240806_T10_Low": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 1.0,
                "img_detail": "low"
            },
            "GPT4o_20241120": {}
        },
        "data": {
            "MME-RealWorld-Lite": {
                "class": "MMERealWorld",
                "dataset": "MME-RealWorld-Lite"
            },
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMBench_Video_8frame_nopack": {},
            "Video-MME_16frame_subs": {
                "class": "VideoMME",
                "dataset": "Video-MME",
                "nframe": 16,
                "use_subtitle": true,
            }
        }
    }
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `vlmeval.vlm` or `vlmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `vlmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.
    - Tip: The defined dataset in the `supported_video_datasets` of `vlmeval/dataset/video_dataset_config.py` \
        can be used as a shortcut.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API VLMs, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    # Priority: --work-dir > .env config > default "./outputs"
    parser.add_argument('--work-dir', type=str, default=None, help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--timeout', type=int, default=None, help='timeout for API VLMs')
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
    parser.add_argument(
        '--use-vllm', action='store_true', help='use vllm to generate, the flag is only supported in Llama4 for now')
    parser.add_argument('--use-verifier', action='store_true', help='use verifier to evaluate')

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

    logger.info(f'Will use the following datasets for eval: {data_all}')
    args.data = data_all

    if RANK == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')
        from vlmeval.smp.log import judge_router_logging
        judge_router_logging()

    if args.work_dir is None:
        if 'MMEVAL_ROOT' in os.environ:
            args.work_dir = os.environ['MMEVAL_ROOT']
        else:
            args.work_dir = './outputs'

    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'timeout' in v.keywords and args.timeout is not None:
                v.keywords['timeout'] = args.timeout
                supported_VLM[k] = v

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

        if use_config:
            model = build_model_from_config(cfg['model'], model_name, args.use_vllm)

        for _, dataset_name in enumerate(args.data):
            if WORLD_SIZE > 1:
                dist.barrier()

            try:
                pred_format = get_pred_file_format()
                result_file_base = f'{model_name}_{dataset_name}.{pred_format}'

                if use_config:
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset_from_config(cfg['data'], dataset_name)
                        dist.barrier()
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                else:
                    dataset_kwargs = {}
                    if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                        from vlmeval.config import supported_APIs
                        if model_name in supported_APIs:
                            dataset_kwargs['model'] = 'DEFAULT'
                        else:
                            dataset_kwargs['model'] = model_name

                    # If distributed, first build the dataset on the main process for doing preparation works
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset(dataset_name, **dataset_kwargs)
                        dist.barrier()

                    dataset = build_dataset(dataset_name, **dataset_kwargs)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                assert dataset.TYPE not in ['T2I', 'TI2I', 'TI2TI'], \
                    f'Dataset {dataset_name} is for generation evaluation, use `run_gen.py` as the entry point. '

                # Handling Multi-Turn Dataset
                result_file = osp.join(pred_root, result_file_base)
                # Reuse the previous prediction file if exists
                if RANK == 0 and len(prev_pred_roots):
                    prepare_reuse_files(
                        pred_root_meta=pred_root_meta, eval_id=eval_id, model_name=model_name,
                        dataset_name=dataset_name, reuse=args.reuse, reuse_aux=args.reuse_aux
                    )

                if WORLD_SIZE > 1:
                    dist.barrier()

                if model is None:
                    model = model_name  # which is only a name

                if args.mode != "eval":
                    # Perform the Inference
                    if dataset.MODALITY == 'VIDEO':
                        model = infer_data_job_video(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            result_file_name=result_file_base,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            use_vllm=args.use_vllm)
                    else:
                        model = infer_data_job(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            use_vllm=args.use_vllm)

                if args.mode == 'infer':
                    continue

                # Set the judge kwargs first before evaluation or dumping
                judge_kwargs = {
                    'model': args.judge if args.judge is not None else getattr(dataset, 'DEFAULT_JUDGE', None),
                    'nproc': min(32, args.api_nproc),
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                if args.use_verifier:
                    judge_kwargs['use_verifier'] = True
                if args.use_vllm:
                    judge_kwargs['use_vllm'] = True

                if RANK == 0:
                    logger.info(judge_kwargs)

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Only RANK 0 handles the evaluation part
                if RANK == 0:
                    # Create the symbolic links for the prediction files
                    files = os.listdir(pred_root)
                    files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        rel_symlink(file_addr, link_addr)

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    # Perform the Evaluation
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

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
