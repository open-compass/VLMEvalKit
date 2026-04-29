import argparse
import asyncio
import copy as cp
import datetime
import json
import os
import shutil
import subprocess
import sys
from functools import partial
from numbers import Real
from pathlib import Path
from typing import List

import pandas as pd
from tabulate import tabulate


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
    except Exception:
        return []


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 1))

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


from vlmeval.api import LMDeployAPI
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.inference import infer_data_job
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.inference_video import infer_data_job_video
from vlmeval.smp import (MMBenchOfficialServer, build_eval_id, collect_run_benchmark_report,
                         get_eval_file_format, get_logger, get_pred_file_format,
                         get_pred_file_path, githash, is_prediction_complete, listinstr, load,
                         load_env, prepare_reuse_files, proxy_set, setup_logger, timestr,
                         upsert_dataset_status, upsert_run_status)
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer

logger = get_logger(__name__)


def _format_fail_rate(failed, total):
    if failed is None or total is None or total <= 0:
        return '-'
    return f'{failed / total * 100:.2f}% ({failed}/{total})'


def _format_sigfig(value):
    if value is None or isinstance(value, bool):
        return '-'
    if isinstance(value, Real):
        try:
            if pd.isna(value):
                return '-'
        except Exception:
            pass
        return f'{float(value):.4g}'
    return str(value)


def _format_metric_field(value):
    if value is None:
        return '-'
    if isinstance(value, Real) and not isinstance(value, bool):
        return _format_sigfig(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _iter_primary_metric_rows(row):
    primary_metric = row.get('primary_metric')
    primary_metric_value = row.get('primary_metric_value')

    if isinstance(primary_metric, (list, tuple)):
        if not primary_metric:
            return [(None, None)]
        value_map = primary_metric_value if isinstance(primary_metric_value, dict) else {}
        return [(metric_name, value_map.get(metric_name)) for metric_name in primary_metric]

    return [(primary_metric, primary_metric_value)]


def log_run_benchmark_report(run_dir):
    rows = collect_run_benchmark_report(run_dir)
    if not rows:
        logger.info(f'No benchmark summary rows found in {Path(run_dir) / "status.json"}')
        return

    report_rows = []
    for row in rows:
        metric_rows = _iter_primary_metric_rows(row)
        eval_error = row['eval_error'] or '-'
        if eval_error != '-' and len(str(eval_error)) > 120:
            eval_error = f'{str(eval_error)[:120]}...'

        for idx, (primary_metric, primary_metric_value) in enumerate(metric_rows):
            report_rows.append({
                'benchmark': row['benchmark'] if idx == 0 else '',
                'infer_fail_rate': _format_fail_rate(row['infer_failed'], row['infer_total']) if idx == 0 else '',
                'judge_fail_rate': _format_fail_rate(row['judge_failed'], row['judge_total']) if idx == 0 else '',
                'primary_metric': _format_metric_field(primary_metric),
                'primary_metric_value': _format_metric_field(primary_metric_value),
                'skip_reason': (row['skip_reason'] or '-') if idx == 0 else '',
                'eval_error': eval_error if idx == 0 else '',
            })

    logger.info('Run Summary Report:')
    logger.info('\n' + tabulate(report_rows, headers='keys'))


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
    import inspect

    import vlmeval.dataset
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


def build_model_from_base_url(args):
    """Build LMDeployAPI model kwargs from command-line arguments.

    Used by both local and API modes when --base-url is specified.
    Returns a dict suitable for LMDeployAPI(**kwargs) or partial(LMDeployAPI, **kwargs).
    """
    model_args = dict(
        model=args.model[0] if isinstance(args.model, list) else args.model,
        api_base=f"{args.base_url.rstrip('/')}/chat/completions",
        key=args.key,
        custom_prompt=args.custom_prompt,
        max_tokens=args.max_tokens,
        retry=args.retry,
        timeout=args.timeout,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        verbose=args.verbose,
        video_llm=args.video_llm,
        local_media=args.local_media,
    )
    model_args = {k: v for k, v in model_args.items() if v is not None}
    if args.thinker:
        logger.warning('[Deprecated] Use `--max-tokens` and `--timeout` directly.')
        model_args.update(dict(timeout=args.timeout * 2, max_tokens=args.max_tokens * 2))
    if args.extra_body:
        try:
            extra = json.loads(args.extra_body)
        except Exception as e:
            raise ValueError(f'Unable to parse the --extra-body value `{args.extra_body}`') from e
        assert isinstance(extra, dict), '--extra-body must be a valid Python dict'
        model_args.update(extra)
    return model_args


def get_judge_kwargs(dataset_name, dataset_type, args):
    """Determine judge kwargs based on dataset name and type.

    Uses run.py's logic as the canonical source for dataset-specific judge model
    assignments, with additional entries from run_api.py (Video-MME).
    Supports both local and API modes with mode-specific fallbacks.
    """
    # Determine nproc with mode-specific fallback
    if args.judge_api_nproc is not None:
        nproc = args.judge_api_nproc
    else:
        nproc = args.api_nproc  # local mode fallback

    # Determine retry with mode-specific fallback
    if args.judge_retry is not None:
        retry = args.judge_retry
    else:
        retry = args.retry

    judge_kwargs = {
        'nproc': nproc,
        'verbose': args.verbose,
        'retry': retry,
        'timeout': args.judge_timeout,
        **(json.loads(args.judge_args) if args.judge_args else {}),
    }

    if args.judge_base_url:
        judge_kwargs['api_base'] = f"{args.judge_base_url.rstrip('/')}/chat/completions"
    if args.judge_key:
        judge_kwargs['key'] = args.judge_key

    if args.judge is not None:
        judge_kwargs['model'] = args.judge
    else:
        if dataset_type in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
            ['moviechat1k', 'mme-reasoning'], dataset_name.lower()
        ):
            if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
                judge_kwargs['model'] = 'gpt-4o-mini'
            elif listinstr(['VisualPuzzles'], dataset_name):
                judge_kwargs['model'] = 'exact_matching'
            elif listinstr(['PuzzleVQA'], dataset_name):
                judge_kwargs['model'] = 'exact_matching'
            elif listinstr(['VisuLogic'], dataset_name):
                judge_kwargs['model'] = 'exact_matching'
            else:
                judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
            if listinstr(['LLaVABench_KO'], dataset_name):
                judge_kwargs['model'] = 'gpt-4o-0806'
            else:
                judge_kwargs['model'] = 'gpt-4-turbo'
        elif listinstr(['VGRPBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(
            ['MathVista', 'MathVerse', 'MathVision', 'LENS', 'DynaMath', 'VL-RewardBench',
             'LogicVista', 'MOAT', 'OCR_Reasoning', 'VTCBench', 'Asclepius',
             'MMSafetyBench', 'MSSBench', 'SIUO', 'SIUO_GEN', 'XSTest', 'Flames'], dataset_name
        ):
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['OlympiadBench'], dataset_name):
            use_api_judger = judge_kwargs.get("olympiad_use_api_judger", False)
            if use_api_judger:
                judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(
            ['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench',
             'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name
        ):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['ChartMimic'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['VDC'], dataset_name):
            judge_kwargs['model'] = 'llama31-8b'
        elif listinstr(['Video_MMLU_QA', 'Video_MMLU_CAP'], dataset_name):
            judge_kwargs['model'] = 'qwen-72b'
        elif listinstr(['MMVMBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['CVQA_EN', 'CVQA_LOC'], dataset_name):
            judge_kwargs['model'] = 'gpt-4.1'
        elif listinstr(['M4Bench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['AyaVisionBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4.1'
        elif listinstr(['MathCanvas'], dataset_name):
            judge_kwargs['model'] = 'gpt-4.1-2025-04-14'
        elif listinstr(['MMReason'], dataset_name):
            judge_kwargs['model'] = 'gpt-4.1'
        elif listinstr(['CoreCognition'], dataset_name):
            judge_kwargs['model'] = 'gpt-4.1'
        elif listinstr(['WorldVQA'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-1120'
        elif listinstr(['Video-MME'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['MaCBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['SciDocBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-mini'

    if args.use_verifier:
        judge_kwargs['use_verifier'] = True
    if args.use_vllm:
        judge_kwargs['use_vllm'] = True

    return judge_kwargs


def parse_reuse_aux_arg(reuse_aux):
    if isinstance(reuse_aux, bool):
        return 'all' if reuse_aux else 'none'
    if isinstance(reuse_aux, int):
        return 'all' if reuse_aux else 'none'
    if isinstance(reuse_aux, str):
        value = reuse_aux.strip().lower()
        if value in ['all', 'infer', 'none']:
            return value
        if value in ['1', 'true', 'yes']:
            return 'all'
        if value in ['0', 'false', 'no']:
            return 'none'
    raise argparse.ArgumentTypeError('reuse_aux must be one of: all, infer, none')


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

--api-mode:
    Switch to the async API pipeline mode (originally run_api.py). This mode uses an optimized pipeline
    for API-based models with cross-dataset unified inference queue, parallel inference and evaluation,
    and better remote model utilization.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)

    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')

    # Work Dir & Mode
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'])

    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=32, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=6, help='retry numbers for API VLMs')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--keep-failed', action='store_true',
                        help='Keep failed predictions as-is instead of retrying them.')
    parser.add_argument(
        '--ignore',
        action='store_true',
        help='[Deprecated] Ignore failed indices, it is the default behavior now. '
        'Use `--keep-failed` to disable it.')
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument(
        '--reuse-aux',
        type=parse_reuse_aux_arg,
        default='all',
        help='Reuse auxiliary files: `all` for infer+eval aux, `infer` for inference-only aux, `none` for no aux.'
    )
    parser.add_argument(
        '--use-vllm', action='store_true', help='use vllm to generate, the flag is only supported in Llama4 for now')
    parser.add_argument('--use-verifier', action='store_true', help='use verifier to evaluate')

    # Judge Args
    parser.add_argument('--judge', type=str, default=None)
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    parser.add_argument('--judge-base-url', type=str, default=None, help='Base URL of judge API')
    parser.add_argument('--judge-key', type=str, default=None, help='API key for judge model')
    parser.add_argument('--judge-api-nproc', type=int, default=None,
                        help='Parallel API calling for judger (defaults to follow --api-nproc)')
    parser.add_argument('--judge-retry', type=int, default=None,
                        help='Retry times for failed judgement (defaults to follow --retry)')
    parser.add_argument('--judge-timeout', type=int, default=600,
                        help='Max time in seconds for judgement.')

    # Inference Model Args (when --base-url is specified)
    parser.add_argument('--base-url', type=str, default=None,
                        help='Base URL of OpenAI-compatible API (e.g. http://localhost:8080/v1). '
                             'If set, LMDeployAPI is used for inference without modifying config.py.')
    parser.add_argument('--key', type=str, default='sk-admin', help='API key for inference model')
    parser.add_argument('--thinker', action='store_true',
                        help='[Deprecated] Enable thinking mode: doubles timeout and max_tokens.')
    parser.add_argument('--max-tokens', type=int, default=2 ** 15,
                        help='Max tokens for model generation.')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--repetition-penalty', type=float, default=None)
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Max time in seconds for a single inference request.')
    parser.add_argument('--custom-prompt', type=str, default=None,
                        help='Manually select a model adapter by name.')
    parser.add_argument('--extra-body', type=str, default=None,
                        help='Extra inference parameters as json dict string')
    parser.add_argument('--video-llm', action='store_true',
                        help='Whether the API model supports native video inputs.')
    parser.add_argument('--local-media', action='store_true',
                        help='Whether to send local media file path to the API model.')

    # API Pipeline Args (only for --api-mode)
    parser.add_argument('--api-mode', action='store_true',
                        help='Switch to async API pipeline mode')
    parser.add_argument('--monitor-interval', type=int, default=30,
                        help='Status monitoring interval in seconds')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: run evaluation in main process')

    args = parser.parse_args()
    if args.ignore:
        logger.warning('[Deprecated] the `--ignore` flag is deprecated since it is '
                       'the default behavior, use `--keep-failed` to disable it.')
    return args


def run_local_mode(args):
    """Original evaluation mode with GPU/distributed support."""
    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        assert len(args.data), '--data should be a list of data files'

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    commit_id = githash(digits=8)
    eval_id = build_eval_id()
    setup_logger(log_file=os.path.join(args.work_dir, 'logs', f'{eval_id}_{timestr()}.log'))

    if args.mode == 'eval':
        args.reuse = True
        logger.info('Force to use `reuse=True` for eval mode.')

    if RANK == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, this run will start from a fresh output directory')
        else:
            logger.info(f'--reuse is set, reuse-aux={args.reuse_aux}')

    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

        # If FWD_API is set, will use class `GPT4V` for all API models in the config
        if os.environ.get('FWD_API', None) == '1':
            from vlmeval.api import GPT4V
            from vlmeval.config import api_models as supported_APIs
            for m in args.model:
                if m in supported_APIs:
                    kws = supported_VLM[m].keywords
                    supported_VLM[m] = partial(GPT4V, **kws)
                    logger.warning(f'FWD_API is set, will use class `GPT4V` for {m}')

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for _, model_name in enumerate(args.model):
        logger.info(f'=========== {model_name} ===========')
        model = None

        pred_root_meta = Path(args.work_dir) / model_name
        pred_root = pred_root_meta / eval_id
        pred_root_meta.mkdir(parents=True, exist_ok=True)
        pred_root.mkdir(parents=True, exist_ok=True)

        if RANK == 0:
            upsert_run_status(
                pred_root,
                eval_id=eval_id,
                created_at=datetime.datetime.now().astimezone().isoformat(),
                commit=commit_id,
                argv=sys.argv,
                api_mode=False,
                world_size=WORLD_SIZE,
                pred_format=get_pred_file_format(),
                eval_format=get_eval_file_format(),
                mode=args.mode,
                reuse=bool(args.reuse),
                reuse_aux=args.reuse_aux,
                model_name=model_name,
            )

        if use_config:
            model = build_model_from_config(cfg['model'], model_name, args.use_vllm)
        elif args.base_url:
            model_args = build_model_from_base_url(args)
            model_args['model'] = model_name
            model = LMDeployAPI(**model_args)

        for _, dataset_name in enumerate(args.data):
            logger.info(f'----------- {dataset_name} -----------')
            if WORLD_SIZE > 1:
                dist.barrier()

            dataset = None
            result_file = None
            judge_model = None

            try:
                result_file = get_pred_file_path(
                    str(pred_root), model_name, dataset_name, use_env_format=True)
                if RANK == 0:
                    upsert_dataset_status(
                        run_dir=pred_root,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        prediction_file=result_file,
                        status='pending',
                    )

                if use_config:
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset_from_config(cfg['data'], dataset_name)
                        dist.barrier()
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        if RANK == 0:
                            upsert_dataset_status(
                                run_dir=pred_root,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                status='done',
                                skip_reason='invalid_dataset',
                            )
                        continue
                else:
                    dataset_kwargs = {}
                    if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                        dataset_kwargs['model'] = model_name

                    # If distributed, first build the dataset on the main process for doing preparation works
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset(dataset_name, **dataset_kwargs)
                        dist.barrier()

                    dataset = build_dataset(dataset_name, **dataset_kwargs)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        if RANK == 0:
                            upsert_dataset_status(
                                run_dir=pred_root,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                status='done',
                                skip_reason='invalid_dataset',
                            )
                        continue

                judge_kwargs = get_judge_kwargs(dataset_name, dataset.TYPE, args)
                judge_model = judge_kwargs.get('model', '')

                if RANK == 0:
                    reuse_ctx = prepare_reuse_files(
                        pred_root_meta=pred_root_meta,
                        eval_id=eval_id,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        dataset=dataset,
                        result_file=result_file,
                        reuse=args.reuse,
                        reuse_aux=args.reuse_aux,
                        retry_failed=not args.keep_failed,
                        judge_model=judge_model if args.mode != 'infer' else None,
                        world_size=WORLD_SIZE,
                    )
                    upsert_dataset_status(
                        run_dir=pred_root,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        source_run=reuse_ctx['source_eval_id'],
                        judge_model=judge_model,
                        reuse_aux=args.reuse_aux,
                    )
                    logger.info(judge_kwargs)

                if WORLD_SIZE > 1:
                    dist.barrier()

                prediction_complete = is_prediction_complete(
                    result_file,
                    dataset_indices=list(dataset.data['index']),
                    retry_failed=not args.keep_failed,
                )
                if args.mode == 'eval' and not prediction_complete:
                    if RANK == 0:
                        logger.error(
                            f'No reusable completed prediction found for {model_name} x {dataset_name}, '
                            'skipping this combination in eval mode.'
                        )
                        if Path(result_file).exists():
                            skip_reason = 'Incomplete infer result'
                        else:
                            skip_reason = 'No infer result found'
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason=skip_reason,
                        )
                    continue

                if model is None:
                    model = model_name  # which is only a name

                if args.mode != "eval":
                    if RANK == 0:
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='infer',
                        )
                    # Perform the Inference
                    if dataset.MODALITY == 'VIDEO':
                        model = infer_data_job_video(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            result_file=result_file,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            use_vllm=args.use_vllm,
                            retry_failed=not args.keep_failed)
                    elif dataset.TYPE == 'MT':
                        model = infer_data_job_mt(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            retry_failed=not args.keep_failed,
                            use_vllm=args.use_vllm)
                    else:
                        model = infer_data_job(
                            model,
                            work_dir=pred_root,
                            model_name=model_name,
                            dataset=dataset,
                            verbose=args.verbose,
                            api_nproc=args.api_nproc,
                            retry_failed=not args.keep_failed,
                            use_vllm=args.use_vllm)

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Only RANK 0 handles the evaluation part
                if RANK == 0:
                    if args.mode != 'infer':
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='eval',
                        )
                    # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
                    if dataset_name in ['MMMU_TEST']:
                        result_json = MMMU_result_transfer(result_file)
                        logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                    f'json file saved in {result_json}')
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason='official_submission_only_mmmu_test',
                        )
                        continue
                    elif 'MMT-Bench_ALL' in dataset_name:
                        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                        logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                    f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                    f'submission file saved in {submission_file}')
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason='official_submission_only_mmt_bench',
                        )
                        continue

                    # Skip the evaluation part if only infer
                    if args.mode == 'infer':
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason='mode_infer',
                        )
                        continue

                    # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
                    if 'MLLMGuard_DS' in dataset_name:
                        logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason='evaluation_not_supported_for_dataset',
                        )
                        continue
                    elif 'AesBench_TEST' == dataset_name:
                        logger.info(f'The results are saved in {result_file}. '
                                    f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason='external_submission_required',
                        )
                        continue
                    elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                        logger.info(f'{dataset_name} is a test split without ground-truth. '
                                    'Thus only the inference part is supported for those datasets. ')
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason='test_split_without_ground_truth',
                        )
                        continue
                    elif dataset_name in [
                        'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                        'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
                    ] and not MMBenchOfficialServer(dataset_name):
                        logger.error(
                            f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason='mmbench_evaluation_requires_official_server',
                        )
                        continue

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    # Perform the Evaluation
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        summary_eval_results = eval_results
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                        logger.info('Evaluation Results:')
                        if isinstance(eval_results, dict):
                            logger.info('\n' + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            logger.info('\n' + tabulate(eval_results))
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            metrics_source=summary_eval_results,
                            dataset_obj=dataset,
                        )
                    else:
                        upsert_dataset_status(
                            run_dir=pred_root,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            status='done',
                            skip_reason='evaluate_returned_none',
                        )

                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    # Create the symbolic links for the prediction files
                    files = [
                        path for path in pred_root.iterdir()
                        if path.is_file() and (
                            f'{model_name}_{dataset_name}' in path.name or path.name == 'status.json'
                        )
                    ]
                    # Exclude temporary intermediate files
                    files = [
                        path for path in files
                        if not path.name.endswith(('_checkpoint.pkl', '_PREV.pkl', '_structs.pkl'))
                    ]
                    for file_addr in files:
                        link_addr = pred_root_meta / file_addr.name
                        if link_addr.is_dir() and not link_addr.is_symlink():
                            shutil.rmtree(link_addr)
                        elif link_addr.exists() or link_addr.is_symlink():
                            link_addr.unlink()
                        rel_target = file_addr.relative_to(pred_root_meta)
                        link_addr.symlink_to(rel_target)

            except Exception as e:
                logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                                 'skipping this combination.')
                if RANK == 0:
                    upsert_dataset_status(
                        run_dir=pred_root,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        status='done',
                        error_message=str(e),
                    )
                continue

        if RANK == 0:
            log_run_benchmark_report(pred_root)

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


def run_api_mode(args):
    """Async API pipeline mode for API-based models.

    Uses an optimized pipeline with cross-dataset unified inference queue,
    parallel inference and evaluation, and better remote model utilization.
    """
    from vlmeval.api.adapters import get_adapter_registry
    from vlmeval.inference_api import APIEvalPipeline, DatasetConfig

    # Validate model: API mode only supports a single model
    if isinstance(args.model, list):
        if len(args.model) > 1:
            raise ValueError('API mode only supports a single model. Got: ' + str(args.model))
        args.model = args.model[0]

    # Validate custom_prompt at runtime
    if args.custom_prompt:
        registry = get_adapter_registry()
        assert args.custom_prompt in registry, \
            f'Unknown adapter: {args.custom_prompt}. Available: {list(registry.keys())}'

    assert args.data, '--data must be set in API mode'

    # Prepare work dir and logging
    commit_id = githash(digits=8)
    eval_id = build_eval_id()
    model_name = args.model.replace('/', '--')

    work_dir = Path(args.work_dir) / model_name
    work_dir.mkdir(parents=True, exist_ok=True)

    pred_root = Path(args.work_dir) / model_name / eval_id
    pred_root.mkdir(exist_ok=True)

    log_file = Path(work_dir) / 'logs' / f'{eval_id}_{datetime.datetime.now().strftime("%H%M%S")}.log'
    setup_logger(log_file=str(log_file))
    logger.info(f'Log file: {log_file}')

    if args.mode == 'eval':
        args.reuse = True
        logger.info('Force to use `reuse=True` for eval mode.')

    if not args.reuse:
        logger.warning('--reuse is not set, this run will start from a fresh output directory')
    else:
        logger.info(f'--reuse is set, reuse-aux={args.reuse_aux}')

    WORLD_SIZE_LOCAL = int(os.environ.get('WORLD_SIZE', 1))
    if WORLD_SIZE_LOCAL > 1:
        logger.error("API pipeline does not support multi-process mode (WORLD_SIZE > 1).")
        return

    # Build model args (shared across all datasets)
    if args.base_url is not None:
        model_args = build_model_from_base_url(args)
        model_builder = partial(LMDeployAPI, **model_args)
    else:
        assert model_name in supported_VLM, \
            f'Model "{model_name}" not found in supported_VLM. Consider using --base-url to specify an API endpoint.'
        model_builder = supported_VLM[model_name]

    upsert_run_status(
        pred_root,
        eval_id=eval_id,
        created_at=datetime.datetime.now().astimezone().isoformat(),
        commit=commit_id,
        argv=sys.argv,
        api_mode=True,
        world_size=1,
        pred_format=get_pred_file_format(),
        eval_format=get_eval_file_format(),
        mode=args.mode,
        reuse=bool(args.reuse),
        reuse_aux=args.reuse_aux,
        model_name=model_name,
    )

    # Prepare all datasets
    dataset_configs: List[DatasetConfig] = []

    for ds_name in args.data:
        logger.info(f'-------------------- {ds_name} --------------------')

        try:
            dataset_kwargs = {}
            if ds_name in [
                'MMLongBench_DOC', 'DUDE', 'DUDE_MINI',
                'SLIDEVQA', 'SLIDEVQA_MINI',
            ]:
                dataset_kwargs['model'] = model_name
            dataset = build_dataset(ds_name, **dataset_kwargs)

            if dataset is None:
                logger.error(f'Dataset {ds_name} is not valid, will be skipped.')
                continue

            # Prepare the result file.
            result_file = get_pred_file_path(
                pred_root, model_name, ds_name, use_env_format=True)

            # Skip special datasets.
            if ds_name in ['MMMU_TEST']:
                logger.info(f'{ds_name} requires special handling, skipped in pipeline.')
                continue
            if 'MMT-Bench_ALL' in ds_name:
                logger.info(f'{ds_name} requires special handling, skipped in pipeline.')
                continue

            judge_kwargs = get_judge_kwargs(ds_name, dataset.TYPE, args)
            judge_model = judge_kwargs.get('model', '')
            logger.info(f'Judge kwargs: {judge_kwargs}')

            reuse_ctx = prepare_reuse_files(
                pred_root_meta=str(work_dir),
                eval_id=eval_id,
                model_name=model_name,
                dataset_name=ds_name,
                dataset=dataset,
                result_file=result_file,
                reuse=args.reuse,
                reuse_aux=args.reuse_aux,
                retry_failed=not args.keep_failed,
                judge_model=judge_model if args.mode != 'infer' else None,
                world_size=1,
            )
            upsert_dataset_status(
                run_dir=pred_root,
                model_name=model_name,
                dataset_name=ds_name,
                prediction_file=result_file,
                source_run=reuse_ctx['source_eval_id'],
                judge_model=judge_model,
                reuse_aux=args.reuse_aux,
            )
            if args.mode == 'eval' and not reuse_ctx['prediction_complete']:
                logger.error(
                    f'No reusable completed prediction found for {model_name} x {ds_name}, '
                    'skipping this dataset in eval mode.'
                )
                try:
                    if Path(result_file).exists():
                        skip_reason = 'Incomplete infer result'
                    else:
                        skip_reason = 'No infer result found'
                    upsert_dataset_status(
                        run_dir=pred_root,
                        model_name=model_name,
                        dataset_name=ds_name,
                        status='done',
                        skip_reason=skip_reason,
                    )
                except Exception as summary_err:
                    logger.warning(
                        f'Failed to update status.json for {model_name} x {ds_name}: {summary_err}'
                    )
                continue

            # Complete the dataset config
            if dataset.MODALITY == 'VIDEO':
                dataset_type = 'video'
            elif dataset.TYPE == 'MT':
                dataset_type = 'mt'
            else:
                dataset_type = 'image'
            dataset_config = DatasetConfig(
                dataset_name=ds_name,
                dataset_obj=dataset,
                dataset_type=dataset_type,
                model_obj=model_builder(),
                model_name=model_name,
                work_dir=str(pred_root),
                result_file=result_file,
                judge_kwargs=judge_kwargs,
                verbose=args.verbose
            )
            dataset_configs.append(dataset_config)

        except Exception as e:
            logger.exception(f'Failed to prepare dataset {ds_name}: {e}')
            continue

    # Create and run pipeline
    if len(dataset_configs) == 0:
        logger.warning('No valid datasets to evaluate.')
        return

    logger.info(f"Starting API Pipeline for model: {model_name}")
    logger.info(f"Total datasets: {len(dataset_configs)}")

    pipeline = APIEvalPipeline(
        dataset_configs=dataset_configs,
        concurrency=args.api_nproc,
        monitor_interval=args.monitor_interval,
        run_infer=args.mode in {'infer', 'all'},
        run_eval=args.mode in {'eval', 'all'},
        debug=args.debug,
        retry_failed=not args.keep_failed
    )

    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
    finally:
        log_run_benchmark_report(pred_root)


def main():
    args = parse_args()
    if args.api_mode:
        run_api_mode(args)
    else:
        run_local_mode(args)


if __name__ == '__main__':
    load_env()
    main()
