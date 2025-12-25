import asyncio
import json
import os
import argparse
import datetime
from functools import partial
from pathlib import Path
from typing import List

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.smp import *
from vlmeval.api import LMDeployAPI

from vlmeval.inference_api import APIEvalPipeline, DatasetConfig


def get_judge_kwargs(dataset_name: str, args) -> dict:
    """Determine the default judge kwargs by dataset name."""
    judge_kwargs = {
        'nproc': args.judge_api_nproc,
        'verbose': args.verbose,
        'retry': args.retry if args.retry is not None else 3,
        **(json.loads(args.judge_args) if args.judge_args else {}),
    }

    if args.judge_base_url:
        judge_kwargs['api_base'] = f"{args.judge_base_url.rstrip('/')}/chat/completions"
    if args.judge_key:
        judge_kwargs['key'] = args.judge_key
    if args.retry is not None:
        judge_kwargs['retry'] = args.retry

    if args.judge is not None:
        judge_kwargs['model'] = args.judge
    else:
        judge_kwargs['model'] = 'gpt-4o-mini'  # default

        dataset_lower = dataset_name.lower()

        if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['VisuLogic'], dataset_name):
            judge_kwargs['model'] = 'exact_matching'
        elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
            if listinstr(['LLaVABench_KO'], dataset_name):
                judge_kwargs['model'] = 'gpt-4o-0806'
            else:
                judge_kwargs['model'] = 'gpt-4-turbo'
        elif listinstr(['VGRPBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath',
                        'VL-RewardBench', 'LogicVista', 'MOAT', 'OCR_Reasoning'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['OlympiadBench'], dataset_name):
            use_api_judger = judge_kwargs.get("olympiad_use_api_judger", False)
            if use_api_judger:
                judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench',
                        'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name):
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
        elif listinstr(['Video-MME'], dataset_name):
            judge_kwargs['model'] = 'chatgpt-0125'

    if args.use_verifier:
        judge_kwargs['use_verifier'] = True
    if args.use_vllm:
        judge_kwargs['use_vllm'] = True

    return judge_kwargs


def parse_args():
    help_msg = """\
VLMEvalKit API Pipeline Runner

This script uses an optimized pipeline for API-based models with the following improvements:
- Cross-dataset unified inference queue
- Parallel inference and evaluation
- Better remote model utilization

You can launch the evaluation by setting either --data and --model.

--data and --model:
    Specify dataset names and model configuration for API-based inference.

For more details, see the documentation in run.py.
"""
    parser = argparse.ArgumentParser(
        description=help_msg,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')

    # ================ For infer model ==============
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--base-url', type=str, default=None, help='API base URL')
    parser.add_argument('--key', type=str, default='sk-admin', help='API key')
    parser.add_argument('--thinker', action='store_true',
                        help='Longer timeout and higher max_tokens')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--repetition-penalty', type=float, default=None)
    parser.add_argument('--api-nproc', type=int, default=32,
                        help='Parallel API calling (inference concurrency)')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Max time for inferencing')

    # ================ For judge model ==============
    parser.add_argument('--judge', type=str, default=None)
    parser.add_argument('--judge-base-url', type=str, default=None,
                        help='The base url of judger')
    parser.add_argument('--judge-key', type=str, default='sk-admin',
                        help='The key of judger')
    parser.add_argument('--judge-api-nproc', type=int, default=32,
                        help='Parallel API calling for judger')
    parser.add_argument('--judge-args', type=str, default=None,
                        help='Judge arguments in JSON format')

    # ==============================================
    parser.add_argument('--custom-prompt',
                        type=str,
                        choices=list(LMDeployAPI.prompt_map.keys()),
                        default=None)

    parser.add_argument('--work-dir', type=str, default='./outputs',
                        help='Select the output directory')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'infer', 'eval'],
                        help='Mode: all (infer+eval), infer (only), eval (only)')
    parser.add_argument('--retry', type=int, default=None,
                        help='Retry numbers for API VLMs')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore', action='store_true',
                        help='Ignore failed indices')
    parser.add_argument('--reuse', action='store_true',
                        help='Reuse existing prediction files')
    parser.add_argument('--reuse-aux', type=int, default=True,
                        help='Reuse auxiliary evaluation files')
    parser.add_argument('--use-vllm', action='store_true',
                        help='Use vllm to generate')
    parser.add_argument('--use-verifier', action='store_true',
                        help='Use verifier to evaluate')
    parser.add_argument('--monitor-interval', type=int, default=30,
                        help='Status monitoring interval (seconds)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: run evaluation in main process')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # ==============================================
    # Prepare work dir
    # ==============================================
    date, commit_id = timestr('day'), githash(digits=8)
    eval_id = f"T{date}_G{commit_id}"
    model_name = args.model.replace('/', '--')

    # Work dir for the specified model
    work_dir = Path(args.work_dir) / model_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Work dir for the current run
    pred_root = Path(args.work_dir) / model_name / eval_id
    # List previous run
    prev_pred_roots = sorted(d for d in work_dir.iterdir() if d.is_dir())
    pred_root.mkdir(exist_ok=True)

    log_file = Path(work_dir) / 'logs' / f'{eval_id}_{datetime.datetime.now().strftime("%H%M%S")}.log'
    logger = setup_logger(log_file=str(log_file))
    logger.info(f'Log file: {log_file}')

    if args.mode == 'eval':
        args.reuse = True
        logger.info('Force to use `reuse=True` for eval mode.')

    if not args.reuse:
        logger.warning('--reuse is not set, will not reuse previous temporary files')
    else:
        logger.info('--reuse is set, will reuse the latest prediction & temporary files')

    WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
    if WORLD_SIZE > 1:
        logger.error("API pipeline does not support multi-process mode (WORLD_SIZE > 1).")
        return

    # ==============================================
    # Prepare all datasets
    # ==============================================
    dataset_configs: List[DatasetConfig] = []

    for ds_name in args.data:
        logger.info(f'Preparing dataset: {ds_name}')

        use_think_args = args.thinker

        # Construct the model builder for the dataset.
        if args.base_url is not None:
            model_args = dict(
                model=args.model,
                api_base=f"{args.base_url.rstrip('/')}/chat/completions",
                key=args.key,
                custom_prompt=args.custom_prompt,
                max_tokens=2**15,
                retry=6,
                timeout=args.timeout,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                verbose=args.verbose,
            )
            if use_think_args:
                model_args.update(dict(timeout=args.timeout * 2, max_tokens=2**16))
            model_builder = partial(LMDeployAPI, **model_args)
        else:
            assert model_name in supported_VLM, \
                f'Unsupported internal VLM name: {model_name}. Consider using `--base-url`.'
            model_builder = supported_VLM[model_name]

        # Construct the dataset.
        try:
            dataset_kwargs = {}
            if ds_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI',
                                'SLIDEVQA', 'SLIDEVQA_MINI']:
                dataset_kwargs['model'] = model_name
            dataset = build_dataset(ds_name, **dataset_kwargs)

            if dataset is None:
                logger.error(f'Dataset {ds_name} is not valid, will be skipped.')
                continue

            # Prepare the result file.
            pred_format = get_pred_file_format()
            result_file_base = f'{model_name}_{ds_name}.{pred_format}'
            result_file = str(pred_root / result_file_base)

            # Prepare the reuse file
            if args.reuse and len(prev_pred_roots):
                prepare_reuse_files(
                    pred_root_meta=str(work_dir),
                    eval_id=eval_id,
                    model_name=model_name,
                    dataset_name=ds_name,
                    reuse=args.reuse,
                    reuse_aux=args.reuse_aux
                )

            # Skip special datasets.
            if ds_name in ['MMMU_TEST']:
                logger.info(f'{ds_name} requires special handling, skipped in pipeline.')
                continue
            if 'MMT-Bench_ALL' in ds_name:
                logger.info(f'{ds_name} requires special handling, skipped in pipeline.')
                continue

            # Prepare judge kwargs
            judge_kwargs = get_judge_kwargs(ds_name, args)

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

    # ==============================================
    # Create and run pipeline
    # ==============================================
    if len(dataset_configs) == 0:
        logger.warning(f'No valid datasets to evaluate.')
        return

    logger.info(f"Starting API Pipeline for model: {model_name}")
    logger.info(f"Total datasets: {len(dataset_configs)}")

    pipeline = APIEvalPipeline(
        dataset_configs=dataset_configs,
        concurrency=args.api_nproc,
        monitor_interval=args.monitor_interval,
        run_infer=args.mode in {'infer', 'all'},
        run_eval=args.mode in {'eval', 'all'},
        debug=args.debug
    )

    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user.")
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")

    # Create symbolic links.
    try:
        files = list(pred_root.iterdir())
        for ds_name in args.data:
            files_to_link = [f for f in files if f.is_file() and f'{model_name}_{ds_name}' in f.name]
            for f in files_to_link:
                file_addr = pred_root.absolute() / f.name
                link_addr = work_dir.absolute() / f.name
                if link_addr.exists() or link_addr.is_symlink():
                    link_addr.unlink()
                link_addr.symlink_to(file_addr.relative_to(link_addr.parent))
    except Exception as e:
        logger.warning(f"Failed to create symbolic links: {e}")


if __name__ == '__main__':
    load_env()
    main()
