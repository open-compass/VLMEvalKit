import json
import os
import re
import subprocess
import argparse
import shutil
from datetime import datetime, timedelta

from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer

def encode_dataset_name(base_name, test_range=""):
    """
    将数据集名称和切片参数编码为一个字符串。
    
    :param base_name: 基础数据集名称
    :param test_range: 切片范围字符串，如 "[::2]", "[:50]" 等
    :return: 编码后的数据集名称
    """
    if not test_range:
        return base_name

    # 使用正则表达式解析 test_range
    pattern = r'\[(\-?\d*)?:(\-?\d*)?(?::(\-?\d+))?\]'
    match = re.match(pattern, test_range)
    
    if not match:
        raise ValueError(f"Invalid test_range format: {test_range}")

    start, end, step = match.groups()
    
    # 处理空参数
    start = start if start is not None else ""
    end = end if end is not None else ""
    step = step if step is not None else ""

    slice_params = []
    if start and start != "0":
        slice_params.append(f"from{start}")
    if end:
        slice_params.append(f"to{end}")
    if step and step != "1":  # 注意：step 是字符串，需要与字符串比较
        slice_params.append(f"step{step}")
    
    if slice_params:
        return f"{base_name}_{'_'.join(slice_params)}"
    else:
        return base_name


def build_api_model(args):
    """构建 API 模型实例"""
    from vlmeval.api.gpt import VLLMAPI
    
    # 构建模型配置
    model_config = {
        'model': args.model_name,
        'api_base': args.base_url,
        'key': args.api_key,
        'temperature': args.temperature,
        'max_out_tokens': args.max_out_tokens,
        'retry': args.retry,
        'verbose': args.verbose,
    }
    
    # 添加可选参数
    if args.top_p is not None:
        model_config['top_p'] = args.top_p
    if args.min_pixels is not None:
        model_config['min_pixels'] = args.min_pixels
    if args.max_pixels is not None:
        model_config['max_pixels'] = args.max_pixels
    if args.img_detail is not None:
        model_config['img_detail'] = args.img_detail
    if args.timeout is not None:
        model_config['timeout'] = args.timeout
    if args.system_prompt is not None:
        model_config['system_prompt'] = args.system_prompt
    
    # 移除 None 值
    model_config = {k: v for k, v in model_config.items() if v is not None}
    
    return VLLMAPI(**model_config)


def parse_args():
    help_msg = """\
A specialized evaluation entry for OpenAI-style API models.

This script allows you to evaluate API models without modifying config files.
Simply provide the API parameters via command line arguments.

Example usage:
    python run_api.py \\
      --model-name "gpt-4o" \\
      --base-url "https://api.openai.com/v1" \\
      --api-key "your-api-key" \\
      --max-tokens 4096 \\
      --temperature 0.0 \\
      --data MME MMBench \\
      --work-dir ./outputs
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    
    # API Model Configuration
    parser.add_argument('--model-name', type=str, required=True, 
                       help='Model name for API calls (e.g., gpt-4o, claude-3-opus)')
    parser.add_argument('--base-url', type=str, required=True,
                       help='API base URL (e.g., https://api.openai.com/v1)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for authentication')
    
    # Generation Parameters
    parser.add_argument('--max-out-tokens', type=int, default=8192,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=None,
                       help='Top-p sampling parameter')
    
    # Vision Parameters
    parser.add_argument('--min-pixels', type=str, default=None,
                       help='Minimum pixels for image processing (e.g., "3k")')
    parser.add_argument('--max-pixels', type=str, default=None,
                       help='Maximum pixels for image processing (e.g., "100w")')
    parser.add_argument('--img-detail', type=str, choices=['low', 'high', 'auto'], default=None,
                       help='Image detail level for vision models')
    
    # Other API Parameters
    parser.add_argument('--timeout', type=int, default=None,
                       help='Request timeout in seconds')
    parser.add_argument('--system-prompt', type=str, default=None,
                       help='System prompt for the model')
    
    # Dataset and Evaluation
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Names of datasets to evaluate')
    
    # Work Directory
    parser.add_argument('--work-dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    # Evaluation Mode
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'],
                       help='Evaluation mode: "all" for inference+evaluation, "infer" for inference only')
    
    # API and Retry Settings
    parser.add_argument('--api-nproc', type=int, default=4,
                       help='Number of parallel API calls')
    parser.add_argument('--retry', type=int, default=10,
                       help='Number of retries for failed API calls')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # Judge Model Settings
    parser.add_argument('--judge', type=str, default=None,
                       help='Judge model for evaluation')
    parser.add_argument('--judge-args', type=str, default=None,
                       help='Judge arguments in JSON format')
    
    # Resume and Reuse Settings
    parser.add_argument('--ignore', action='store_true',
                       help='Ignore failed indices')
    parser.add_argument('--reuse', action='store_true',
                       help='Reuse existing prediction files')
    parser.add_argument('--reuse-aux', type=bool, default=True,
                       help='Reuse auxiliary evaluation files')
    
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN_API')
    args = parse_args()
    
    logger.info(f'Starting evaluation for API model: {args.model_name}')
    logger.info(f'Base URL: {args.base_url}')
    logger.info(f'Datasets: {args.data}')
    
    if not args.reuse:
        logger.warning('--reuse is not set, will not reuse previous temporary files')
    else:
        logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    # Build the API model
    model = build_api_model(args)
    model_name = args.model_name
    
    # Setup directories
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
    
    # Process each dataset
    for dataset_name in args.data:
        try:
            # Parse dataset name with potential slicing
            original_dataset_name = dataset_name
            test_range = None
            
            # Check if dataset_name contains slicing expression
            if re.search(r'\[\s*[-?\d]*\s*:\s*[-?\d]*\s*(?::\s*[-?\d]+\s*)?\]$', dataset_name):
                match = re.search(r'(\[\s*[-?\d]*\s*:\s*[-?\d]*\s*(?::\s*[-?\d]+\s*)?\])$', dataset_name)
                if match:
                    test_range = match.group(1).strip()
                    dataset_name = dataset_name[:match.start()].strip()
            
            dataset_kwargs = {}
            if test_range:
                dataset_kwargs['test_range'] = test_range
            
            # Special handling for certain datasets
            if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                dataset_kwargs['model'] = model_name
            
            dataset = build_dataset(dataset_name, **dataset_kwargs)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_name} could not be built with kwargs {dataset_kwargs}")
            
            # Update dataset name with encoding
            dataset.dataset_name = encode_dataset_name(dataset.dataset_name, test_range=test_range)
            dataset_name = dataset.dataset_name
            
            if dataset is None:
                logger.error(f'Dataset {dataset_name} is not valid, will be skipped.')
                continue
            
            # Setup result file
            result_file_base = f'{model_name}_{dataset_name}.xlsx'
            if dataset.TYPE == 'MT':
                result_file_base = result_file_base.replace('.xlsx', '.tsv')
            
            result_file = osp.join(pred_root, result_file_base)
            
            # Handle reuse of previous results
            if len(prev_pred_roots):
                prev_result_files = []
                prev_pkl_file_list = []
                for root in prev_pred_roots[::-1]:
                    if osp.exists(osp.join(root, result_file_base)):
                        if args.reuse_aux:
                            prev_result_files = fetch_aux_files(osp.join(root, result_file_base))
                        else:
                            prev_result_files = [osp.join(root, result_file_base)]
                        break
                    elif commit_id in root and len(ls(root)) and root != pred_root:
                        temp_files = ls(root, match=[dataset_name, '.pkl'])
                        if len(temp_files):
                            prev_pkl_file_list.extend(temp_files)
                            break
                
                if not args.reuse:
                    prev_result_files = []
                    prev_pkl_file_list = []
                
                if len(prev_result_files):
                    for prev_result_file in prev_result_files:
                        src = prev_result_file
                        tgt = osp.join(pred_root, osp.basename(src))
                        if not osp.exists(tgt):
                            shutil.copy(src, tgt)
                            logger.info(f'--reuse is set, will reuse the prediction file {src}.')
                        else:
                            logger.warning(f'File already exists: {tgt}')
                
                elif len(prev_pkl_file_list):
                    for fname in prev_pkl_file_list:
                        target_path = osp.join(pred_root, osp.basename(fname))
                        if not osp.exists(target_path):
                            shutil.copy(fname, target_path)
                            logger.info(f'--reuse is set, will reuse the prediction pickle file {fname}.')
                        else:
                            logger.warning(f'File already exists: {target_path}')
            
            # Perform inference
            if dataset.MODALITY == 'VIDEO':
                model = infer_data_job_video(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    result_file_name=result_file_base,
                    verbose=args.verbose,
                    api_nproc=args.api_nproc)
            elif dataset.TYPE == 'MT':
                model = infer_data_job_mt(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=args.api_nproc,
                    ignore_failed=args.ignore)
            else:
                model = infer_data_job(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=args.api_nproc,
                    ignore_failed=args.ignore)
            
            # Setup judge kwargs
            judge_kwargs = {
                'nproc': args.api_nproc,
                'verbose': args.verbose,
                'retry': args.retry,
                **(json.loads(args.judge_args) if args.judge_args else {}),
            }
            
            if args.judge is not None:
                judge_kwargs['model'] = args.judge
            else:
                # Auto-select judge model based on dataset
                if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(['moviechat1k'], dataset_name.lower()):
                    if listinstr(['WeMath'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['VisuLogic'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o-mini'
                    else:
                        judge_kwargs['model'] = 'chatgpt-0125'
                elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4-turbo'
                elif listinstr(['VGRPBench'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'
                elif listinstr(['CharXiv_reasoning_val'], dataset_name):
                    judge_kwargs['model'] = 'xhs-deepseek'
                elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT', 'OCR_Reasoning', 'CharXiv_descriptive_val'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o-mini'
                elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'
                elif listinstr(['VDC'], dataset_name):
                    judge_kwargs['model'] = 'llama31-8b'
                elif listinstr(['VideoMMLU_QA', 'VideoMMLU_CAP'], dataset_name):
                    judge_kwargs['model'] = 'qwen-72b'
            
            logger.info(f'Judge kwargs: {judge_kwargs}')
            
            # Handle special submission formats
            if dataset_name in ['MMMU_TEST']:
                result_json = MMMU_result_transfer(result_file)
                logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                            f'json file saved in {result_json}')
                continue
            elif 'MMT-Bench_ALL' in dataset_name:
                submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation, '
                            f'submission file saved in {submission_file}')
                continue
            
            # Skip evaluation if only inference is requested
            if args.mode == 'infer':
                continue
            
            # Skip evaluation for datasets without ground truth
            skip_eval_datasets = [
                'MLLMGuard_DS', 'AesBench_TEST', 'DocVQA_TEST', 'InfoVQA_TEST', 
                'Q-Bench1_TEST', 'A-Bench_TEST', 'MMBench_TEST_CN', 'MMBench_TEST_EN', 
                'MMBench', 'MMBench_CN', 'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 
                'MMBench_V11', 'MMBench_CN_V11'
            ]
            
            if any(skip_name in dataset_name for skip_name in skip_eval_datasets):
                logger.info(f'{dataset_name} is a test split without ground-truth or evaluation is not supported. '
                            'Skipping evaluation.')
                continue
            
            # Setup proxy for evaluation
            eval_proxy = os.environ.get('EVAL_PROXY', None)
            old_proxy = os.environ.get('HTTP_PROXY', '')
            if eval_proxy is not None:
                proxy_set(eval_proxy)
            
            # Perform evaluation
            try:
                if judge_kwargs.get('model', None) == 'xhs-deepseek':
                    judge_kwargs['nproc'] = min(judge_kwargs['nproc'], 16)
                elif judge_kwargs.get('model', None) == 'gpt-4o':
                    judge_kwargs['nproc'] = min(judge_kwargs['nproc'], 4)

                eval_results = dataset.evaluate(result_file, **judge_kwargs)
                if eval_results is not None:
                    logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished!')
                    logger.info('Evaluation Results:')
                    if isinstance(eval_results, dict):
                        logger.info('\n' + json.dumps(eval_results, indent=4))
                    elif isinstance(eval_results, pd.DataFrame):
                        if len(eval_results) < len(eval_results.columns):
                            eval_results = eval_results.T
                        logger.info('\n' + tabulate(eval_results))
            except Exception as e:
                logger.error(f'Evaluation failed for {dataset_name}: {e}')
            
            # Restore proxy
            if eval_proxy is not None:
                proxy_set(old_proxy)
            
            # Create symbolic links
            files = os.listdir(pred_root)
            files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
            for f in files:
                cwd = os.getcwd()
                file_addr = osp.join(cwd, pred_root, f)
                link_addr = osp.join(cwd, pred_root_meta, f)
                if osp.exists(link_addr) or osp.islink(link_addr):
                    os.remove(link_addr)
                os.symlink(file_addr, link_addr)
        
        except Exception as e:
            logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                           'skipping this combination.')
            continue

if __name__ == '__main__':
    load_env()
    main()
