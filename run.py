import torch
import torch.distributed as dist
from vlmeval.smp import *
from vlmeval.evaluate import *
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.config import supported_VLM
from vlmeval.dataset import dataset_URLs, DATASET_TYPE, abbr2full
from vlmeval.utils import MMMU_result_transfer, MMTBench_result_transfer


def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    # Args that only apply to Video Dataset
    parser.add_argument('--nframe', type=int, default=8)
    parser.add_argument('--pack', action='store_true')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='.', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Rerun: will remove all evaluation temp files
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')

    args = parse_args()
    assert len(args.data), '--data should be a list of data files'

    if args.retry is not None:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))

    for _, model_name in enumerate(args.model):
        model = None

        pred_root = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root, exist_ok=True)

        for _, dataset_name in enumerate(args.data):
            custom_flag = False

            if dataset_name not in dataset_URLs:
                file_path = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
                if not osp.exists(file_path):
                    dataset_name = abbr2full(dataset_name)
                else:
                    dataset_name = dataset_name

            if dataset_name not in dataset_URLs:
                logger.warning(f'Dataset {dataset_name} is not officially supported. ')
                file_path = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
                if not osp.exists(file_path):
                    logger.error(f'Cannot find the local dataset {dataset_name}. ')
                    continue
                else:
                    custom_flag = True

            if dataset_name in ['MMBench-Video']:
                packstr = 'pack' if args.pack else 'nopack'
                result_file = f'{pred_root}/{model_name}_{dataset_name}_{args.nframe}frame_{packstr}.xlsx'
            else:
                result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'
            if osp.exists(result_file) and args.rerun:
                os.system(f'rm {pred_root}/{model_name}_{dataset_name}_*')

            if model is None:
                model = model_name  # which is only a name

            # Perform the Inference
            if dataset_name == 'MMBench-Video':
                model = infer_data_job_video(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    nframe=args.nframe,
                    pack=args.pack,
                    verbose=args.verbose,
                    api_nproc=args.nproc)
            else:
                model = infer_data_job(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    verbose=args.verbose,
                    api_nproc=args.nproc,
                    ignore_failed=args.ignore)

            # Set the judge kwargs first before evaluation or dumping
            judge_kwargs = {
                'nproc': args.nproc,
                'verbose': args.verbose,
            }
            if args.retry is not None:
                judge_kwargs['retry'] = args.retry
            if args.judge is not None:
                judge_kwargs['model'] = args.judge
            else:
                if DATASET_TYPE(dataset_name) in ['multi-choice', 'Y/N']:
                    judge_kwargs['model'] = 'chatgpt-0125'
                elif listinstr(['MMVet', 'MathVista', 'LLaVABench', 'MMBench-Video'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4-turbo'
            if 'OPENAI_API_KEY_JUDGE' in os.environ and len(os.environ['OPENAI_API_KEY_JUDGE']):
                judge_kwargs['key'] = os.environ['OPENAI_API_KEY_JUDGE']
            if 'OPENAI_API_BASE_JUDGE' in os.environ and len(os.environ['OPENAI_API_BASE_JUDGE']):
                judge_kwargs['api_base'] = os.environ['OPENAI_API_BASE_JUDGE']

            if rank == 0:
                if dataset_name in ['MMMU_TEST']:
                    result_json = MMMU_result_transfer(result_file)
                    logger.info(f'Transfer MMMU_TEST result to json for official evaluation, json file saved in {result_json}')  # noqa: E501
                    continue
                elif 'MMT-Bench_ALL' in dataset_name:
                    submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                    logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation (https://eval.ai/web/challenges/challenge-page/2328/overview), submission file saved in {submission_file}')  # noqa: E501
                    continue
                elif 'MLLMGuard_DS' in dataset_name:
                    logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')  # noqa: E501
                    continue

            if dataset_name in [
                'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN'
                'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
            ]:
                if not MMBenchOfficialServer(dataset_name):
                    logger.error(
                        f'Can not evaluate {dataset_name} on non-official servers, '
                        'will skip the evaluation. '
                    )
                    continue

            if rank == 0 and args.mode == 'all':
                if DATASET_TYPE(dataset_name) == 'multi-choice':
                    dataset_name = 'default' if custom_flag else dataset_name
                    multiple_choice_eval(
                        result_file,
                        dataset=dataset_name,
                        **judge_kwargs)

                elif DATASET_TYPE(dataset_name) == 'Y/N':
                    YOrN_eval(
                        result_file,
                        dataset=dataset_name,
                        **judge_kwargs)

                elif DATASET_TYPE(dataset_name) == 'Caption':
                    COCO_eval(result_file)
                elif dataset_name == 'MMVet':
                    MMVet_eval(result_file, **judge_kwargs)
                elif dataset_name == 'OCRBench':
                    OCRBench_eval(result_file)
                elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA'], dataset_name):
                    VQAEval(result_file, dataset_name)
                elif listinstr(['MathVista'], dataset_name):
                    MathVista_eval(result_file, **judge_kwargs)
                elif listinstr(['LLaVABench'], dataset_name):
                    LLaVABench_eval(result_file, **judge_kwargs)
                elif listinstr(['MMBench-Video'], dataset_name):
                    MMBenchVideo_eval(result_file, **judge_kwargs)
                else:
                    logger.error(f'Dataset {dataset_name} is not handled by evaluator, will be skipped. ')


if __name__ == '__main__':
    load_env()
    main()
