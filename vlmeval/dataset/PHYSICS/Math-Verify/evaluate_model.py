from datetime import timedelta
import argparse
from pathlib import Path
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate model on math tasks')
    parser.add_argument('--task', type=str, required=True,
                       choices=['gsm8k', 'math', 'math_hard', 'math_500', 'aime24', 'amc23'],
                       help='Task to evaluate')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--use_chat_template', action='store_true', default=False,
                       help='Use chat template')
    parser.add_argument('--override_bs', type=int, default=-1,
                       help='Batch size; -1 for automatic batch size')
    return parser.parse_args()


def main() -> None:
    """Main function to run model evaluation."""
    args = parse_args()
    
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=1000,
        custom_tasks_directory="math_verify.tasks",
        env_config=EnvConfig(cache_dir="tmp/"),
        override_batch_size=args.override_bs,
    )

    model_config = TransformersModelConfig(
        pretrained=args.model,
        dtype="bfloat16",
        use_chat_template=args.use_chat_template,
    )

    pipeline = Pipeline(
        tasks=f"lighteval|{args.task}|4|1",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.show_results()
    pipeline.save_and_push_results()

if __name__ == "__main__":
    main()