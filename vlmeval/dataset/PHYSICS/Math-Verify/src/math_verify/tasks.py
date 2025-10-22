import logging
from textwrap import dedent
from typing import Callable, Optional

import numpy as np
from lighteval.metrics.dynamic_metrics import SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

from math_verify.few_shots import GSM8K_FEW_SHOTS, MATH_HARD_FEW_SHOTS
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

logger = logging.getLogger(__name__)


def as_lighteval_metric(
    metric: Callable[
        [list[str], list[str]], tuple[float, Optional[tuple[list[str], list[str]]]]
    ],
) -> SampleLevelMetric:
    def sample_level_fn(
        formatted_doc: Doc, golds: list[str], predictions: list[str]
    ) -> float:
        result, extracted_predictions = metric(golds, predictions)
        if extracted_predictions is not None:
            if not formatted_doc.specific:
                formatted_doc.specific = {}
            formatted_doc.specific["extracted_predictions"] = extracted_predictions
        return result

    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=sample_level_fn,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def math_hard_prompt_function(x: dict, task_name: str) -> Doc:
    if x.get("__few_shots"):
        index = x["__index"]
        few_shot_doc = (
            MATH_HARD_FEW_SHOTS[index]
            if index < len(MATH_HARD_FEW_SHOTS)
            else MATH_HARD_FEW_SHOTS[-1]
        )
        answer = few_shot_doc["answer"]
        question = few_shot_doc["question"]
    else:
        answer = str(x["solution"])
        question = x["problem"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()

    choices = [answer]
    return Doc(query=query, choices=choices, gold_index=0)


def math_prompt_function(x: dict, task_name: str) -> Doc:
    if x.get("__few_shots"):
        index = x["__index"]
        few_shot_doc = (
            MATH_HARD_FEW_SHOTS[index]
            if index < len(MATH_HARD_FEW_SHOTS)
            else MATH_HARD_FEW_SHOTS[-1]
        )
        answer = few_shot_doc["answer"]
        question = few_shot_doc["question"]
    else:
        answer = str(x["answer"])
        question = x["problem"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()

    choices = [answer]
    return Doc(query=query, choices=choices, gold_index=0)


def math_aime24_prompt_function(x: dict, task_name: str) -> Doc:
    if x.get("__few_shots"):
        index = x["__index"]
        few_shot_doc = (
            MATH_HARD_FEW_SHOTS[index]
            if index < len(MATH_HARD_FEW_SHOTS)
            else MATH_HARD_FEW_SHOTS[-1]
        )
        answer = few_shot_doc["answer"]
        question = few_shot_doc["question"]
    else:
        answer = str(x["reference_solution"])
        question = x["problem"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()

    choices = [f" {answer}"]
    return Doc(query=query, choices=choices, gold_index=0)


def math_amc23_prompt_function(x: dict, task_name: str) -> Doc:
    if x.get("__few_shots"):
        index = x["__index"]
        few_shot_doc = (
            MATH_HARD_FEW_SHOTS[index]
            if index < len(MATH_HARD_FEW_SHOTS)
            else MATH_HARD_FEW_SHOTS[-1]
        )
        answer = few_shot_doc["answer"]
        question = few_shot_doc["question"]
    else:
        answer = str(x["answer"])
        question = x["question"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()
    choices = [f" {answer}"]
    return Doc(query=query, choices=choices, gold_index=0)


def gsm8k_prompt_function(x: dict, task_name: str) -> Doc:
    if x.get("__few_shots"):
        index = x["__index"]
        few_shot_doc = (
            GSM8K_FEW_SHOTS[index]
            if index < len(GSM8K_FEW_SHOTS)
            else GSM8K_FEW_SHOTS[-1]
        )
        answer = few_shot_doc["answer"]
        question = few_shot_doc["question"]
    else:
        answer = f"{x['answer'].split('####')[-1].strip()}"
        question = x["question"]

    query = dedent(
        f"""\
Question: {question}
Step-by-Step Answer:\
"""
    ).strip()

    choices = [f" {answer}"]
    return Doc(query=query, choices=choices, gold_index=0)


math_hard_lighteval = [
    LightevalTaskConfig(
        name=f"math_hard:{subset}",
        suite=["lighteval", "math"],
        prompt_function=math_hard_prompt_function,
        hf_repo="lighteval/MATH-Hard",
        hf_subset=subset,
        evaluation_splits=["test"],
        few_shots_split="train",
        generation_size=1024,
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(
                        LatexExtractionConfig(boxed_match_priority=0),
                    ),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                )
            ),
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
    for subset in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
]

math_500_lighteval = [
    LightevalTaskConfig(
        name="math_500",
        suite=["lighteval", "math"],
        prompt_function=math_prompt_function,
        hf_repo="HuggingFaceH4/MATH-500",
        hf_subset="default",
        evaluation_splits=["test"],
        few_shots_split="test",
        generation_size=1024,
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(
                        LatexExtractionConfig(boxed_match_priority=0),
                    ),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                )
            ),
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
]


aime24_lighteval = [
    LightevalTaskConfig(
        name="aime24",
        suite=["lighteval", "math"],
        prompt_function=math_aime24_prompt_function,
        hf_repo="zwhe99/aime24",
        hf_subset="default",
        evaluation_splits=["test"],
        few_shots_split="test",
        generation_size=1024,
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(LatexExtractionConfig(),),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                )
            ),
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
]

amc23_lighteval = [
    LightevalTaskConfig(
        name="amc23",
        suite=["lighteval", "math"],
        prompt_function=math_amc23_prompt_function,
        hf_repo="zwhe99/amc23",
        hf_subset="default",
        hf_filter=lambda x: len(x["question"].strip()) > 0,
        evaluation_splits=["test"],
        few_shots_split="test",
        generation_size=1024,
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(ExprExtractionConfig(),),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                )
            ),
        ],
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        trust_dataset=True,
        version=0,
    )
]

gsm8k_lighteval = [
    LightevalTaskConfig(
        name="gsm8k",
        suite=["lighteval", "math"],
        prompt_function=gsm8k_prompt_function,
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_filter=lambda x: len(x["question"].strip()) > 0,
        evaluation_splits=["test"],
        few_shots_split="test",
        generation_size=1024,
        stop_sequence=["\nQuestion:", "\nProblem:", "\nquestion:", "\nproblem:"],
        metric=[
            as_lighteval_metric(
                math_metric(
                    gold_extraction_target=(ExprExtractionConfig(),),
                    pred_extraction_target=(
                        LatexExtractionConfig(),
                        ExprExtractionConfig(),
                    ),
                    fallback_mode="first_match",
                )
            ),
        ],
    )
]


TASKS_TABLE = [
    *gsm8k_lighteval,
    *math_hard_lighteval,
    *math_500_lighteval,
    *aime24_lighteval,
    *amc23_lighteval,
]
