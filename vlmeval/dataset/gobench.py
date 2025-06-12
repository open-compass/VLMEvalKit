# vlmeval/dataset/gobench.py

import pandas as pd
import re
import numpy as np
from ..smp import *
from .image_base import ImageBaseDataset
import os
import warnings


class GOBenchDataset(ImageBaseDataset):
    """
    GOBenchDataset for evaluating image generation models based on reality,
    aesthetics, and instruction consistency.
    It uses a custom prompt and a regression-style evaluation logic.
    """

    TYPE = 'GOBench_QA'
    FAIL_MSG = 'Failed to obtain answer'

    SYS = (
        "Please analyze the image based on the provided prompt and answer the following three questions.\n\n"
        "---\n"
        "**Question 1: Reality Assessment**\n"
        "Please answer each of the provided reality questions with only Yes, No, or Cannot Determine.\n\n"
        "---\n"
        "**Question 2: Aesthetics Rating**\n"
        "Please rate the aesthetics of the image on a scale of 1-5.\n\n"
        "---\n"
        "**Question 3: Instruction Consistency Rating**\n"
        "Based on the image and the provided prompt, \
        please rate the instruction consistency of the generated image on a scale of 1-5.\n"
    )

    DATASET_URL = {
        "GOBench": "https://huggingface.co/datasets/bonnot/GOBench/resolve/main/GOBench.tsv"
    }

    DATASET_MD5 = {
        "GOBench": "9f37ad20d99a9d4159fe46ab928741be"
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def build_prompt(self, line, **kwargs):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_paths = self.dump_image(line)

        if not image_paths:
            print(f"ERROR: Could not find or dump image for index {line.get('index', 'N/A')}")
            return [dict(type='text', value='Failed to load image.')]

        image_path = image_paths[0]
        question = line['question']

        message = [
            dict(type='image', value=image_path),
            dict(type='text', value=self.SYS + '\n' + question)
        ]
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        results_df = load(eval_file)
        eval_results = []
        for index, row in results_df.iterrows():
            prediction_str = str(row.get('prediction', ''))
            failed_scores = {'reality': -1.0, 'aesthetics': -1.0, 'consistency': -1.0}

            if pd.isna(row.get('prediction')) or self.FAIL_MSG in prediction_str:
                eval_results.append(failed_scores)
                continue

            try:
                gt_answer_str = str(row.get('answer', ''))
                gt_scores = self.parse_ground_truth(gt_answer_str)
                pred_scores = self.parse_prediction(prediction_str)

                if not gt_scores or not pred_scores:
                    eval_results.append(failed_scores)
                    continue

                scores = {}
                for key in gt_scores:
                    gt = gt_scores[key]
                    pred = pred_scores.get(key)
                    if pred is None:
                        scores = failed_scores
                        break
                    diff = abs(gt - pred)
                    scores[key] = max(0,(0.5 - diff) / 0.5)
                eval_results.append(scores)

            except Exception as e:
                print(f"ERROR evaluating row {index}: {e}\nPrediction was: {prediction_str}")
                eval_results.append(failed_scores)

        if not eval_results:
            warnings.warn("No evaluation results were generated.")
            return pd.DataFrame()

        scores_df = pd.DataFrame(eval_results)
        final_df = pd.concat([results_df, scores_df.add_suffix('_score')], axis=1)

        total_questions = len(final_df)
        failed_mask = (scores_df == -1.0).all(axis=1)
        num_failed = failed_mask.sum()

        print(
            f'Among {total_questions} questions, failed to obtain a valid,'
            f' parsable prediction for {num_failed} questions. '
            f'These questions will be marked with a score of -1 and excluded from the final average calculation.'
        )

        valid_scores_df = scores_df.replace(-1.0, np.nan)
        avg_scores = valid_scores_df.mean().fillna(0)
        overall_score = avg_scores.mean()

        report = pd.DataFrame({
            'split': ['none'],
            'Overall': [overall_score * 100],
            'Reality_Score': [avg_scores.get('reality', 0) * 100],
            'Aesthetics_Score': [avg_scores.get('aesthetics', 0) * 100],
            'Instruction_Consistency_Score': [avg_scores.get('consistency', 0) * 100]
        })

        score_file = eval_file.replace('.xlsx', '_score.xlsx')
        dump(final_df, score_file)
        print(f"Detailed scores including failed attempts saved to {score_file}")

        return report.round(2)

    def parse_ground_truth(self, gt_str):
        scores = {}
        patterns = {
            'reality': r"Reality Assessment Score:\s*([\d.]+)",
            'aesthetics': r"Aesthetics Rating:\s*([\d.]+)",
            'consistency': r"Instruction Consistency Rating:\s*([\d.]+)"
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, gt_str, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))
        return scores if len(scores) == 3 else None

    def parse_prediction(self, pred_str):
        scores = {}
        reality_match = re.search(r"Total Score:[\s\*]*([\d.]+)", pred_str, re.IGNORECASE)
        if reality_match:
            scores['reality'] = float(reality_match.group(1))

        aesthetics_match = re.search(r"Aesthetics Rating\D*([\d.]+)", pred_str, re.IGNORECASE)
        if aesthetics_match:
            scores['aesthetics'] = float(aesthetics_match.group(1))

        consistency_match = re.search(r"Instruction Consistency Rating\D*([\d.]+)", pred_str, re.IGNORECASE)
        if consistency_match:
            scores['consistency'] = float(consistency_match.group(1))

        return scores if len(scores) == 3 else None
