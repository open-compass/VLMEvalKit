# vlmeval/dataset/gobench.py

import pandas as pd
import re
from ..smp import *
from .image_base import ImageBaseDataset
import os


class GOBenchDataset(ImageBaseDataset):
    """
    GOBenchDataset for evaluating image generation models based on reality,
    aesthetics, and instruction consistency.
    It uses a custom prompt and a regression-style evaluation logic.
    """

    TYPE = 'GOBench_QA'

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
            self.logger.error(f"Could not find or dump image for index {line.get('index', 'N/A')}")
            return None

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
            gt_answer_str = str(row.get('answer', ''))
            prediction_str = str(row.get('prediction', ''))

            # 默认得分为0
            scores = {'reality': 0.0, 'aesthetics': 0.0, 'consistency': 0.0}

            if pd.isna(prediction_str) or "Failed to obtain answer" in prediction_str:
                eval_results.append(scores)
                continue

            try:
                gt_scores = self.parse_ground_truth(gt_answer_str)
                pred_scores = self.parse_prediction(prediction_str)

                if not gt_scores or not pred_scores:
                    eval_results.append(scores)
                    continue

                for key in gt_scores:
                    gt = gt_scores[key]
                    pred = pred_scores.get(key, 0)
                    index = 1
                    diff = abs(gt - pred)

                    if diff <= 1:
                        scores[key] = (1 - diff) / 1
                    else:
                        scores[key] = 0.0
                eval_results.append(scores)

            except Exception as e:
                self.logger.error(f"Error evaluating row {index}: {e}\nPrediction was: {prediction_str}")
                eval_results.append({'reality': 0, 'aesthetics': 0, 'consistency': 0})

        if not eval_results:
            self.logger.warning("No evaluation results were generated.")
            return pd.DataFrame()

        scores_df = pd.DataFrame(eval_results)
        avg_scores = scores_df.mean()
        avg_scores = avg_scores.fillna(0)
        overall_score = avg_scores.mean()

        report = pd.DataFrame({
            'split': ['none'],
            'Overall': [overall_score * 100],
            'Reality_Assessment': [avg_scores.get('reality', 0) * 100],
            'Aesthetics_Rating': [avg_scores.get('aesthetics', 0) * 100],
            'Instruction_Consistency': [avg_scores.get('consistency', 0) * 100]
        })

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
        """
        Parses the prediction string from the judge model (e.g., GPT-4).
        This function is designed to be robust against minor formatting variations.
        """
        scores = {}
        # Key Change: For Reality, look for "Total Score:" as produced by the judge.
        reality_match = re.search(r"Total Score:[\s\*]*([\d.]+)", pred_str, re.IGNORECASE)
        if reality_match:
            scores['reality'] = float(reality_match.group(1))

        # For Aesthetics, look for the title and then any number nearby, even across newlines.
        # \D* matches any non-digit characters (including newlines and symbols).
        aesthetics_match = re.search(r"Aesthetics Rating\D*([\d.]+)", pred_str, re.IGNORECASE)
        if aesthetics_match:
            scores['aesthetics'] = float(aesthetics_match.group(1))

        # For Consistency, do the same.
        consistency_match = re.search(r"Instruction Consistency Rating\D*([\d.]+)", pred_str, re.IGNORECASE)
        if consistency_match:
            scores['consistency'] = float(consistency_match.group(1))

        # Returns the dictionary. If any key is missing, the evaluate function will handle it.
        return scores if scores else None
