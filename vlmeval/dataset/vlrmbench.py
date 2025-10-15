from ast import literal_eval
from collections import defaultdict
import re
import numpy as np
from sklearn.metrics import f1_score

from .image_base import ImageBaseDataset
from ..smp import *
from ..smp.file import get_intermediate_file_path


def format_model_answer_tolist(model_answer, task_gt):
    """
    Extract 0/1 list from model answer

    Args:
        model_answer: Model's prediction answer (string)
        task_gt: Ground truth list, used to determine expected length

    Returns:
        list: 0/1 list
    """
    numbers = re.findall(r'\d+', str(model_answer))

    result = [int(num) for num in numbers]

    # Convert non-0/1 numbers to 1
    result = [num if num == 0 or num == 1 else 1 for num in result]

    # Adjust length to match task_gt
    if len(result) >= len(task_gt):
        return result[:len(task_gt)]
    else:
        return result + [0] * (len(task_gt) - len(result))


def get_F1Score(gathered_model_answer, gathered_task_gt):
    """
    Calculate F1 score

    Args:
        gathered_model_answer: List of all model answers
        gathered_task_gt: List of all ground truth

    Returns:
        tuple: (F1_pos, F1_neg, F1_w) - positive class F1, negative class F1, weighted F1
    """
    model_answer = np.array(gathered_model_answer)
    task_gt = np.array(gathered_task_gt)

    pos_count = np.sum(task_gt == 1)
    neg_count = np.sum(task_gt == 0)

    F1_pos = f1_score(task_gt, model_answer, pos_label=1, zero_division=0)
    F1_neg = f1_score(task_gt, model_answer, pos_label=0, zero_division=0)

    w_pos = neg_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
    w_neg = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0

    F1_w = w_neg * F1_neg + w_pos * F1_pos

    return F1_pos, F1_neg, F1_w


class VLRMBench(ImageBaseDataset):
    """
    VLRMBench Dataset - Visual Language Reasoning Model Benchmark

    A comprehensive benchmark for evaluating visual reasoning capabilities including:
    - step_correctness: Step correctness detection
    - redundant_det: Redundancy detection
    - most_confidence: Highest confidence judgment
    - attribute_hallucination: Attribute hallucination detection
    - existence_hallucination: Existence hallucination detection
    - detail_error: Detail error detection
    - image_ref_error: Image reference error detection
    - location_error: Location error detection
    """

    TYPE = 'VQA'
    DATASET_URL = {
        'VLRMBench': 'https://huggingface.co/datasets/Winston-Yuan/VLRMBench/resolve/main/VLRMBench.tsv?download=true'
    }
    DATASET_MD5 = {
        'VLRMBench': 'f1dedeac74fc1112545390d6e2ecf4a2'
    }

    def build_prompt(self, line):
        """
        Build prompt information

        Args:
            line: Data row, can be int index or pd.Series

        Returns:
            list: Multimodal message list, format is [dict(type='image', value=path), dict(type='text', value=text),]
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Create a copy of line to avoid SettingWithCopyWarning
        line = line.copy()

        # Use parent class method to save image (decode from base64 and save locally)
        tgt_path = self.dump_image(line)

        # Build messages
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        # Add question text
        question = line.get('question', '')
        if question:
            msgs.append(dict(type='text', value=question))

        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluate model prediction results

        Args:
            eval_file: Path to model prediction results file
            **judge_kwargs: Other evaluation parameters

        Returns:
            pd.DataFrame: Evaluation results, including F1 scores for each category
        """
        # Load prediction data
        data = load(eval_file)

        # Ensure necessary fields exist
        assert 'answer' in data.columns, "Evaluation file missing 'answer' field"
        assert 'prediction' in data.columns, "Evaluation file missing 'prediction' field"
        assert 'category' in data.columns, "Evaluation file missing 'category' field"

        # Collect model answers and ground truth by category
        category_model_answers = defaultdict(list)
        category_task_gts = defaultdict(list)
        category_total = defaultdict(int)

        for idx in range(len(data)):
            item = data.iloc[idx]
            category = item['category']

            try:
                # Parse task_gt (answer field)
                task_gt = item['answer']
                if isinstance(task_gt, str):
                    # Try to parse string as list
                    task_gt = literal_eval(task_gt)

                # Get model answer (prediction field)
                model_answer = item.get('prediction', '')

                # Format model answer using format_model_answer_tolist
                formatted_model_answer = format_model_answer_tolist(model_answer, task_gt)

                # Collect answers for each category
                category_task_gts[category].extend(task_gt)
                category_model_answers[category].extend(formatted_model_answer)
                category_total[category] += 1
            except Exception as e:
                # If parsing fails, log and skip the sample
                print(f"Failed to process sample (idx={idx}, category={category}): {e}")
                continue

        # Calculate F1 scores for each category
        results = {}
        for category in category_task_gts:
            gathered_task_gt = category_task_gts[category]
            gathered_model_answer = category_model_answers[category]

            if len(gathered_task_gt) > 0:
                F1_pos, F1_neg, F1_w = get_F1Score(gathered_model_answer, gathered_task_gt)

                results[f'{category}_F1_pos'] = F1_pos
                results[f'{category}_F1_neg'] = F1_neg
                results[f'{category}_F1_weighted'] = F1_w
                results[f'{category}_count'] = category_total[category]
            else:
                results[f'{category}_F1_pos'] = 0.0
                results[f'{category}_F1_neg'] = 0.0
                results[f'{category}_F1_weighted'] = 0.0
                results[f'{category}_count'] = 0

        # Calculate overall F1 score (all categories combined)
        all_task_gts = []
        all_model_answers = []
        for category in category_task_gts:
            all_task_gts.extend(category_task_gts[category])
            all_model_answers.extend(category_model_answers[category])

        if len(all_task_gts) > 0:
            F1_pos_overall, F1_neg_overall, F1_w_overall = get_F1Score(all_model_answers, all_task_gts)
            results['Overall_F1_pos'] = F1_pos_overall
            results['Overall_F1_neg'] = F1_neg_overall
            results['Overall_F1_weighted'] = F1_w_overall
            results['Overall_count'] = sum(category_total.values())
        else:
            results['Overall_F1_pos'] = 0.0
            results['Overall_F1_neg'] = 0.0
            results['Overall_F1_weighted'] = 0.0
            results['Overall_count'] = 0

        # Convert to DataFrame format
        results_df = pd.DataFrame([results])

        # Save results
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(results_df, score_file)

        return results_df
