import json
import argparse
import nltk
from nltk.metrics import precision, recall, f_measure
import numpy as np
import jieba
import re
from nltk.translate import meteor_score
from typing import List, Dict, Any


def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))


def cal_per_metrics(pred, gt):
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics


def compute_ocr_metrics(pred_text: str, gt_text: str) -> Dict[str, float]:
    """Compute OCR metrics between predicted and ground truth text."""
    if not pred_text or not gt_text:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    # Convert texts to character lists
    pred_chars = list(pred_text)
    gt_chars = list(gt_text)

    # Compute confusion matrix
    tp = sum(1 for p, g in zip(pred_chars, gt_chars) if p == g)
    fp = len(pred_chars) - tp
    fn = len(gt_chars) - tp

    # Compute metrics
    accuracy = tp / len(gt_chars) if gt_chars else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def evaluate_ocr(pred_texts: List[str], gt_texts: List[str]) -> Dict[str, float]:
    """Evaluate OCR results using multiple metrics."""
    if len(pred_texts) != len(gt_texts):
        raise ValueError("Number of predicted and ground truth texts must be equal")

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }

    for pred_text, gt_text in zip(pred_texts, gt_texts):
        result = compute_ocr_metrics(pred_text, gt_text)
        for metric in metrics:
            metrics[metric].append(result[metric])

    return {
        metric: np.mean(values) for metric, values in metrics.items()
    }


if __name__ == "__main__":
    # Examples for region text recognition and read all text tasks
    predict_text = "metrics['edit_dist'] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"
    true_text = "metrics = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"

    scores = cal_per_metrics(predict_text, true_text)

    predict_text = "metrics['edit_dist'] len(gt))"
    true_text = "metrics = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"

    scores = cal_per_metrics(predict_text, true_text)
    print(scores)
