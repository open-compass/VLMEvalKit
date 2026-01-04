from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import jieba
import nltk
from nltk.metrics import precision, recall, f_measure
from nltk.translate import meteor_score
from rouge import Rouge

from .image_base import ImageBaseDataset
from ..smp import *


def _contain_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", text))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _cal_ocr_metrics(pred: str, gt: str) -> Dict[str, float]:
    if _contain_chinese(gt) or _contain_chinese(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics = {
        "bleu": nltk.translate.bleu([reference], hypothesis),
        "meteor": meteor_score.meteor_score([reference], hypothesis),
    }

    reference_set = set(reference)
    hypothesis_set = set(hypothesis)
    metrics["f_measure"] = f_measure(reference_set, hypothesis_set)
    metrics["precision"] = precision(reference_set, hypothesis_set)
    metrics["recall"] = recall(reference_set, hypothesis_set)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt), 1)
    return metrics


def _cal_summary_metrics(pred: str, gt: str) -> Dict[str, float]:
    rouge = Rouge()
    res = rouge.get_scores(hyps=" ".join(pred.split()), refs=" ".join(gt.split()))
    rl = res[0]["rouge-l"]
    return {"rouge_l_r": rl["r"], "rouge_l_p": rl["p"], "rouge_l_f": rl["f"]}


def _cal_qa_metric(pred: str, gt: str) -> Dict[str, float]:
    return {"accuracy": 1.0 if pred == gt else 0.0}


def _classify(question: str) -> str:
    q = _normalize_question(question)
    q_lower = q.lower()
    if q_lower.startswith("which page's box contains more characters?"):
        return "qa"
    if q_lower.startswith("what is this in the box"):
        return "qa"
    if q_lower.startswith("summarize the text in the region"):
        return "summary"
    if q_lower.startswith("translate the content for the area"):
        return "translation"
    return "ocr"


def _normalize_question(question: str) -> str:
    if question is None:
        return ""
    parts = [segment.strip() for segment in str(question).strip().split("\n") if segment.strip()]
    return parts[-1] if parts else ""


class FoxBench(ImageBaseDataset):

    TYPE = "QA"
    MODALITY = "IMAGE"
    DATASET_URL = {'FoxBench':'https://huggingface.co/datasets/EasonFan/fox_benchmark.tsv/resolve/main/fox_benchmark_data.tsv'}  # noqa: E501
    DATASET_MD5 = {'FoxBench': '153b88812ad58495558326bab9c90f8e'}

    def load_data(self, dataset):
        repo_root = Path(__file__).resolve().parents[4]
        candidates = [repo_root / "fox_benchmark_data.tsv", repo_root / "fox_bench.tsv"]
        for path in candidates:
            if path.exists():
                self.data_path = str(path)
                return load(path)

        data = super().load_data(dataset)
        self.data_path = getattr(self, "data_path", None)
        return data

    def build_prompt(self, line):
        return super().build_prompt(line)

    def evaluate(self, eval_file, **judge_kwargs):
        preds = load(eval_file)
        assert "prediction" in preds, "Prediction file must contain a `prediction` column"

        if not hasattr(self, "data_path") or self.data_path is None:
            self.load_data(self.dataset_name)
        refs = load(self.data_path)

        ref_map = {str(idx): (q, ans) for idx, q, ans in zip(refs["index"], refs["question"], refs["answer"])}

        ocr_metrics: List[Dict[str, float]] = []
        translation_metrics: List[Dict[str, float]] = []
        summary_metrics: List[Dict[str, float]] = []
        qa_metrics: List[Dict[str, float]] = []
        details: List[Dict[str, object]] = []

        for idx, pred in zip(preds["index"], preds["prediction"]):
            idx_str = str(idx)
            if idx_str not in ref_map:
                continue
            question, gt = ref_map[idx_str]
            normalized_question = _normalize_question(question)
            pred_text = "" if pred is None else str(pred)
            task = _classify(normalized_question)

            if task == "qa":
                m = _cal_qa_metric(pred_text, gt)
                qa_metrics.append(m)
            elif task == "summary":
                m = _cal_summary_metrics(pred_text, gt)
                summary_metrics.append(m)
            elif task == "translation":
                m = _cal_ocr_metrics(pred_text, gt)
                translation_metrics.append(m)
            else:
                m = _cal_ocr_metrics(pred_text, gt)
                ocr_metrics.append(m)

            details.append({
                "index": idx_str,
                "task": task,
                "question": normalized_question,
                "gt": gt,
                "pred": pred_text,
                **m,
            })

        def _avg(dicts: List[Dict[str, float]]) -> Dict[str, float]:
            if not dicts:
                return {}
            keys = dicts[0].keys()
            return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

        scores = {
            "ocr": _avg(ocr_metrics),
            "translation": _avg(translation_metrics),
            "summary": _avg(summary_metrics),
            "qa": _avg(qa_metrics),
            "counts": {
                "ocr": len(ocr_metrics),
                "translation": len(translation_metrics),
                "summary": len(summary_metrics),
                "qa": len(qa_metrics),
                "total": len(details),
            },
        }

        score_file = get_intermediate_file_path(eval_file, "_foxbench_score", "json")
        dump(scores, score_file)
        detail_file = get_intermediate_file_path(eval_file, "_foxbench_detail", "json")
        dump(details, detail_file)

        print("FoxBench metrics (avg):", json.dumps(scores, ensure_ascii=False, indent=2))
        print(f"FoxBench score file: {score_file}")
        print(f"FoxBench detail file: {detail_file}")

        return scores
