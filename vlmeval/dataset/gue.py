import warnings
import pandas as pd
import re
from abc import abstractmethod
from sklearn.metrics import matthews_corrcoef
from ..smp import *
from .text_base import TextBaseDataset


def remove_think_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "<think>" not in text:
        return text
    if "</think>" not in text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def GUE_postprocessor(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = remove_think_tags(text)
    if text == "":
        return ""

    match = re.search(r'\bThe prediction result is\s+(positive|negative)\b', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    positive_patterns = [
        r'\bpositive\b', r'\bpositively\b', r'\bpresence\b', r'\bdetected\b',
        r'\bidentified\b', r'\bidentifiable\b', r'\bfound\b', r'\byes\b',
        r'\blocated\b', r'\bdetectable\b', r'\bobservable\b', r'\bevident\b',
        r'\babsolutely\b', r'\baffirmative\b', r'\bcan\b', r'\baffirm\b',
        r'\bconfirm\b', r'\bconfirms\b', r'\breveals\b', r'\bexistence\b',
        r'\bcertainly\b', r'\bconsistent\b', r'\brecognizable\b',
        r'\bshows core\b', r'\bshows promoter\b', r'\bshows characteristic\b',
        r'\bevidenced by\b', r'\bseeing characteristic patterns\b',
        r'\bincludes\b', r'\bcontains sequences\b', r'\bexhibits clear\b',
        r'\bcontains transcription\b', r'\bexhibits sequences\b',
        r'\bclearly contains\b', r'\brecognized\b', r'\bexhibits features\b',
        r'\bcontains regulatory\b', r'\bshows clear\b', r'\bdisplays\b',
        r'\bdefinitely has\b', r'\bexhibits patterns\b', r'\bclear evidence\b',
        r'\bcontains a\b', r'\byep\b', r'\bcontains sites\b',
        r'\bshows sequences\b'
    ]

    negative_patterns = [
        r'\bnegative\b', r'\bno\b', r'\babsence\b', r'\bnot\b',
        r'\bcannot\b', r'\bfails\b', r'\babsent\b', r'\blacks\b'
    ]

    for pattern in negative_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "negative"

    for pattern in positive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "positive"

    return ""

data_path = "~/LMUData"

GUE_sub_tasks = {
        'cpd-prom_core_all': 'cpd-prom_core_all.tsv',
        'cpd-prom_core_notata': 'cpd-prom_core_notata.tsv',
        'cpd-prom_core_tata': 'cpd-prom_core_tata.tsv',
        'pd-prom_300_all': 'pd-prom_300_all.tsv',
        'pd-prom_300_notata': 'pd-prom_300_notata.tsv',
        'pd-prom_300_tata': 'pd-prom_300_tata.tsv',
        'tf-h-0': 'tf-h-0.tsv',
        'tf-h-1': 'tf-h-1.tsv',
        'tf-h-2': 'tf-h-2.tsv',
        'tf-h-3': 'tf-h-3.tsv',
        'tf-h-4': 'tf-h-4.tsv',
}

class GUE(TextBaseDataset):
    TYPE = 'TEXT'

    DATASET_URL = {
        task_name: f"{data_path}/{file_name}"
        for task_name, file_name in GUE_sub_tasks.items()
    }

    DATASET_MD5 = {task_name: "" for task_name, _ in GUE_sub_tasks.items()}  # MD5 暂空

    @staticmethod
    def score(predictions, references):
        def normalize(label):
            if not isinstance(label, str):
                return None
            label = label.strip().lower()
            if label == 'positive':
                return 1
            elif label == 'negative':
                return 0
            return None

        if isinstance(predictions[0], list):
            predictions = [p[0] for p in predictions]

        pred_bin_all = [1 if str(p).strip().lower() == 'positive' else 0 for p in predictions]
        ref_bin_all = [1 if str(r).strip().lower() == 'positive' else 0 for r in references]
        mcc_all = matthews_corrcoef(ref_bin_all, pred_bin_all)

        filtered_pred, filtered_ref = [], []
        skipped = 0
        for p, r in zip(predictions, references):
            p_norm = normalize(p)
            r_norm = normalize(r)
            if p_norm is None or r_norm is None:
                skipped += 1
                continue
            filtered_pred.append(p_norm)
            filtered_ref.append(r_norm)

        mcc_filtered = matthews_corrcoef(filtered_ref, filtered_pred) if filtered_pred else 0.0

        return {
            'matthews_correlation_all': mcc_all * 100,
            'matthews_correlation_filtered': mcc_filtered * 100,
            'non_pos_neg_count': skipped,
            'total_count': len(predictions)
        }

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = data[~pd.isna(data["prediction"])]
        assert 'category' in data and 'prediction' in data, "eval file must contain 'category' and 'prediction' columns"

        predictions = [GUE_postprocessor(p) for p in data['prediction']]
        references = [r.strip().lower() for r in data['category']]

        result = cls.score(predictions, references)

        df_result = pd.DataFrame([{
            'matthews_correlation_all': result['matthews_correlation_all'],
            'matthews_correlation_filtered': result['matthews_correlation_filtered'],
            'non_pos_neg_count': result['non_pos_neg_count'],
            'total_count': result['total_count']
        }])

        return df_result
