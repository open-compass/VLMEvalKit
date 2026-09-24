import json

import pandas as pd

from vlmeval.smp import load
from ..evaluator import NativeEnd2EndEvaluator


class mdpbench_evaluator:
    def __init__(self, eval_file, tsv_path, **judge_kwargs):
        self.eval_file = eval_file
        self.tsv_path = tsv_path
        self.judge_kwargs = judge_kwargs or {}

        predictions = load(eval_file)
        references = load(tsv_path)

        if 'index' not in predictions.columns or 'index' not in references.columns:
            raise ValueError('Both prediction file and dataset TSV must contain an index column.')

        merged = pd.merge(predictions, references, on='index', how='inner')
        ans_col = 'answer_y' if 'answer_y' in merged.columns else 'answer'
        if ans_col not in merged.columns:
            raise ValueError('Dataset TSV must contain answer column for MDPBench evaluation.')

        if not self._detect_layout_answer(merged, ans_col):
            raise ValueError(
                'MDPBench now only supports layout_dets-based end2end evaluation. '
                'Please provide layout_dets ground truth in answer column.'
            )

    @staticmethod
    def _parse_answer(answer):
        if isinstance(answer, dict):
            return answer
        if isinstance(answer, str):
            try:
                return json.loads(answer)
            except Exception:
                return {'text': answer}
        return {'text': str(answer)}

    @classmethod
    def _detect_layout_answer(cls, merged_df, answer_col):
        for val in merged_df[answer_col].tolist():
            obj = cls._parse_answer(val)
            if isinstance(obj, dict) and isinstance(obj.get('layout_dets'), list):
                return True
        return False

    def score(self):
        match_method = self.judge_kwargs.get('match_method', 'quick_match')
        filter_types = self.judge_kwargs.get('filter_types', None)
        evaluator = NativeEnd2EndEvaluator(
            self.eval_file,
            self.tsv_path,
            match_method=match_method,
            filter_types=filter_types,
        )
        return evaluator.score()


__all__ = ['mdpbench_evaluator']
