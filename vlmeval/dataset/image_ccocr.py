# flake8: noqa

import os
import re
import tempfile
import json
from functools import partial
import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *
from ..smp.file import get_intermediate_file_path

# should be the same as  FAIL_MSG definded in vlmeval/inference.py
FAIL_MSG = 'Failed to obtain answer via API.'


class CCOCRDataset(ImageBaseDataset):
    TYPE = 'VQA'
    # define data path
    DATASET_URL = {
        "CCOCR": "https://opencompass.openxlab.space/utils/VLMEval/CCOCR.tsv"
    }
    DATASET_MD5 = {
        "CCOCR": "f8927b76510ffe04e59d45e3f8e8b620"
    }
    JUDGE_FORMAT = '{model_name}_{dataset_name}_comprehensive_eval.json'
    RATING_FORMAT = '{model_name}_{dataset_name}_score.json'

    def _evaluate_single_dataset(self, sub_df, data_name):
        """
        Evaluate a single sub-dataset from the combined CCOCR tsv
        """
        dict_list = sub_df.to_dict(orient='records')

        gt_info, ptd_info = {}, {}
        for data_info in dict_list:
            image_name = data_info['image_name']
            gt_info[image_name] = data_info['answer']

            # warning the FAIL samples
            if data_info['prediction'] != FAIL_MSG:
                ptd_info[image_name] = data_info['prediction']

        # Extract metadata from the sub-dataset
        group_name = str(sub_df['category'].iloc[0])
        op_name = str(sub_df['l2-category'].iloc[0])

        data_info = {"op": op_name, "group": group_name, "dataset": data_name, "num": len(gt_info)}

        try:
            from .utils.ccocr_evaluator import evaluator_map_info as ccocr_evaluator_map
        except ImportError as err:
            import warnings
            warnings.warn('The dependency of CCOCR evaluator is not properly installed')
            warnings.warn(f'{type(err)}: {err}')
            return None, None

        eval_func = ccocr_evaluator_map.get(group_name, None)
        if eval_func is None:
            print(f"Warning: evaluator not defined for: {group_name}")
            return None, None

        meta_info, eval_info = eval_func(ptd_info, gt_info, **data_info)

        return {"meta": meta_info, "evaluation": eval_info, "config": data_info}, eval_info.get("summary")

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate the combined CCOCR dataset containing all sub-datasets
        """
        df = load(eval_file)
        df['prediction'] = [str(x) for x in df['prediction']]
        required_colume_list = ['answer', 'prediction', "category", "image_name", "l2-category", "split"]
        for required_colume in required_colume_list:
            assert required_colume in df, "required_colume: {} NOT found".format(required_colume)

        # Create unique sub-dataset identifiers using category, l2-category, and split
        df['sub_dataset_id'] = df['category'].astype(str) + '_' + df['l2-category'].astype(str) + '_' + df['split'].astype(str)

        # Get all unique sub-datasets from the combined identifier
        unique_sub_datasets = df['sub_dataset_id'].unique()

        all_results = {}
        all_summaries = {}

        # Process each sub-dataset separately
        for sub_dataset_id in tqdm(unique_sub_datasets, desc="Processing sub-datasets"):
            print(f"Processing sub-dataset: {sub_dataset_id}")

            # Filter data for this specific sub-dataset
            sub_df = df[df['sub_dataset_id'] == sub_dataset_id].copy()

            if len(sub_df) == 0:
                print(f"Warning: No data found for sub-dataset: {sub_dataset_id}")
                continue

            # Get the original split name for compatibility (use the split value)
            split_name = sub_df['split'].iloc[0]

            # Evaluate this sub-dataset
            result_info, summary = self._evaluate_single_dataset(sub_df, split_name)

            if result_info is not None:
                all_results[sub_dataset_id] = result_info
                all_summaries[sub_dataset_id] = summary
                print(f"Completed evaluation for {sub_dataset_id}: {summary}")
            else:
                print(f"Failed to evaluate {sub_dataset_id}")

        # Save comprehensive results
        judge_file = self.get_judge_file_path(eval_file)
        comprehensive_result = {
            "meta": {"total_datasets": len(all_results), "datasets": list(all_results.keys())},
            "results": all_results,
            "summaries": all_summaries
        }
        dump(comprehensive_result, judge_file)
        print(f"Comprehensive results saved to: {judge_file}")

        # Final Aggregation Logic
        lan_ocr_scores = []
        scene_ocr_scores = []
        kie_scores = []
        doc_parsing_scores = []

        for key, summary in all_summaries.items():
            if not isinstance(summary, dict):
                continue

            if 'lan_ocr' in key:
                if 'macro_f1_score' in summary:
                    lan_ocr_scores.append(summary['macro_f1_score'])
            elif 'scene_ocr' in key:
                if 'macro_f1_score' in summary:
                    scene_ocr_scores.append(summary['macro_f1_score'])
            elif 'kie' in key:
                if 'acc' in summary:
                    kie_scores.append(summary['acc'])
            elif 'doc_parsing' in key:
                if 'score' in summary:
                    doc_parsing_scores.append(summary['score'])

        res = {}
        category_averages = []

        if lan_ocr_scores:
            avg = sum(lan_ocr_scores) / len(lan_ocr_scores)
            res['lan_ocr'] = avg
            category_averages.append(avg)

        if scene_ocr_scores:
            avg = sum(scene_ocr_scores) / len(scene_ocr_scores)
            res['scene_ocr'] = avg
            category_averages.append(avg)

        if kie_scores:
            avg = sum(kie_scores) / len(kie_scores)
            res['kie'] = avg
            category_averages.append(avg)

        if doc_parsing_scores:
            avg = sum(doc_parsing_scores) / len(doc_parsing_scores)
            res['doc_parsing'] = avg
            category_averages.append(avg)

        if category_averages:
            res['Overall'] = sum(category_averages) / len(category_averages)
        else:
            res['Overall'] = 0

        print("\n" + "="*80)
        print("Final Aggregated Results:")
        print("="*80)
        for k, v in res.items():
            print(f"  {k.upper():<20}: {v:.4f}")
        print("="*80)
        rating_file = self.get_rating_file_path(eval_file)
        dump(res, rating_file)
        return res
