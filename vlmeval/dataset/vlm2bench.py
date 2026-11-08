# flake8: noqa
import os

import pandas as pd

from vlmeval.smp import dump, load, toliststr
from vlmeval.smp.file import get_intermediate_file_path
from .image_base import ImageBaseDataset
from .utils.vlm2bench import (cnt_aggregate_metric, common_process_results, grp_aggregate_accuracy,
                              tf_pair_aggregate_accuracy)


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ''
    if isinstance(value, (list, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _positive_int(value):
    if _is_missing(value):
        return None
    try:
        count = int(value)
    except Exception:
        return None
    return count if count > 0 else None


def _field_seq_len(value):
    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple)):
        count = len(value)
    else:
        try:
            count = len(toliststr(value))
        except Exception:
            return None
    return count if count > 0 else None


def _infer_image_seq_len(record, default=2):
    seq_len = _positive_int(record.get('image_seq_len'))
    if seq_len is not None:
        return seq_len
    for key in ('image_path', 'image'):
        if key not in record:
            continue
        seq_len = _field_seq_len(record[key])
        if seq_len is not None:
            return seq_len
    return default


class VLM2Bench(ImageBaseDataset):
    TYPE = "VQA"

    DATASET_URL = {
        "VLM2Bench": 'https://huggingface.co/datasets/Sterzhang/vlm2-bench/resolve/main/VLM2Bench_img.tsv' # all 2860 image cases from VLM2Bench huggingface repo
    }
    # DATASET_MD5
    DATASET_MD5 = {'VLM2Bench': '16f474bfc4e269c583468bf89139da8f'}

    def build_prompt(self, line):
        """
        Build multimodal input:
        - If the record does not have "image_path", generate the image_path list based on the "image" field (stored as a regular list of image encodings),
          and update the "image" field to contain a list of multiple image paths.
        - Call dump_image to process the image and image_path fields to obtain all local paths of the images.
        - Construct the text prompt in the format "Question: {question}".
        - Encapsulate all image paths as image messages and append the text message, returning the final multimodal message list.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # If there is no image_path, generate the image_path list based on the image field
        if "image_path" not in line:
            img_field = line.get("image")
            # Assume the image field is already a regular list of image encodings, not a JSON-encoded string
            image_paths = [f"{line['index']}_{i}.jpg" for i in range(len(img_field))]
            line["image_path"] = image_paths
            # Also update the image field to the list of image encodings
            line["image"] = img_field

        # Call dump_image (implemented in the parent class) to process the image and image_path fields, returning the list of local image paths
        img_paths = self.dump_image(line)
        if not isinstance(img_paths, list):
            img_paths = [img_paths]

        # Construct the text prompt (only containing the question)
        prompt = f"Question: {line['question']}\n"

        # Encapsulate all image paths as image messages and append the text message
        msgs = [{"type": "image", "value": p} for p in img_paths]
        msgs.append({"type": "text", "value": prompt})
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluation function:
        - Automatically read the model prediction result file (xlsx or TSV), which contains fields: index, question, answer, category, prediction
        - Directly use the original fields for evaluation without additional conversion;
        - For categories "oc-cnt" or "pc-cnt", calculate image_seq_len based on the "image" field (stored as a regular multi-image encoding)
          and write it into each record;
        - Group by category and use different evaluation functions to calculate metrics for each sub-task:
                • tf pair: suitable for gc-mat, gc-trk, oc-cpr, pc-cpr
                • cnt: suitable for oc-cnt, pc-cnt
                • grp: suitable for oc-grp, pc-grp
        - Write the scores of each sub-task to a CSV file and return a DataFrame.
        """
        model = judge_kwargs.get("model")
        if model:
            storage = get_intermediate_file_path(eval_file, f'_{model}')
            score_file = get_intermediate_file_path(eval_file, f'_{model}_score', 'csv')
            tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
            if os.path.exists(storage):
                data = load(storage)
            else:
                data = load(eval_file)
        else:
            data = load(eval_file)

        results = data.to_dict(orient="records")
        processed = common_process_results(results)

        # For cnt category, calculate image_seq_len (i.e., number of images) from
        # inference/eval records, which usually keep image_path instead of image.
        for rec in processed:
            if rec.get("category", "").lower() in ["oc-cnt", "pc-cnt"]:
                rec["image_seq_len"] = _infer_image_seq_len(rec)

        eval_scores = {}
        for cat in sorted(set([r["category"] for r in processed])):
            sub_results = [r for r in processed if r["category"] == cat]
            if cat in ["gc-mat", "gc-trk", "oc-cpr", "pc-cpr"]:
                score = tf_pair_aggregate_accuracy(sub_results)
            elif cat in ["oc-cnt", "pc-cnt"]:
                score = cnt_aggregate_metric(sub_results)
            elif cat in ["oc-grp", "pc-grp"]:
                score = grp_aggregate_accuracy(sub_results)
            else:
                score = None
            eval_scores[cat] = score

        score_df = pd.DataFrame({k: [v] for k, v in eval_scores.items()})
        if model:
            final_score_file = score_file
        else:
            final_score_file = get_intermediate_file_path(eval_file, "_score", "csv")
        dump(score_df, final_score_file)
        return score_df
