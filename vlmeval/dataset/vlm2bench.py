# flake8: noqa
import os
import pandas as pd
from .image_base import ImageBaseDataset
from .utils.vlm2bench import (
    common_process_results,
    tf_pair_aggregate_accuracy,
    cnt_aggregate_metric,
    grp_aggregate_accuracy,
)
from ..smp import *
from ..smp.file import get_intermediate_file_path


class VLM2Bench(ImageBaseDataset):
    TYPE = "VQA"

    DATASET_URL = {
        "VLM2Bench": 'https://huggingface.co/datasets/Sterzhang/vlm2-bench/resolve/main/VLM2Bench_img.tsv' # all 2860 image cases from VLM2Bench huggingface repo
    }
    # DATASET_MD5
    DATASET_MD5 = {'VLM2Bench': '16f474bfc4e269c583468bf89139da8f'}
    RATING_FORMAT = '{model_name}_{dataset_name}_score.csv'

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
        data = load(eval_file)
        results = data.to_dict(orient="records")
        processed = common_process_results(results)

        # For cnt category, calculate image_seq_len (i.e., number of images) based on the list of image encodings stored in the image field
        for rec in processed:
            if rec.get("category", "").lower() in ["oc-cnt", "pc-cnt"]:
                try:
                    rec["image_seq_len"] = len(rec["image"])
                except Exception as e:
                    rec["image_seq_len"] = 2

        eval_stats = {}
        for cat in sorted(set([r["category"] for r in processed])):
            sub_results = [r for r in processed if r["category"] == cat]
            if cat in ["gc-mat", "gc-trk", "oc-cpr", "pc-cpr"]:
                corr, tot = tf_pair_aggregate_accuracy(sub_results)
            elif cat in ["oc-cnt", "pc-cnt"]:
                corr, tot = cnt_aggregate_metric(sub_results)
            elif cat in ["oc-grp", "pc-grp"]:
                corr, tot = grp_aggregate_accuracy(sub_results)
            else:
                raise NotImplementedError(f"Category {cat} is not implemented.")
            eval_stats[cat] = (corr, tot)
        eval_scores = {k: v[0] / v[1] * 100 if v[1] > 0 else 0 for k, v in eval_stats.items()}
        keys = ["gc-mat", "gc-trk", "oc-cpr", "pc-cpr", "oc-cnt", "pc-cnt", "oc-grp", "pc-grp"]
        eval_scores["micro.avg"] = sum([eval_stats[k][0] for k in keys]) / sum([eval_stats[k][1] for k in keys]) * 100
        eval_scores["macro.avg"] = np.mean([eval_scores[k] for k in keys])

        score_df = pd.DataFrame({k: [v] for k, v in eval_scores.items()})
        final_score_file = get_intermediate_file_path(eval_file, "_score", "csv")
        dump(score_df, final_score_file)
        return score_df

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        rating_file = cls.RATING_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
        rating_file = osp.join(root, rating_file)
        rating = load(rating_file)
        rating = {k: rating.iloc[0][k] for k in rating.columns}
        res = {'overall': (rating['macro.avg'] + rating['micro.avg']) / 2}
        if verbose:
            res['rating'] = rating
        return res
