import json
import os.path as osp
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import (
    dump,
    get_cache_path,
    get_file_extension,
    get_intermediate_file_path,
    load,
)

from .utils.multiple_choice import extract_characters_regex
from .video_base import VideoBaseDataset

CHOICES = ("A", "B", "C", "D")

SPATIAL_COGNITION_TASKS = {
    "object_existence",
    "object_attribute",
    "relative_direction",
    "landmark_appearance_order",
    "landmark_recall",
    "positional_relationship",
    "spatial_consistency",
    "spatio-temporal_consistency",
}

SELF_AWARENESS_TASKS = {
    "action_recognition",
    "action_sequence",
    "action_recall",
    "action_prediction",
    "path_planning",
}

ALL_TASKS = SPATIAL_COGNITION_TASKS | SELF_AWARENESS_TASKS

TSV_COLUMNS = [
    "index",
    "question_id",
    "video",
    "concat_num",
    "task_type",
    "question",
    "A",
    "B",
    "C",
    "D",
    "answer",
]


class SISBench(VideoBaseDataset):
    """SIS-Bench: spatial intelligence evaluation in UAV videos."""

    TYPE = "Video-MCQ"

    def __init__(self, dataset="SIS-Bench", nframe=32, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ["SIS-Bench"]

    @staticmethod
    def _safe_video_name(video_name):
        path = Path(str(video_name))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid SIS-Bench video path: {video_name!r}")
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"SIS-Bench video must be an MP4 file: {video_name!r}")
        return path.as_posix()

    @classmethod
    def _dimension_for_task(cls, task_type):
        if task_type in SPATIAL_COGNITION_TASKS:
            return "spatial_cognition"
        if task_type in SELF_AWARENESS_TASKS:
            return "self_awareness"
        raise ValueError(f"Unknown SIS-Bench task type: {task_type!r}")

    @classmethod
    def _check_integrity(cls, dataset_path, dataset_name):
        data_file = osp.join(dataset_path, f"{dataset_name}.tsv")
        video_root = osp.join(dataset_path, "video")
        if not osp.isfile(data_file) or not osp.isdir(video_root):
            return False

        try:
            data = load(data_file)
        except Exception:
            return False

        if any(column not in data for column in TSV_COLUMNS):
            return False
        if not data["question_id"].is_unique or not data["index"].is_unique:
            return False
        if not set(data["answer"].astype(str).str.upper()).issubset(CHOICES):
            return False
        if not set(data["task_type"]).issubset(ALL_TASKS):
            return False

        for video in data["video"].astype(str).unique():
            if not osp.isfile(osp.join(video_root, video + ".mp4")):
                return False
        return True

    @classmethod
    def _generate_tsv(cls, dataset_path, dataset_name):
        jsonl_path = osp.join(dataset_path, "SIS-Bench.jsonl")
        data_file = osp.join(dataset_path, f"{dataset_name}.tsv")
        video_root = osp.join(dataset_path, "video")
        if not osp.isfile(jsonl_path):
            raise FileNotFoundError(f"SIS-Bench annotations not found: {jsonl_path}")
        if not osp.isdir(video_root):
            raise FileNotFoundError(
                f"SIS-Bench video directory not found: {video_root}"
            )

        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for index, raw_line in enumerate(file):
                item = json.loads(raw_line)
                video_name = cls._safe_video_name(item["video_name"])
                options = item["options"]
                if set(options) != set(CHOICES):
                    raise ValueError(
                        f"Expected A-D options for {item['question_id']}, got {sorted(options)}"
                    )
                answer = str(item["answer"]).strip().upper()
                if answer not in CHOICES:
                    raise ValueError(
                        f"Invalid answer for {item['question_id']}: {item['answer']!r}"
                    )
                task_type = item["task_type"]
                if task_type not in ALL_TASKS:
                    raise ValueError(f"Unknown SIS-Bench task type: {task_type!r}")

                video = osp.splitext(video_name)[0]
                video_path = osp.join(video_root, video_name)
                if not osp.isfile(video_path):
                    raise FileNotFoundError(f"SIS-Bench video not found: {video_path}")
                rows.append(
                    {
                        "index": index,
                        "question_id": item["question_id"],
                        "video": video,
                        "concat_num": item["concat_num"],
                        "task_type": task_type,
                        "question": item["question"],
                        "A": options["A"],
                        "B": options["B"],
                        "C": options["C"],
                        "D": options["D"],
                        "answer": answer,
                    }
                )

        data = pd.DataFrame(rows, columns=TSV_COLUMNS)
        data.to_csv(data_file, sep="\t", index=False)
        return data_file

    def prepare_dataset(self, dataset_name="SIS-Bench", repo_id="choucsan/SIS-Bench"):
        dataset_path = get_cache_path(repo_id)
        if dataset_path is None:
            dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

        if not self._check_integrity(dataset_path, dataset_name):
            self._generate_tsv(dataset_path, dataset_name)
            if not self._check_integrity(dataset_path, dataset_name):
                raise RuntimeError(
                    "Generated SIS-Bench metadata failed integrity checks"
                )

        return {
            "root": osp.join(dataset_path, "video"),
            "data_file": osp.join(dataset_path, f"{dataset_name}.tsv"),
        }

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        prompt = (
            f"{line['question']}\n"
            "Options:\n"
            f"(A) {line['A']}\n"
            f"(B) {line['B']}\n"
            f"(C) {line['C']}\n"
            f"(D) {line['D']}\n"
            "Answer with only the letter of the correct option."
        )
        message = []
        if video_llm:
            video_path = osp.join(self.data_root, line["video"] + ".mp4")
            message.append(dict(type="video", value=video_path))
        else:
            if not self.split_frame:
                raise ValueError(
                    "SIS-Bench frame inference requires nframe or fps to be set."
                )
            frames = self.save_video_frames(line["video"])
            message.extend(dict(type="image", value=frame) for frame in frames)
        message.append(dict(type="text", value=prompt))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        del judge_kwargs
        assert get_file_extension(eval_file) in ["xlsx", "json", "tsv"], (
            "data file should be a supported format (xlsx/json/tsv) file"
        )

        data = load(eval_file)
        required_columns = {"prediction", "answer", "task_type"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(
                f"SIS-Bench prediction file is missing columns: {sorted(missing)}"
            )

        dimensions = []
        parsed_predictions = []
        scores = []
        for _, row in data.iterrows():
            task_type = str(row["task_type"])
            dimension = cls._dimension_for_task(task_type)
            prediction = "" if pd.isna(row["prediction"]) else str(row["prediction"])
            parsed_prediction = extract_characters_regex(
                prediction, choices=["(A)", "(B)", "(C)", "(D)"]
            )
            answer = str(row["answer"]).strip().upper()
            dimensions.append(dimension)
            parsed_predictions.append(parsed_prediction)
            scores.append(int(parsed_prediction == answer))

        data["parsed_prediction"] = parsed_predictions
        data["dimension"] = dimensions
        data["score"] = scores
        score_file = get_intermediate_file_path(eval_file, "_score")
        dump(data, score_file)

        def accuracy(mask):
            selected = data.loc[mask, "score"]
            return 100.0 * selected.mean() if len(selected) else 0.0

        result = {
            "overall_accuracy": accuracy(pd.Series(True, index=data.index)),
            "spatial_cognition_accuracy": accuracy(
                data["dimension"] == "spatial_cognition"
            ),
            "self_awareness_accuracy": accuracy(data["dimension"] == "self_awareness"),
        }
        for task_type in sorted(ALL_TASKS):
            metric_name = f"sis_bench_{task_type.replace('-', '_')}_accuracy"
            result[metric_name] = accuracy(data["task_type"] == task_type)

        rating_file = get_intermediate_file_path(eval_file, "_rating", "json")
        dump(result, rating_file)
        return result
