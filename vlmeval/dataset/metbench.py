"""MET-Bench datasets for VLMEvalKit."""

import hashlib
import io
import math
from pathlib import Path

import pandas as pd
from datasets import Image as DatasetImage
from datasets import List, load_dataset

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.metbench_core import build_messages, clustered_ratio_interval, score
from vlmeval.smp import LMUDataRoot, dump, load
from vlmeval.smp.file import INFER_FAIL_MSG

EVALUATION_RELEASES = {
    "minecraft": (
        "vanyacohen/MET-Bench-Minecraft",
        "ae0b474dd22986b8b292c2a42f5e81cf8675661e",
    ),
    "chess": ("vanyacohen/MET-Bench-Chess", "6fd525cd537a9c25efe64ae88102c0ab68f94b1d"),
    "shell": ("vanyacohen/MET-Bench-Shell", "75725717b63cf6e54a5b8abd22e6ad96b47a12e3"),
}


class METBenchImage(ImageBaseDataset):
    """Evaluate paired MET-Bench tasks with the released prompts and metrics."""

    TYPE = "VQA"
    MODALITY = "IMAGE"
    modality = "image"
    force_use_dataset_prompt = True

    @classmethod
    def supported_datasets(cls):
        return [
            f"METBench_{domain}_{cls.modality}"
            for domain in ("minecraft", "chess", "shell")
        ]

    def __init__(self, dataset="METBench_minecraft_image", limit=500):
        if dataset not in self.supported_datasets():
            raise ValueError(f"Unsupported MET-Bench task: {dataset}")
        if not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        self.dataset_name = dataset
        self.domain = dataset.split("_")[1]
        self.img_root = str(Path(LMUDataRoot()) / "images" / "METBench")
        self.meta_only = True
        self.skip_noimg = False
        repo, revision = EVALUATION_RELEASES[self.domain]
        config = "evaluation_text_only" if self.modality == "text" else "evaluation"
        data = load_dataset(repo, config, split="test", revision=revision)
        if len(data) != 500:
            raise ValueError("Expected the released 500-example evaluation split")
        if self.modality == "image":
            if self.domain == "minecraft":
                columns = {
                    "image_initial_state": DatasetImage(decode=False),
                    "image_action": DatasetImage(decode=False),
                    "image_candidate_states": List(DatasetImage(decode=False)),
                }
            else:
                columns = {"image_actions": List(DatasetImage(decode=False))}
            for name, feature in columns.items():
                data = data.cast_column(name, feature)
        self.examples = []
        for row in data.select(range(limit)):
            row["metbench_domain"] = self.domain
            row["target"] = row["correct_choice" if self.domain == "minecraft" else "final_state"]
            row["source_row"] = int(row["example_id"].rsplit("-", 1)[1])
            self.examples.append(row)
        self.by_index = {str(row["source_row"]): row for row in self.examples}
        self.data = pd.DataFrame(
            {
                "index": row["source_row"],
                "question": f"MET-Bench {self.domain} ({self.modality})",
                "answer": str(row["target"]),
                "example_id": row["example_id"],
            }
            for row in self.examples
        )

    def build_prompt(self, line):
        """Keep every image in its original position in the user message."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        row = self.by_index[str(line["index"])]
        content = []
        for part in build_messages(row, self.modality)[0]["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "value": part["text"]})
            else:
                buffer = io.BytesIO()
                part["url"].save(buffer, format="PNG")
                data = buffer.getvalue()
                path = Path(self.img_root) / (hashlib.sha256(data).hexdigest() + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(data)
                content.append({"type": "image", "value": str(path.resolve())})
        return content

    def dump_image(self, line):
        """Return the ordered image paths for framework inspection tools."""
        return [p["value"] for p in self.build_prompt(line) if p["type"] == "image"]

    def evaluate(self, eval_file, **judge_kwargs):
        """Score completed model predictions with MET-Bench's domain scorer."""
        predictions = load(eval_file)
        if predictions["index"].duplicated().any():
            raise ValueError("Prediction file contains duplicate example indices")
        if set(predictions["index"].astype(str)) != set(self.by_index):
            raise ValueError("Prediction file does not match the selected examples")
        if (
            predictions["prediction"]
            .astype(str)
            .str.contains(INFER_FAIL_MSG, regex=False)
            .any()
        ):
            raise ValueError("Retry failed inference requests before scoring MET-Bench")
        scores = []
        for _, prediction in predictions.iterrows():
            response = prediction["prediction"]
            if pd.isna(response):
                response = ""
            scores.append(
                score(self.by_index[str(prediction["index"])], str(response))
            )
        accuracy = sum(scores) / len(scores)
        if self.domain == "chess":
            lower, upper = clustered_ratio_interval([(value, 1) for value in scores])
        else:
            trials = len(scores)
            z = 1.959963984540054
            denominator = 1 + z * z / trials
            center = (accuracy + z * z / (2 * trials)) / denominator
            margin = z * math.sqrt(accuracy * (1 - accuracy) / trials + z * z / (4 * trials * trials)) / denominator
            lower, upper = max(0.0, center - margin), min(1.0, center + margin)
        result = pd.DataFrame(
            [
                {
                    "domain": self.domain,
                    "modality": self.modality,
                    "Overall": 100 * accuracy,
                    "ci_lower": 100 * lower if lower is not None else None,
                    "ci_upper": 100 * upper if upper is not None else None,
                    "examples": len(predictions),
                }
            ]
        )
        dump(result, str(Path(eval_file).with_suffix("")) + "_acc.csv")
        return result


class METBenchText(METBenchImage):
    """The text modality of the same MET-Bench examples."""

    MODALITY = "TEXT"
    modality = "text"

    def __init__(self, dataset="METBench_minecraft_text", limit=500):
        super().__init__(dataset, limit=limit)
