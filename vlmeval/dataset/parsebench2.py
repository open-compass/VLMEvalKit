"""Public ParseBench-2 inputs, full-page media, and pinned official scoring."""

from pathlib import Path

import pandas as pd

from vlmeval.smp import dump, load
from vlmeval.smp.file import INFER_FAIL_MSG
from . import parsebench2_utils as pb2
from .image_base import ImageBaseDataset


class ParseBench2(ImageBaseDataset):
    TYPE = "VQA"
    MODALITY = "IMAGE"
    # Both native inference paths must retain the benchmark's fixed prompt.
    force_use_dataset_prompt = True

    @classmethod
    def supported_datasets(cls):
        return ["ParseBench2", *[f"ParseBench2_{subset}" for subset in pb2.SUBSETS]]

    def load_data(self, dataset):
        if dataset not in self.supported_datasets():
            raise ValueError(f"Unsupported ParseBench-2 dataset: {dataset}")
        rows = pb2.load_inputs()
        if dataset != "ParseBench2":
            subset = dataset.removeprefix("ParseBench2_")
            rows = [row for row in rows if row["subset"] == subset]
        return pd.DataFrame(
            [{**row, "index": row["example_id"], "question": pb2.PROMPT} for row in rows]
        )

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        # Render on demand; references are never loaded on the inference path.
        target = Path(self.img_root) / pb2.DATA_REVISION / line["example_id"]
        target.mkdir(parents=True, exist_ok=True)
        messages = []
        for number, image in enumerate(pb2.doc_to_visual(dict(line)), start=1):
            path = target / f"page-{number}.png"
            image.save(path)
            messages.append({"type": "image", "value": str(path)})
        return [*messages, {"type": "text", "value": pb2.PROMPT}]

    def evaluate(self, eval_file, **judge_kwargs):
        """Score every selected input; missing predictions stay in the denominator.

        No LLM judge, provider credentials, reference edits, or eligibility
        exceptions. Empty official metric lists stop the run before score output.
        """
        stem = Path(eval_file).with_suffix("")
        details_file = str(stem) + "_parsebench2_details.jsonl"
        score_file = str(stem) + "_score.json"
        # A failed rerun must not leave an earlier successful aggregate visible.
        for artifact in (details_file, score_file):
            Path(artifact).unlink(missing_ok=True)
        predictions = load(eval_file)
        if not isinstance(predictions, pd.DataFrame) or not {"index", "prediction"} <= set(
            predictions
        ):
            raise ValueError(
                "ParseBench-2 requires a prediction table with index and prediction columns"
            )
        predictions = predictions.copy()
        predictions["index"] = predictions["index"].map(str)
        if predictions["index"].duplicated().any():
            raise ValueError("Duplicate ParseBench-2 prediction IDs")
        known = set(self.data["index"].map(str))
        unknown = set(predictions["index"]) - known
        if unknown:
            raise ValueError(f"Unknown ParseBench-2 prediction IDs: {sorted(unknown)[:5]}")
        if not known:
            raise ValueError("ParseBench-2 has no selected input documents")
        by_id = predictions.set_index("index")["prediction"].to_dict()
        items = []
        for row in self.data.to_dict("records"):
            response = by_id.get(str(row["index"]))
            if response is None or (pd.api.types.is_scalar(response) and pd.isna(response)):
                response = None
            elif not isinstance(response, str):
                raise ValueError(f"Non-text prediction for {row['example_id']}")
            elif INFER_FAIL_MSG in response:
                response = None
            items.append(pb2.score_document(row, response))
        report = pb2.combined_report(items)
        metrics = {
            "overall": report["next_score"],
            **{name: report["dimensions"][name]["score"] for name in pb2.DIMENSIONS},
            "response_failure_rate": sum(
                item["response_status"] in {"missing", "invalid"} for item in items
            )
            / len(items),
            "scored_documents": len(items),
        }
        layout = [item for item in items if item["subset"] == "layout"]
        metrics["layout_coverage"] = (
            sum(item["layout_supported"] for item in layout) / len(layout) if layout else None
        )
        # Write only after all cases and the official aggregate succeed.
        dump(items, details_file)
        dump(
            {
                "dataset_revision": pb2.DATA_REVISION,
                "scorer_version": pb2.SCORER_VERSION,
                "renderer_version": pb2.RENDERER_VERSION,
                "prompt_version": "v1",
                "metrics": metrics,
                "official_report": report,
            },
            score_file,
        )
        return metrics
