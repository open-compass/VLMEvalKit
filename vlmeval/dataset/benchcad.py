import re
from typing import Any, Dict, List, Union

import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import file
from vlmeval.smp.file import get_intermediate_file_path

# Question types graded by exact match (counts / integers / yes-no). Everything
# else (dimensions, ratios) uses symmetric ratio accuracy. Mirrors the official
# BenchCAD scoring (https://github.com/BenchCAD/BenchCAD-main, CodeQA/scoring).
_EXACT_TYPES = {"integer", "count", "boolean", "bool"}

_RULES = ("Answer with a SINGLE number and nothing else — no words, no units, no "
          "explanation. For yes/no questions output 1 for yes and 0 for no. For counts "
          "output an integer (e.g. 12). For ratios output a decimal (e.g. 2.5). For "
          "dimensions answer in millimetres.")

_TSV = "https://huggingface.co/datasets/BenchCAD/BenchCAD/resolve/main/vlmevalkit/BenchCAD_QA.tsv"
_MD5 = "e19747288563fe807d0ceb8049da1989"


def _to_number(x: Any) -> Union[float, None]:
    """Extract the first numeric value from a model prediction string."""
    if isinstance(x, (int, float)):
        return float(x)
    if x is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
    return float(m.group()) if m else None


def _score_one(pred: Union[float, None], gt: float, qa_type: str) -> float:
    if pred is None:
        return 0.0
    if (qa_type or "").lower() in _EXACT_TYPES:
        return 1.0 if pred == gt else 0.0
    if gt <= 0 or pred <= 0:
        return 0.0
    return min(pred, gt) / max(pred, gt)


class BenchCAD(ImageBaseDataset):
    """BenchCAD numeric QA over mechanical CAD parts (image, optionally + code).

    Each item shows rendered views of an industry-standard part and asks one
    numeric question (dimension / count / ratio). Scoring is execution-grounded
    and deterministic — no LLM judge: exact match for counts/integers/yes-no,
    symmetric ratio accuracy min(p, gt) / max(p, gt) for dimensions and ratios.

    - BenchCAD_VQA:    vision-only (rendered views -> number)
    - BenchCAD_CodeQA: multimodal (rendered views + CadQuery source -> number)

    Featured as an external benchmark in Anthropic's Claude Opus 4.8 system card.
    Source: https://github.com/BenchCAD/BenchCAD-main
    """

    TYPE = "VQA"
    DATASET_URL = {"BenchCAD_VQA": _TSV, "BenchCAD_CodeQA": _TSV}
    DATASET_MD5 = {"BenchCAD_VQA": _MD5, "BenchCAD_CodeQA": _MD5}

    def build_prompt(self, line: Union[int, pd.Series]) -> List[Dict[str, str]]:
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = str(line["question"])

        text = "You are an expert CAD engineer. "
        if "CodeQA" in self.dataset_name:
            text += ("You are shown rendered views of a mechanical part and its "
                     "CadQuery source.\n\nCadQuery code:\n```python\n" + str(line["gt_code"]) + "\n```\n\n")
        else:
            text += "You are shown rendered views of a mechanical part.\n\n"
        text += f"Question: {question}\n\n{_RULES}"

        msgs = [dict(type="image", value=p) for p in tgt_path]
        msgs.append(dict(type="text", value=text))
        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        data = file.load(eval_file)
        data["_score"] = [
            _score_one(_to_number(p), float(a), t)
            for p, a, t in zip(data["prediction"], data["answer"], data["qa_type"])
        ]

        rows = {"Overall": round(data["_score"].mean() * 100, 2)}
        for t, sub in data.groupby("qa_type"):
            rows[str(t)] = round(sub["_score"].mean() * 100, 2)
        score = pd.DataFrame([rows])

        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        file.dump(score, score_file)
        return score
