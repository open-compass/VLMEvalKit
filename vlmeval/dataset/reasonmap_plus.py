# vlmeval/dataset/reasonmap_plus_vqa.py
import re
import pandas as pd
import numpy as np
from typing import Any
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import load, d2df

_BOXED_PAT = re.compile(r'(?:\\boxed|boxed)\{([^}]*)\}', re.IGNORECASE)
_TEXT_PAT  = re.compile(r'\\text\{([^}]*)\}', re.IGNORECASE)

_YES = {"yes", "y", "true", "t", "1"}
_NO  = {"no", "n", "false", "f", "0"}

def _strip(s: Any) -> str:
    return ("" if s is None else str(s)).strip()

def _lower(s: Any) -> str:
    return _strip(s).lower()

def _extract_boxed(s: str) -> str | None:
    m = list(_BOXED_PAT.finditer(s))
    if not m:
        return None
    raw = m[-1].group(1).strip()
    texts = _TEXT_PAT.findall(raw)
    return " ".join(t.strip() for t in texts) if texts else raw

def _extract_after_phrases(s: str) -> str:
    phrases = [
        "the final answer is", "final answer is",
        "the answer is", "answer is",
        "the correct answer is", "correct answer is",
        "final answer:", "final:", "answer:", "ans:"
    ]
    lo = s.lower()
    for ph in phrases:
        if ph in lo:
            part = s[lo.rfind(ph) + len(ph):].strip()
            cand = re.split(r'(?:\n|\. |\.$)', part, maxsplit=1)[0]
            return cand.strip()
    return s.strip()

def _normalize_yesno(s: str) -> str | None:
    t = _lower(s)
    if t in _YES: return "yes"
    if t in _NO:  return "no"
    return None

def _normalize_abcd(s: str) -> str | None:
    m = re.search(r'\b([ABCD])\b', s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def _extract_int(s: str) -> int | None:
    m = re.search(r'[-+]?\d+', s)
    return int(m.group(0)) if m else None

def normalize_prediction(pred_raw: Any, typ: str) -> str:
    s = _strip(pred_raw)
    if not s:
        return ""
    
    boxed = _extract_boxed(s)
    cand = boxed if boxed else _extract_after_phrases(s)

    t = (typ or "").lower()
    if "torf" in t:
        yn = _normalize_yesno(cand)
        if yn is None:
            yn = _normalize_yesno(s)
        return yn or cand

    if t == "counting1" or "counting1" in t:
        abcd = _normalize_abcd(cand)
        if abcd is None:
            abcd = _normalize_abcd(s)
        return abcd or cand

    if t in {"counting2", "counting3"} or t.startswith("counting"):
        num = _extract_int(cand)
        if num is None:
            num = _extract_int(s)
        return str(num) if num is not None else cand

    return cand


class ReasonMap_Plus(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "ReasonMap-Plus": "https://opencompass.openxlab.space/utils/VLMEval/ReasonMap-Plus.tsv"
        # "ReasonMap-Plus": "/home/tuokaiwen/LMUData/ReasonMap-Plus.tsv"
    }
    
    # TODO: get md5 code
    # DATASET_MD5 = {"ReasonMap-Plus": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
    
    @classmethod
    def supported_datasets(cls):
        return ['ReasonMap-Plus']

    def build_prompt(self, line):
        if not isinstance(line, pd.Series):
            line = self.data_df.iloc[line]

        img_val = line.get("image", None)
        if not img_val:
            img_val = line.get("image_path", "")
        prompt = line.get("question", "")
        
        return [
            dict(type="image", value=img_val),
            dict(type="text",  value=prompt),
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        df = load(eval_file) 
        if len(df) == 0:
            return pd.DataFrame([dict(metric="accuracy", value=0.0, n=0)])

        df["_pred_norm"] = [
            normalize_prediction(p, t) for p, t in zip(df.get("prediction", ""), df.get("type", ""))
        ]

        def _score_one(a, p, t):
            tlo = (t or "").lower()
            try:
                if "torf" in tlo:
                    gt = "yes" if int(a) == 1 else "no"
                    pp = _normalize_yesno(p)
                    return 1 if (pp == gt) else 0

                if tlo == "counting1" or "counting1" in tlo:
                    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
                    pp = _normalize_abcd(p)
                    if pp is None:
                        return 0
                    return 1 if mapping[pp] == int(a) else 0

                if tlo in {"counting2", "counting3"} or tlo.startswith("counting"):
                    return 1 if int(str(p)) == int(a) else 0

                return 1 if _strip(a).lower() == _strip(p).lower() else 0
            except Exception:
                return 0

        df["_correct"] = [
            _score_one(a, p, t) for a, p, t in zip(df.get("answer", ""), df["_pred_norm"], df.get("type", ""))
        ]

        overall = float(np.mean(df["_correct"])) if len(df) else 0.0
        out_rows = [dict(metric="accuracy", value=overall, n=len(df))]

        for tname, sub in df.groupby(df.get("type", "")):
            if len(sub):
                out_rows.append(dict(metric=f"accuracy[{tname}]", value=float(np.mean(sub["_correct"])), n=len(sub)))

        # add some another info to the result .xlsx file
        # if "difficulty_city" in df.columns:
        #     for k, sub in df.groupby("difficulty_city"):
        #         if len(sub):
        #             out_rows.append(dict(metric=f"accuracy[difficulty={k}]", value=float(np.mean(sub["_correct"])), n=len(sub)))

        return pd.DataFrame(out_rows, columns=["metric", "value", "n"])
