import os
import re
import json
import base64
import hashlib
import warnings
from typing import Dict, List, Tuple, Any, Union

import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import misc, file
from vlmeval import utils
from vlmeval.dataset.utils import build_judge
import re
import copy
import base64
import mimetypes
# -------------------- Your judge rubric (kept) --------------------
JUDGE_WORLDQA_PROMPT_EN = """
### Role
You are an expert judge specialized in evaluating the correctness of answers. Your task is to assess whether a model-generated answer is correct based on a given question, the model's response, and the ground truth answer.

### Task: Evaluate Answer Correctness
Please classify the model's response into one of the following three categories. Ignore differences in formatting, punctuation, language (Chinese vs. English), or abbreviations/full names. Focus strictly on the **core semantics** and the **level of detail (granularity)**:

1. **Correct**:
    - The model answer contains the core information of the ground truth.
    - The model answer is semantically consistent with the ground truth and contains no contradictions.
    - **The granularity of the model answer is equal to or finer than the ground truth.**
    - Extra irrelevant information is allowed as long as it does not conflict with the ground truth.

2. **Incorrect**:
    - The model answer provides information that contradicts the ground truth.
    - The model answer provides the wrong specific entity, value, or description.
    - **The granularity of the model answer is coarser than the ground truth**, leading to incomplete or insufficiently specific information.
    - Even if the model expresses uncertainty but follows up with a wrong answer (e.g., "I'm not sure, maybe it's B" when the truth is A), it is considered Incorrect.

3. **Unattempted**:
    - The model explicitly states it does not know the answer (e.g., "I don't know," "I cannot answer this question").
    - The model suggests the user search elsewhere (e.g., "Please search the internet").
    - The model answer contains no information from the ground truth but provides no incorrect or contradictory information.

### Output Format
Please strictly follow this two-line format for your output:
1. **Evaluation**: [A brief explanation of your reasoning]
2. **Label**: [Final classification: "Correct", "Incorrect", or "Unattempted"]

---
### Examples

**Example 1 (Incorrect - Granularity Mismatch/Too Coarse)**
Input:
'''
Question: 图片中属于什么类型的田地？
Model Answer: 图片中展示的是梯田。梯田是在山坡地上开垦并修筑的阶梯状农田。
Ground Truth Answer: 龙脊梯田
'''
Evaluation: 标准答案特指“龙脊梯田”，模型只回答了通用的“梯田”。模型答案层级比答案层级更粗略，未能提供标准答案所需的特指信息，属于层级不一致导致的回答错误。
Label: Incorrect

**Example 2 (Correct - Finer Granularity)**
Input:
'''
Question: What weather phenomenon is in the image?
Model Answer: Based on the visual evidence in the image, the weather phenomenon shown is a **severe storm with extremely high winds**, most likely a **tornado** or a very powerful **hurricane/typhoon**.
Ground Truth Answer: High winds
'''
Evaluation: The ground truth is "high winds," and a "tornado" is a more specific and granular type of high wind. The semantics are correct and the detail is finer.
Label: Correct

**Example 3 (Correct)**
Input:
'''
Question: 图中内容是什么品牌的logo？
Model Answer: via浏览器
Ground Truth Answer: via
'''
Evaluation: 模型答案“via浏览器”包含了标准答案“via”，核心语义一致，且“via浏览器”是更具体的描述，层级上是匹配的。
Label: Correct

**Example 4 (Unattempted)**
Input:
'''
Question: Which athlete is in the image?
Model Answer: I cannot answer this question as I do not have relevant sports data.
Ground Truth Answer: Wout Weghorst
'''
Evaluation: The model explicitly states its inability to answer and provides no incorrect information.
Label: Unattempted

**Example 5 (Incorrect)**
Input:
'''
Question: 图片中展示的是什么苹果品种？
Model Answer: 我觉得可能是阿克苏苹果。
Ground Truth Answer: 烟台苹果
'''
Evaluation: 虽然模型用了“可能”等词汇，但它给出的具体答案“阿克苏苹果”与标准答案“烟台苹果”不符，提供了错误信息。
Label: Incorrect

**Example 6 (Unattempted)**
Input:
'''
Question: What is the name of the insect in this image?
Model Answer: This is a photo of an insect. To find the species, consult an entomologist or use recognition software.
Ground Truth Answer: Japanese rhinoceros beetle
'''
Evaluation: The model does not attempt to name the insect and suggests the user search elsewhere, providing no incorrect information.
Label: Unattempted

---
### Current Task
Input:
'''
Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth_answer}
'''

Evaluation:
"""


# -------------------- helpers --------------------
def _strip_think(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    # remove <think>...</think> or <thinking>...</thinking>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.S)
    return text.strip()


def _normalize_base64_image(b64: str) -> str:
    """
    Accept:
      - raw base64
      - data:image/...;base64,....
    Return raw base64 part.
    """
    if not isinstance(b64, str):
        raise ValueError("image field is not a string")
    if b64.startswith("data:image"):
        # data:image/jpeg;base64,xxxx
        return b64.split(",", 1)[1]
    return b64


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -------------------- auxeval in vlmeval style --------------------
def auxeval(judge_model: Any, line: pd.Series, **kwargs: Any) -> Dict[str, Any]:
    """
    Expect eval_file has at least:
      - question
      - answer (ground truth) OR ground_truth
      - prediction
    """
    failure_result = {"extract_answer": "Failed to parse response", "score": 0.0}

    question = str(line.get("question", ""))
    gt = line.get("answer", None)
    if gt is None:
        gt = line.get("ground_truth", "")
    gt = str(gt)

    pred = str(line.get("prediction", ""))
    model_answer = _strip_think(pred)

    judge_prompt = JUDGE_WORLDQA_PROMPT_EN.format(
        question=question,
        model_answer=model_answer,
        ground_truth_answer=gt,
    )

    # vlmeval judge wrapper expects "prompt" (string) typically
    retry = kwargs.get("retry", 10)
    max_tokens = kwargs.get("max_tokens", 2048)
    temperature = kwargs.get("temperature", 0)
    seed = kwargs.get("seed", 42)
    top_p = kwargs.get("top_p", 1)
    # breakpoint()

    for _ in range(retry):
        try:
            resp = judge_model.generate(
                judge_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
            )
            print(resp)
            # must be json
            text = resp.strip().replace('*', '')
            

            eval_match = re.search(r"Evaluation:\s*(.*?)(?:\n\s*Label:|\Z)", text, re.S | re.I)
            evaluation_text = eval_match.group(1).strip() if eval_match else text
            # print(f"{eval_match=}, {evaluation_text=}")

            label_match = re.search(r"Label:\s*(Correct|Incorrect|Unattempted)", text, re.I)
            print(f"{label_match=}")
            if not label_match:
                continue

            label = label_match.group(1).strip().lower()
            label = label.replace('label:', '').strip().strip('\t').strip('\n')
            
            ans = copy.deepcopy(line)
            ans['judge_model_output'] = text
            if label == "correct":
                ans['extract_answer'] = "Correct"
            elif label == "incorrect":
                ans['extract_answer'] = "InCorrect"
            elif label == "unattempted":
                ans['extract_answer'] = "Unattempted"
            else:
                ans['extract_answer'] = 'failed'

            return ans
        except Exception:
            continue

    return failure_result


# -------------------- dataset --------------------
class WorldVQA(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        # 改成你自己的 tsv 路径或挂载路径
        "WorldVQA": "https://huggingface.co/datasets/moonshotai/WorldVQA/blob/main/WorldVQA.tsv",
    }
    DATASET_MD5 = {"WorldVQA": '3353b1151968179e5264190ece028fed'}

    def _image_cache_dir(self) -> str:
        # 把 base64 解出来的图片放这里（跟 result 文件同级也行，这里统一放在 ~/.cache）
        root = os.environ.get("VLM_EVAL_IMAGE_CACHE", os.path.expanduser("~/.cache/vlmeval/worldvqa_images"))
        _safe_mkdir(root)
        return root

    def _dump_base64_to_file(self, b64: str, suffix: str = ".jpg") -> str:
        raw = _normalize_base64_image(b64)
        # 用 hash 去重，避免重复写盘
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()
        out_path = os.path.join(self._image_cache_dir(), f"{h}{suffix}")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        img_bytes = base64.b64decode(raw)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        return out_path

    def build_prompt(self, line: Union[int, pd.Series]) -> List[Dict[str, str]]:
        """
        这里遵循 vlmeval：messages = [{"type":"image","value": <path>}, {"type":"text","value": question}]
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        def image_path_to_base64_url(path: str) -> str:
    
            mime, _ = mimetypes.guess_type(path)
            if mime is None:
                mime = "image/png"  # 兜底

            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            return f"data:{mime};base64,{b64}"

        img_path = line.get("image_path")

        if img_path:
            img_base64_url = image_path_to_base64_url(img_path)
        else:
            img_base64_url = None
        


        # breakpoint()
        messages: List[Dict[str, str]] = []
        messages.append({"type": "image", "value": img_base64_url})

        question = str(line.get("question", ""))
        lang = str(line.get("language", "en"))
        if lang == "zh":
            messages.append({"type": "text", "value": "请尽可能提供详细的回答。\n" + question})
        else:
            messages.append({"type": "text", "value": "Please provide as much detail as possible in your answer.\n" + question})

        return messages

    def get_scores(self, result_file: str) -> pd.DataFrame:
        df = file.load(result_file)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        s = df["extract_answer"].fillna("").str.lower()
        # breakpoint()
        df = df.assign(
            _attempted=s.isin(["correct", "incorrect", "failed"]),
            _correct=(s == "correct")
        )

        def agg(g: pd.DataFrame) -> pd.Series:
            tot = len(g)
            att = int(g["_attempted"].sum())
            cor = int(g["_correct"].sum())
            acc = cor / tot if tot else 0.0
            cga = cor / att if att else 0.0
            attempt = att / tot if tot else 0.0
            f = 2 * cga * acc / (cga + acc) if (cga + acc) else 0.0
            return pd.Series(
                {"Total": tot, "Attempted": att, "Correct": cor,
                "Accuracy": acc, "CGA": cga, "AttemptRate": attempt, "F-score": f}
            )


        out = []

        # difficulty
        if "difficulty" in df.columns:
            d = df.groupby("difficulty", dropna=False).apply(agg).reset_index().rename(columns={"difficulty": "Metric"})
            out.append(d)

        # category
        if "category" in df.columns:
            c = df.groupby("category", dropna=False).apply(agg).reset_index().rename(columns={"category": "Metric"})
            c["Metric"] = "category_" + c["Metric"].astype(str)
            out.append(c)

        # overall
        overall = agg(df).to_frame().T
        overall.insert(0, "Metric", "Overall")
        out.append(overall)

        return pd.concat(out, ignore_index=True)

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        if "LOCAL_LLM" in os.environ:
            judge_name = os.path.basename(os.environ.get("LOCAL_LLM"))
        else:
            judge_name = judge_kwargs.get("model", "gpt-4o-1120")

        if judge_name != "gpt-4o-mini":
            warnings.warn(f"judge_model='{judge_name}' is not gpt-4o-mini; results may vary.")

        judge_kwargs["model"] = judge_kwargs.get("model", "gpt-4o-1120")
        judge_model = build_judge(**judge_kwargs)
        judge_model_name = judge_model.model

        suffix = eval_file.split(".")[-1]
        result_file = eval_file.replace(f".{suffix}", f"_{judge_model_name}.xlsx")
        temp_result_file = eval_file.replace(f".{suffix}", f"_{judge_model_name}.pkl")
        score_file = result_file.replace(".xlsx", "_score.csv")

        if os.path.exists(result_file):
            score = self.get_scores(result_file)
            file.dump(score, score_file)
            return score

        data = file.load(eval_file)

        # 兼容字段：ground_truth / answer
        if "answer" not in data.columns and "ground_truth" in data.columns:
            data["answer"] = data["ground_truth"]

        if "score" not in data.columns:
            data["score"] = 0.0
        if "extract_answer" not in data.columns:
            data["extract_answer"] = ""

        processed_results = {}
        if os.path.exists(temp_result_file):
            processed_results = file.load(temp_result_file)

        indices = [i for i in range(len(data)) if i not in processed_results]
        tups = [(judge_model, data.iloc[i]) for i in range(len(data))]

        nproc = judge_kwargs.pop("nproc", 4)
        if len(indices):
            utils.track_progress_rich(
                auxeval,
                tups,
                nproc=nproc,          
                chunksize=nproc,
                keys=indices,
                save=temp_result_file,
                **judge_kwargs,
            )
            processed_results = file.load(temp_result_file)

        data["score"] = data.apply(lambda x: processed_results[x.name]["score"], axis=1)
        data["extract_answer"] = data.apply(lambda x: processed_results[x.name]["extract_answer"], axis=1)

        file.dump(data, result_file)
        score = self.get_scores(result_file)
        file.dump(score, score_file)
        return score
