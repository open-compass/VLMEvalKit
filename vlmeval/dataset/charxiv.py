import os
import json
from typing import Dict, List, Tuple, Any, Union
import pandas as pd
import warnings
import random
from time import sleep

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import misc, file
from vlmeval.smp.file import get_intermediate_file_path
from vlmeval import utils
from vlmeval.dataset.utils import build_judge


def auxeval(judge_model: Any, line: pd.Series, **kwargs: Any) -> Dict[str, Any]:
    """
    Evaluate a line using the judge model.

    Args:
        judge_model: The model used for evaluation
        line: A pandas Series containing the data to evaluate
        **kwargs: Additional arguments for the judge model

    Returns:
        Dict containing evaluation results with extract_answer and score
    """
    failure_result = {"extract_answer": "Failed to parse response", "score": 0.0, "response": None}
    if "<think>" in line["prediction"] and "</think>" not in line["prediction"]:
        return {"extract_answer" : "'<think>....' pattern, treat not finished inference result.", "score": 0.0, "response": None}
    prompt = line["grading_query"].replace("{PREDICTION}", line["prediction"])

    retry = kwargs.get("retry", 10)
    if "oss" in judge_model.model:
        max_tokens = kwargs.get("max_tokens", 8192)
        temperature = kwargs.get("temperature", 0.1)
        judge_model.timeout = kwargs.get("timeout", 1200)
        extra_kwargs = {"reasoning_effort": "high"}
    else:
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0)
        extra_kwargs = {}
    seed = kwargs.get("seed", 42)
    top_p = kwargs.get("top_p", 1)

    from copy import deepcopy
    _content = deepcopy(failure_result)
    for _ in range(retry):
        try:
            response = judge_model.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
                **extra_kwargs,
            )
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            try:
                content = json.loads(response.strip())
            except json.JSONDecodeError as e:         
                try:
                    import ast
                    content = ast.literal_eval(response.strip())
                except Exception as e:
                    raise e

            if not isinstance(content, dict):
                content = _content
            if "score" not in content or "extract_answer" not in content:
                content = _content
            content["response"] = str(response)
            return content
        except Exception as e:
            sleep(random.random() * retry)
            content = _content
            content["response"] = f"{response} {e}"
            continue
    return content


def qid2category(mode: str) -> Tuple[Dict[int, str], str]:
    """
    Map question IDs to their categories based on the evaluation mode.

    Args:
        mode: Either "descriptive" or "reasoning"

    Returns:
        Tuple containing a mapping dictionary and the index column name

    Raises:
        ValueError: If the mode is not recognized
    """
    if mode == "descriptive":
        index_col = "qid"
        return {
            1: "Information Extraction",
            2: "Information Extraction",
            3: "Information Extraction",
            4: "Information Extraction",
            5: "Information Extraction",
            6: "Information Extraction",
            7: "Information Extraction",
            8: "Enumeration",
            9: "Enumeration",
            10: "Counting",
            11: "Pattern Recognition",
            12: "Counting",
            13: "Enumeration",
            14: "Enumeration",
            15: "Enumeration",
            16: "Pattern Recognition",
            17: "Compositionality",
            18: "Pattern Recognition",
            19: "Counting",
        }, index_col
    elif mode == "reasoning":
        index_col = "inst_category"
        return {
            1: "Text-in-Chart",
            2: "Text-in-General",
            3: "Number-in-Chart",
            4: "Number-in-General",
        }, index_col
    else:
        raise ValueError(f"Invalid mode: {mode}")


class CharXiv(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "CharXiv_descriptive_val": "http://opencompass.openxlab.space/utils/VLMEval/CharXiv_descriptive_val.tsv",
        "CharXiv_reasoning_val": "http://opencompass.openxlab.space/utils/VLMEval/CharXiv_reasoning_val.tsv",
    }
    DATASET_MD5 = {
        "CharXiv_descriptive_val": "e165037032f169a59dd09ea5d7ad3073",
        "CharXiv_reasoning_val": "98eeff269b40726982627b19338ccd45",
    }

    def build_prompt(self, line: Union[int, pd.Series]) -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = misc.toliststr(line["image"])
        else:
            tgt_path = self.dump_image(line)

        messages = [{"type": "image", "value": tgt_path[0]}]
        messages.append({"type": "text", "value": line["question"]})
        return messages

    def get_scores(self, result_file: str) -> pd.DataFrame:
        """
        Calculate scores by category from evaluation results.

        Args:
            result_file: Path to the file containing evaluation results

        Returns:
            DataFrame with scores for each category and overall score

        Raises:
            ValueError: If the dataset name is invalid
        """
        data = file.load(result_file)

        if "descriptive" in self.dataset_name:
            mode = "descriptive"
        elif "reasoning" in self.dataset_name:
            mode = "reasoning"
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        category_map, index_col = qid2category(mode)

        # Group scores by category
        scores_by_category = {}
        for _, row in data.iterrows():
            category = category_map[row[index_col]]
            if category not in scores_by_category:
                scores_by_category[category] = []
            scores_by_category[category].append(row["score"])

        # Calculate average score for each category
        result = {}
        for category, scores in scores_by_category.items():
            result[category] = [sum(scores) / len(scores)]

        # Calculate overall score
        result["Overall"] = [
            sum(sum(scores) for scores in scores_by_category.values()) / len(data)
        ]

        return pd.DataFrame(result)

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """
        Evaluate model predictions on the CharXiv dataset.

        Args:
            eval_file: Path to the file containing model predictions
            **judge_kwargs: Additional arguments for the judge model

        Returns:
            DataFrame with evaluation scores by category
        """
        # Set up judge model
        if "LOCAL_LLM" in os.environ:
            judge_model = os.path.basename(os.environ.get("LOCAL_LLM"))
        else:
            judge_model = judge_kwargs.get("model", "gpt-4o-mini")

        if judge_model != "gpt-4o-mini":
            warnings.warn(
                f"The judge_model '{judge_model}' is not gpt-4o-mini. Evaluation results may not be accurate."
            )
        judge_kwargs["model"] = judge_model
        judge_model = build_judge(**judge_kwargs)
        judge_model_name = judge_model.model
        if "/" in judge_model_name:
            judge_model_name = judge_model_name.replace("/", "_")

        # Define file paths
        result_file = get_intermediate_file_path(eval_file, f"_{judge_model_name}")
        temp_result_file = get_intermediate_file_path(eval_file, f"_{judge_model_name}", "pkl")
        score_file = get_intermediate_file_path(result_file, "_acc", "csv")

        # Return existing results if available
        if os.path.exists(result_file):
            score = self.get_scores(result_file)
            file.dump(score, score_file)
            return score

        data = file.load(eval_file)
        if "score" not in data.columns:
            data["score"] = 0
        if "extract_answer" not in data.columns:
            data["extract_answer"] = ""

        # Load intermediate results if available
        processed_results = {}
        if os.path.exists(temp_result_file):
            processed_results = file.load(temp_result_file)

        # Identify unprocessed indices
        indices = [i for i in range(len(data)) if i not in processed_results]
        tups = [(judge_model, data.iloc[i]) for i in range(len(data)) if i not in processed_results]

        # Process remaining examples
        nproc = judge_kwargs.pop("nproc", 4)
        print(f"nproc: {nproc}")
        if len(indices):
            print(f"Processing {len(indices)} examples, {len(tups)} total")
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

        # Update data with evaluation results
        data["score"] = data.apply(lambda x: processed_results[x.name]["score"], axis=1)
        data["extract_answer"] = data.apply(
            lambda x: processed_results[x.name]["extract_answer"], axis=1
        )
        data["response"] = data.apply(
            lambda x: processed_results[x.name].get("response", ""), axis=1
        )

        # Save results and return scores
        file.dump(data, result_file)
        score = self.get_scores(result_file)
        file.dump(score, score_file)
        return score
