import os
import json
from typing import Dict, List, Tuple, Any, Union
import pandas as pd
import warnings

from .image_base import ImageBaseDataset
from ..smp import misc, file, track_progress_rich, get_intermediate_file_path, parse_json
from .utils import build_judge


def auxeval(judge_model: Any, line: pd.Series, **kwargs: Any) -> Dict[str, Any]:
    """
    Evaluate a line using the judge model.

    Args:
        judge_model: The model used for evaluation
        line: A pandas Series containing the data to evaluate
        **kwargs: Additional arguments for the judge model

    Returns:
        Dict containing evaluation results with extracted_answer and score
    """
    failure_result = {"extracted_answer": "Failed to parse response", "score": 0.0}
    prompt = line["grading_query"].replace("{PREDICTION}", line["prediction"])
    retry = kwargs.get("retry", 3)
    max_tokens = kwargs.get("max_tokens", 1024)
    temperature = kwargs.get("temperature", 0)
    seed = kwargs.get("seed", 42)
    top_p = kwargs.get("top_p", 1)

    for _ in range(retry):
        try:
            response = judge_model.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
            )
            content = parse_json(response)
            if 'score' in content and 'extracted_answer' in content:
                return content
            else:
                print(response, flush=True)
            temperature += 0.5
        except Exception:
            continue

    return failure_result


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
        "CharXiv_descriptive_val": "https://opencompass.openxlab.space/utils/VLMEval/CharXiv_descriptive_val.tsv",
        "CharXiv_reasoning_val": "https://opencompass.openxlab.space/utils/VLMEval/CharXiv_reasoning_val.tsv",
    }
    DATASET_MD5 = {
        'CharXiv_descriptive_val': '8507c3740f8ddaedcb6b5c1cfcb3fa06',
        'CharXiv_reasoning_val': '6fc1a522ad32c2e3d72a89857b8cf10b',
    }
    DEFAULT_JUDGE = 'gpt-4o'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'

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
        judge_model = judge_kwargs.pop("model", "gpt-4o-mini")

        if judge_model != "gpt-4o-mini":
            warnings.warn(
                f"The judge_model '{judge_model}' is not gpt-4o-mini. Evaluation results may not be accurate."
            )

        judge_model_name = judge_model
        nproc = judge_kwargs.pop("nproc", 16)
        judge_model = build_judge(model=judge_model, **judge_kwargs)

        # Define file paths
        result_file = get_intermediate_file_path(eval_file, f"_{judge_model_name}")
        temp_result_file = get_intermediate_file_path(eval_file, f"_{judge_model_name}", "pkl")
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")

        # Return existing results if available
        if os.path.exists(result_file):
            score = self.get_scores(result_file)
            file.dump(score, score_file)
            return score

        data = file.load(eval_file)
        data['prediction'] = data['prediction'].astype(str)
        if "score" not in data.columns:
            data["score"] = 0
        if "extracted_answer" not in data.columns:
            data["extracted_answer"] = ""

        # Load intermediate results if available
        processed_results = {}
        if os.path.exists(temp_result_file):
            processed_results = file.load(temp_result_file)

        # Identify unprocessed indices
        indices = [i for i in range(len(data)) if i not in processed_results]
        tups = [(judge_model, data.iloc[i]) for i in range(len(data)) if i not in processed_results]

        # Process remaining examples
        if len(indices):
            track_progress_rich(
                auxeval,
                tups,
                nproc=nproc,
                chunksize=nproc,
                keys=indices,
                save=temp_result_file,
            )
            processed_results = file.load(temp_result_file)

        # Update data with evaluation results
        data["score"] = data.apply(lambda x: processed_results[x.name]["score"], axis=1)
        data["extracted_answer"] = data.apply(
            lambda x: processed_results[x.name]["extracted_answer"], axis=1
        )

        # Save results and return scores
        file.dump(data, result_file)
        score = self.get_scores(result_file)
        file.dump(score, score_file)
        return score
