import os
import json
import zipfile
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import pandas as pd
import numpy as np
from huggingface_hub import snapshot_download
from sklearn.metrics import f1_score
from ..smp import *
from .image_base import ImageBaseDataset


class VLRMBenchBase(ImageBaseDataset):
    """
    Base class for VLRMBench dataset.
    Supports downloading and extracting data from HuggingFace,
    and processing JSONL-formatted reasoning error detection data.
    """

    MODALITY = "IMAGE"
    TYPE = "VQA"  # Set as VQA type to support open-ended QA

    # List of supported subsets
    SUPPORTED_SUBSETS = [
        "attribute_hallucination",
        "detail_error",
        "error_correction",
        "error_reason_analysis",
        "existence_hallucination",
        "foresight",
        "image_ref_error",
        "location_error",
        "most_confidence",
        "multi_solution",
        "redundant_det",
        "step_correctness",
    ]

    # HuggingFace repository information
    HF_REPO = "Winston-Yuan/VLRMBench"

    @classmethod
    def supported_datasets(cls):
        """Return list of supported dataset names."""
        return [f"VLRMBench_{subset}" for subset in cls.SUPPORTED_SUBSETS]

    def __init__(self, dataset="VLRMBench_attribute_hallucination", **kwargs):
        """
        Initialize VLRMBench dataset.

        Args:
            dataset: Dataset name in format VLRMBench_{subset}
            **kwargs: Additional arguments
        """
        # Extract subset name from dataset name
        if dataset.startswith("VLRMBench_"):
            subset = dataset[len("VLRMBench_") :]
        else:
            subset = "attribute_hallucination"  # Default subset

        if subset not in self.SUPPORTED_SUBSETS:
            raise ValueError(f"Unsupported subset: {subset}. Supported subsets: {self.SUPPORTED_SUBSETS}")

        self.subset = subset
        self.dataset_name = dataset

        # Set data root directory
        ROOT = LMUDataRoot()
        self.data_root = osp.join(ROOT, "datasets", "VLRMBench")
        os.makedirs(self.data_root, exist_ok=True)

        # Download and extract data
        self.data_dir = self._download_and_extract()

        # Load data
        data = self._load_jsonl_data()

        # Ensure data has index field
        for i, item in enumerate(data):
            if 'index' not in item:
                item['index'] = i
        
        # Convert to DataFrame format
        self.data = pd.DataFrame(data)

        # Set image root directory
        self.img_root = osp.join(self.data_dir, "images")

        # Set evaluation mode
        self.evaluation_mode = self._get_evaluation_mode()

        # Post-processing
        self.post_build(dataset)

    def _download_and_extract(self) -> str:
        """
        Download data from HuggingFace and extract images.

        Returns:
            str: Path to the extracted data directory
        """
        local_dir = osp.join(self.data_root, "VLRMBench-HF")

        # Check if already downloaded
        if osp.exists(local_dir) and osp.exists(osp.join(local_dir, "benchmark_data")):
            print(f"VLRMBench data already exists at {local_dir}")
            return local_dir

        print(f"Downloading VLRMBench from HuggingFace: {self.HF_REPO}")


        # Download data
        snapshot_download(
            repo_id=self.HF_REPO,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            tqdm_class=None,  # Use default tqdm progress bar
        )

        # Extract image files
        self._extract_images(local_dir)

        print(f"VLRMBench data downloaded and extracted to {local_dir}")
        return local_dir

    def _extract_images(self, data_dir: str):
        """
        Extract image files from zip archive.

        Args:
            data_dir: Path to the data directory
        """
        zip_file = osp.join(data_dir, "Image.zip")
        images_dir = osp.join(data_dir, "images")

        if osp.exists(images_dir):
            print("Images already extracted")
            return

        if not osp.exists(zip_file):
            raise FileNotFoundError(f"Image.zip not found at {zip_file}")

        print(f"Extracting images from {zip_file}")
        os.makedirs(images_dir, exist_ok=True)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(images_dir)

        print(f"Images extracted to {images_dir}")

    def _load_jsonl_data(self) -> List[Dict]:
        """
        Load JSONL data file.

        Returns:
            List[Dict]: List of loaded data items
        """
        jsonl_file = osp.join(self.data_dir, "benchmark_data", f"{self.subset}.jsonl")

        if not osp.exists(jsonl_file):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")

        data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        print(f"Loaded {len(data)} samples from {self.subset}")
        return data

    def _get_evaluation_mode(self) -> str:
        """
        Determine evaluation mode based on subset.

        Returns:
            str: Evaluation mode
        """
        # Foresight reasoning uses task-level evaluation
        if self.subset == "foresight":
            return "foresight"

        # Multi-solution task uses special evaluation
        if self.subset == "multi_solution":
            return "multi_solution"

        # Generation tasks use judge evaluation (skipped for now)
        if self.subset in ["error_correction", "error_reason_analysis"]:
            return "generation"

        # Other subsets use binary classification evaluation
        return "binary_classification"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = dict(self.data.iloc[idx])

        # Process image paths
        if "image" in item and isinstance(item["image"], list):
            # Convert relative paths to absolute paths
            image_paths = []
            for img_path in item["image"]:
                full_path = osp.join(self.img_root, img_path)
                if osp.exists(full_path):
                    image_paths.append(full_path)
                else:
                    print(f"Warning: Image not found: {full_path}")

            item["image"] = image_paths[0] if len(image_paths) == 1 else image_paths

        return item

    def post_build(self, dataset):
        """Post-processing to set dataset-specific attributes."""
        # Set evaluation metrics
        if self.evaluation_mode == "binary_classification":
            self.metrics = ["f1_positive", "f1_negative", "f1_weighted", "step_accuracy"]
        elif self.evaluation_mode == "foresight":
            self.metrics = ["task_accuracy"]
        elif self.evaluation_mode == "multi_solution":
            self.metrics = ["multi_solution_accuracy"]
        elif self.evaluation_mode == "generation":
            self.metrics = ["win_rate", "judge_score"]
        else:
            self.metrics = ["overall_accuracy"]

    def build_prompt(self, line):
        """
        Build prompt based on evaluation mode.

        Args:
            line: Data row

        Returns:
            str: Constructed prompt
        """
        question = line["question"]

        if self.evaluation_mode == "binary_classification":
            step_list = line.get("step_list", [])
            prompt = f"Question: {question}\n\nReasoning Steps:\n"
            for i, step in enumerate(step_list, 1):
                prompt += f"Step {i}: {step}\n"
            prompt += "\nPlease identify which steps contain errors. Output format: [0,1,0,1,...]"

        elif self.evaluation_mode == "foresight":
            step_list = line.get("step_list", [])
            prompt = f"Question: {question}\n\nReasoning Steps:\n"
            for i, step in enumerate(step_list, 1):
                prompt += f"Step {i}: {step}\n"
            prompt += "\nDoes this reasoning show good foresight? Answer: yes/no"

        elif self.evaluation_mode == "multi_solution":
            prompt = f"Question: {question}\n\nPlease provide two different solution approaches."

        elif self.evaluation_mode == "generation":
            reasoning_error = line.get("reasoning_error", [])
            prompt = f"Question: {question}\n\nReasoning with errors:\n"
            for i, step in enumerate(reasoning_error, 1):
                prompt += f"Step {i}: {step}\n"
            prompt += "\nPlease analyze and correct the errors in this reasoning."

        return prompt

    def format_model_answer(self, model_answer: str, task_gt: List) -> List[int]:
        """
        Format model answer based on original evaluation script logic.

        Args:
            model_answer: Raw model output
            task_gt: Ground truth labels

        Returns:
            List[int]: Formatted prediction results
        """
        if self.evaluation_mode == "binary_classification":
            return self._format_binary_classification_answer(model_answer, task_gt)
        elif self.evaluation_mode == "multi_solution":
            return self._format_multi_solution_answer(model_answer, task_gt)
        else:
            return []

    def _format_binary_classification_answer(self, model_answer: str, task_gt: List) -> List[int]:
        """
        Format binary classification task model answer.
        Based on format_model_answer_tolist function in get_sc_mc_rd_eval_res.py
        """
        # Extract numbers
        numbers = re.findall(r"\d+", model_answer)
        result = [int(num) for num in numbers]

        # Convert non-0/1 numbers to 1
        result = [num if num == 0 or num == 1 else 1 for num in result]

        # Adjust length to match ground truth
        if len(result) >= len(task_gt):
            return result[: len(task_gt)]
        else:
            return result + [0] * (len(task_gt) - len(result))

    def _format_multi_solution_answer(self, model_answer: str, task_gt: List) -> List[int]:
        """
        Format multi-solution task model answer.
        Based on format_ms_ec_era_model_answer_tolist function in get_ms_eval_res.py
        """
        numbers = re.findall(r"\d+", model_answer)
        result = [int(num) for num in numbers]

        if len(result) >= len(task_gt):
            return result[-len(task_gt) :]
        else:
            return result + [0] * (len(task_gt) - len(result))

    def evaluate_binary_classification(
        self, predictions: List[List[int]], ground_truths: List[List[int]]
    ) -> Dict[str, float]:
        """
        Evaluate binary classification tasks.
        Based on evaluation logic in get_sc_mc_rd_eval_res.py
        """
        # Flatten all predictions and ground truths
        flat_predictions = []
        flat_ground_truths = []

        for pred, gt in zip(predictions, ground_truths):
            flat_predictions.extend(pred)
            flat_ground_truths.extend(gt)

        # Convert to numpy arrays
        pred_array = np.array(flat_predictions)
        gt_array = np.array(flat_ground_truths)

        # Calculate F1 scores
        f1_pos = f1_score(gt_array, pred_array, pos_label=1)
        f1_neg = f1_score(gt_array, pred_array, pos_label=0)

        # Calculate weighted F1
        pos_count = np.sum(gt_array == 1)
        neg_count = np.sum(gt_array == 0)
        total_count = pos_count + neg_count

        if total_count > 0:
            f1_weighted = (f1_pos * pos_count + f1_neg * neg_count) / total_count
        else:
            f1_weighted = 0.0

        # Calculate step accuracy
        step_accuracy = np.mean(pred_array == gt_array)

        return {
            "f1_positive": f1_pos,
            "f1_negative": f1_neg,
            "f1_weighted": f1_weighted,
            "step_accuracy": step_accuracy,
        }

    def evaluate_foresight(self, predictions: List[str], ground_truths: List[bool]) -> Dict[str, float]:
        """
        Evaluate foresight reasoning tasks.
        Based on evaluation logic in get_fores_eval_res.py
        """
        correct = 0
        total = len(predictions)

        for pred, gt in zip(predictions, ground_truths):
            if gt == True:
                if re.search(r"\b(yes|true)\b", pred, re.IGNORECASE):
                    correct += 1
            elif gt == False:
                if re.search(r"\b(no|false)\b", pred, re.IGNORECASE):
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {"task_accuracy": accuracy}

    def evaluate_multi_solution(self, predictions: List[Dict], ground_truths: List[List[int]]) -> Dict[str, float]:
        """
        Evaluate multi-solution tasks.
        Based on evaluation logic in get_ms_eval_res.py
        """
        correct = 0
        total = len(predictions)

        for pred, gt in zip(predictions, ground_truths):
            # Assume predictions contain front and back answers
            front_answer = pred.get("front", [0, 0])
            back_answer = pred.get("back", [0, 0])

            # Calculate scores
            score1 = front_answer[0] + back_answer[1]
            score2 = front_answer[1] + back_answer[0]

            if score1 > score2:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {"multi_solution_accuracy": accuracy}


# Create specific classes for each subset
class VLRMBenchAttributeHallucination(VLRMBenchBase):
    """Attribute hallucination detection subset."""

    def __init__(self, dataset="VLRMBench_attribute_hallucination", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchDetailError(VLRMBenchBase):
    """Detail error detection subset."""

    def __init__(self, dataset="VLRMBench_detail_error", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchStepCorrectness(VLRMBenchBase):
    """Step correctness evaluation subset."""

    def __init__(self, dataset="VLRMBench_step_correctness", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchForesight(VLRMBenchBase):
    """Foresight reasoning evaluation subset."""

    def __init__(self, dataset="VLRMBench_foresight", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchErrorCorrection(VLRMBenchBase):
    """Error correction subset."""

    def __init__(self, dataset="VLRMBench_error_correction", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchErrorReasonAnalysis(VLRMBenchBase):
    """Error reason analysis subset."""

    def __init__(self, dataset="VLRMBench_error_reason_analysis", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchExistenceHallucination(VLRMBenchBase):
    """Existence hallucination detection subset."""

    def __init__(self, dataset="VLRMBench_existence_hallucination", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchImageRefError(VLRMBenchBase):
    """Image reference error detection subset."""

    def __init__(self, dataset="VLRMBench_image_ref_error", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchLocationError(VLRMBenchBase):
    """Location error detection subset."""

    def __init__(self, dataset="VLRMBench_location_error", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchMostConfidence(VLRMBenchBase):
    """Most confidence evaluation subset."""

    def __init__(self, dataset="VLRMBench_most_confidence", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchMultiSolution(VLRMBenchBase):
    """Multi-solution evaluation subset."""

    def __init__(self, dataset="VLRMBench_multi_solution", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchRedundantDet(VLRMBenchBase):
    """Redundant detection subset."""

    def __init__(self, dataset="VLRMBench_redundant_det", **kwargs):
        super().__init__(dataset=dataset, **kwargs)
