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
    VLRMBench数据集基础类
    支持从HuggingFace下载和解压数据，处理JSONL格式的推理错误检测数据
    """

    MODALITY = "IMAGE"
    TYPE = "VQA"  # 设置为VQA类型以支持开放式问答

    # 支持的子集列表
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

    # HuggingFace仓库信息
    HF_REPO = "Winston-Yuan/VLRMBench"

    @classmethod
    def supported_datasets(cls):
        """返回支持的数据集名称列表"""
        return [f"VLRMBench_{subset}" for subset in cls.SUPPORTED_SUBSETS]

    def __init__(self, dataset="VLRMBench_attribute_hallucination", **kwargs):
        """
        初始化VLRMBench数据集

        Args:
            dataset: 数据集名称，格式为VLRMBench_{subset}
            **kwargs: 其他参数
        """
        # 从数据集名称中提取子集名称
        if dataset.startswith("VLRMBench_"):
            subset = dataset[len("VLRMBench_") :]
        else:
            subset = "attribute_hallucination"  # 默认子集

        if subset not in self.SUPPORTED_SUBSETS:
            raise ValueError(f"Unsupported subset: {subset}. Supported subsets: {self.SUPPORTED_SUBSETS}")

        self.subset = subset
        self.dataset_name = dataset

        # 设置数据根目录
        ROOT = LMUDataRoot()
        self.data_root = osp.join(ROOT, "datasets", "VLRMBench")
        os.makedirs(self.data_root, exist_ok=True)

        # 下载和解压数据
        self.data_dir = self._download_and_extract()

        # 加载数据
        data = self._load_jsonl_data()

        # 确保数据有index字段
        for i, item in enumerate(data):
            if 'index' not in item:
                item['index'] = i
        
        # 转换为DataFrame格式
        self.data = pd.DataFrame(data)

        # 设置图片根目录
        self.img_root = osp.join(self.data_dir, "images")

        # 设置评估模式
        self.evaluation_mode = self._get_evaluation_mode()

        # 后处理
        self.post_build(dataset)

    def _download_and_extract(self) -> str:
        """
        从HuggingFace下载数据并解压

        Returns:
            str: 解压后的数据目录路径
        """
        local_dir = osp.join(self.data_root, "VLRMBench-HF")

        # 检查是否已经下载
        if osp.exists(local_dir) and osp.exists(osp.join(local_dir, "benchmark_data")):
            print(f"VLRMBench data already exists at {local_dir}")
            return local_dir

        print(f"Downloading VLRMBench from HuggingFace: {self.HF_REPO}")


        # 下载数据
        snapshot_download(
            repo_id=self.HF_REPO,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            tqdm_class=None,  # 使用默认的tqdm进度条
        )

        # 解压图片文件
        self._extract_images(local_dir)

        print(f"VLRMBench data downloaded and extracted to {local_dir}")
        return local_dir

    def _extract_images(self, data_dir: str):
        """
        解压图片文件

        Args:
            data_dir: 数据目录路径
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
        加载JSONL数据文件

        Returns:
            List[Dict]: 加载的数据列表
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
        根据子集确定评估模式

        Returns:
            str: 评估模式
        """
        # 前瞻性推理使用任务级评估
        if self.subset == "foresight":
            return "foresight"

        # 多解任务使用特殊评估
        if self.subset == "multi_solution":
            return "multi_solution"

        # 生成任务使用judge评估 (暂时跳过)
        if self.subset in ["error_correction", "error_reason_analysis"]:
            return "generation"

        # 其他子集使用二进制分类评估
        return "binary_classification"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = dict(self.data.iloc[idx])

        # 处理图片路径
        if "image" in item and isinstance(item["image"], list):
            # 将相对路径转换为绝对路径
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
        """后处理，设置数据集特定属性"""
        # 设置评估指标
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
        构建提示词 - 根据评估模式构建不同的提示词

        Args:
            line: 数据行

        Returns:
            str: 构建的提示词
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
        格式化模型答案 - 基于原始评测脚本的逻辑

        Args:
            model_answer: 模型原始输出
            task_gt: 任务真实标签

        Returns:
            List[int]: 格式化后的预测结果
        """
        if self.evaluation_mode == "binary_classification":
            return self._format_binary_classification_answer(model_answer, task_gt)
        elif self.evaluation_mode == "multi_solution":
            return self._format_multi_solution_answer(model_answer, task_gt)
        else:
            return []

    def _format_binary_classification_answer(self, model_answer: str, task_gt: List) -> List[int]:
        """
        格式化二进制分类任务的模型答案
        基于 get_sc_mc_rd_eval_res.py 中的 format_model_answer_tolist 函数
        """
        # 提取数字
        numbers = re.findall(r"\d+", model_answer)
        result = [int(num) for num in numbers]

        # 将非0/1的数字转换为1
        result = [num if num == 0 or num == 1 else 1 for num in result]

        # 调整长度以匹配真实标签
        if len(result) >= len(task_gt):
            return result[: len(task_gt)]
        else:
            return result + [0] * (len(task_gt) - len(result))

    def _format_multi_solution_answer(self, model_answer: str, task_gt: List) -> List[int]:
        """
        格式化多解任务的模型答案
        基于 get_ms_eval_res.py 中的 format_ms_ec_era_model_answer_tolist 函数
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
        评估二进制分类任务
        基于 get_sc_mc_rd_eval_res.py 中的评估逻辑
        """
        # 展平所有预测和真实标签
        flat_predictions = []
        flat_ground_truths = []

        for pred, gt in zip(predictions, ground_truths):
            flat_predictions.extend(pred)
            flat_ground_truths.extend(gt)

        # 转换为numpy数组
        pred_array = np.array(flat_predictions)
        gt_array = np.array(flat_ground_truths)

        # 计算F1分数
        f1_pos = f1_score(gt_array, pred_array, pos_label=1)
        f1_neg = f1_score(gt_array, pred_array, pos_label=0)

        # 计算加权F1
        pos_count = np.sum(gt_array == 1)
        neg_count = np.sum(gt_array == 0)
        total_count = pos_count + neg_count

        if total_count > 0:
            f1_weighted = (f1_pos * pos_count + f1_neg * neg_count) / total_count
        else:
            f1_weighted = 0.0

        # 计算步骤准确率
        step_accuracy = np.mean(pred_array == gt_array)

        return {
            "f1_positive": f1_pos,
            "f1_negative": f1_neg,
            "f1_weighted": f1_weighted,
            "step_accuracy": step_accuracy,
        }

    def evaluate_foresight(self, predictions: List[str], ground_truths: List[bool]) -> Dict[str, float]:
        """
        评估前瞻性推理任务
        基于 get_fores_eval_res.py 中的评估逻辑
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
        评估多解任务
        基于 get_ms_eval_res.py 中的评估逻辑
        """
        correct = 0
        total = len(predictions)

        for pred, gt in zip(predictions, ground_truths):
            # 假设predictions包含front和back两个答案
            front_answer = pred.get("front", [0, 0])
            back_answer = pred.get("back", [0, 0])

            # 计算得分
            score1 = front_answer[0] + back_answer[1]
            score2 = front_answer[1] + back_answer[0]

            if score1 > score2:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {"multi_solution_accuracy": accuracy}


# 为每个子集创建具体的类
class VLRMBenchAttributeHallucination(VLRMBenchBase):
    """属性幻觉检测子集"""

    def __init__(self, dataset="VLRMBench_attribute_hallucination", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchDetailError(VLRMBenchBase):
    """细节错误检测子集"""

    def __init__(self, dataset="VLRMBench_detail_error", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchStepCorrectness(VLRMBenchBase):
    """步骤正确性评估子集"""

    def __init__(self, dataset="VLRMBench_step_correctness", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchForesight(VLRMBenchBase):
    """前瞻性推理评估子集"""

    def __init__(self, dataset="VLRMBench_foresight", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchErrorCorrection(VLRMBenchBase):
    def __init__(self, dataset="VLRMBench_error_correction", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchErrorReasonAnalysis(VLRMBenchBase):
    def __init__(self, dataset="VLRMBench_error_reason_analysis", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchExistenceHallucination(VLRMBenchBase):
    def __init__(self, dataset="VLRMBench_existence_hallucination", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchImageRefError(VLRMBenchBase):
    def __init__(self, dataset="VLRMBench_image_ref_error", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchLocationError(VLRMBenchBase):
    def __init__(self, dataset="VLRMBench_location_error", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchMostConfidence(VLRMBenchBase):
    def __init__(self, dataset="VLRMBench_most_confidence", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchMultiSolution(VLRMBenchBase):
    def __init__(self, dataset="VLRMBench_multi_solution", **kwargs):
        super().__init__(dataset=dataset, **kwargs)


class VLRMBenchRedundantDet(VLRMBenchBase):
    def __init__(self, dataset="VLRMBench_redundant_det", **kwargs):
        super().__init__(dataset=dataset, **kwargs)
