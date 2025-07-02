# vlmeval/dataset/static_embodied_bench_circular.py
from vlmeval.dataset.image_mcq import ImageMCQDataset
import string
import pandas as pd
from vlmeval.smp import *


class StaticEmbodiedBench_circular(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_NAME = "StaticEmbodiedBench_circular"
    DATASET_URL = {
        "StaticEmbodiedBench_circular": (
            "https://huggingface.co/datasets/xiaojiahao/StaticEmbodiedBench/"
            "resolve/main/StaticEmbodiedBench_circular.tsv"
        )
    }
    DATASET_MD5 = {
        "StaticEmbodiedBench_circular": "034cf398a3c7d848d966e1081e4baf68"
    }

    @classmethod
    def supported_datasets(cls):
        return [cls.DATASET_NAME]

    def build_prompt(self, line):
        return super().build_prompt(line)

    def evaluate(self, eval_file, **judge_kwargs):
        return super().evaluate(eval_file, **judge_kwargs)
