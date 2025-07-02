# vlmeval/dataset/static_embodied_bench.py
from vlmeval.dataset.image_mcq import ImageMCQDataset
import string
import pandas as pd
from vlmeval.smp import *


class StaticEmbodiedBench(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_NAME = "StaticEmbodiedBench"
    DATASET_URL = {
        "StaticEmbodiedBench": (
            "https://huggingface.co/datasets/xiaojiahao/StaticEmbodiedBench/resolve/main/StaticEmbodiedBench.tsv"
        )
    }
    DATASET_MD5 = {
        "StaticEmbodiedBench": "5c50611650ca966970180a80d49429f0"
    }

    @classmethod
    def supported_datasets(cls):
        return [cls.DATASET_NAME]

    def build_prompt(self, line):
        return super().build_prompt(line)

    def evaluate(self, eval_file, **judge_kwargs):
        return super().evaluate(eval_file, **judge_kwargs)
