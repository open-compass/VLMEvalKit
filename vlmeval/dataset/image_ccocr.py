import os
import re
import tempfile
from functools import partial
import pandas as pd

from .utils import ccocr_evaluator_map
from .image_base import ImageBaseDataset
from ..smp import *


class CCOCRDataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        "CCOCR_Kie_Sroie2019Word": "https://duguang-mld.oss-cn-hangzhou.aliyuncs.com/songling/keepme/openai/kie_tsv/kie/constrained_category/sroie2019_word_347.tsv",
        "CCOCR_Kie_Cord": "https://duguang-mld.oss-cn-hangzhou.aliyuncs.com/songling/keepme/openai/kie_tsv/kie/constrained_category/CORD_100.tsv",
        "CCOCR_Kie_EphoieScut": "https://duguang-mld.oss-cn-hangzhou.aliyuncs.com/songling/keepme/openai/kie_tsv/kie/constrained_category/EPHOIE_SCUT_311.tsv",
        "CCOCR_Kie_Poie": "https://duguang-mld.oss-cn-hangzhou.aliyuncs.com/songling/keepme/openai/kie_tsv/kie/constrained_category/POIE_250.tsv",
        "CCOCR_Kie_ColdSibr": "https://duguang-mld.oss-cn-hangzhou.aliyuncs.com/songling/keepme/openai/kie_tsv/kie/open_category/COLD_SIBR_400.tsv",
        "CCOCR_Kie_ColdCell": "https://duguang-mld.oss-cn-hangzhou.aliyuncs.com/songling/keepme/openai/kie_tsv/kie/open_category/COLD_CELL_600.tsv"
    }

    DATASET_MD5 = {
        "CCOCR_Kie_Sroie2019Word": "e5d10a0e5238b40aee7e02ccf9dde6da",
        "CCOCR_Kie_Cord": "ab297cadcbc7158884a301c366f3330a",
        "CCOCR_Kie_EphoieScut": "bb8fa3ba7ea91cbf17be0904956ad3f3",
        "CCOCR_Kie_Poie": "882b64317989ecbfed6518051cdffb14",
        "CCOCR_Kie_ColdSibr": "109d5dad8b7081fb6a2f088e963196d4",
        "CCOCR_Kie_ColdCell": "17eaba669ef0504753aad0def6b8d9f8"
    }

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        """
        """
        df = load(eval_file)
        assert 'answer' in df and 'prediction' in df and "category" in df and "image_name" in df
        dict_list = df.to_dict(orient='records')

        gt_info, ptd_info = {}, {}
        for data_info in dict_list:
            image_name = data_info['image_name']
            gt_info[image_name] = data_info['answer']
            ptd_info[image_name] = data_info['prediction']

        # assert eval_file is a single dataset
        group_name = set([str(x) for x in df['category']]).pop()
        op_name = set([str(x) for x in df['l2-category']]).pop()
        data_name = set([str(x) for x in df['split']]).pop()

        data_info = {"op": op_name, "group": group_name, "dataset": data_name,  "num": len(gt_info)}
        eval_func = ccocr_evaluator_map.get(group_name, None)
        if eval_func is None:
            raise ValueError("error: evaluator not defined for: {}".format(group_name))
        meta_info, eval_info = eval_func(ptd_info, gt_info, **data_info)

        output_info = {"meta": meta_info, "evaluation": eval_info.get("summary"), "config": data_info}
        result_file = os.path.split(os.path.abspath(eval_file))[0] + "_eval.json"
        dump(output_info, result_file)
        print("--> [CCOCR]Evaluation result file: {}".format(result_file))
        return output_info
