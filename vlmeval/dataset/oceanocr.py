import json
import os
import copy
import pandas as pd
import base64
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from .image_base import ImageBaseDataset
from ..smp import *

import glob
import random
import re
from PIL import Image


def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))


def cal_per_metrics(pred, gt):

    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics


def lable_norm(eval_type, pred):
    if 'scene' in eval_type:
        # 去除一些额外的符号
        pred = pred.replace('\\text{', '').replace('}', '').replace('\n', ' ').replace('-', ' ').replace('*', ' ')
        pred = pred.strip()
        # 根据一些符号划分开并排序
        pred = re.split(r'[（）()，,\s]+', pred)
        pred.sort()
        pred = [p for p in pred if (p != ' ') and (p != '')]
        pred = ' '.join(pred)
    if 'handwritten' in eval_type:
        pred = pred.replace('*', ' ').replace('\n', ' ')
    return pred


def doc_text_eval(eval_type, predicts):

    result = []
    for ann in tqdm(predicts):
        ann['label'] = lable_norm(eval_type, ann['label'])
        ann['answer'] = lable_norm(eval_type, ann['answer'])
        ans = cal_per_metrics(ann["label"], ann["answer"])
        result.append(ans)

    mean_dict = {}

    mean_dict["eval question num"] = len(result)
    for k, v in result[0].items():
        mean_dict[k] = 0

    for each in result:
        for k, v in each.items():
            if v is None:
                v = 0
            mean_dict[k] += v

    for k, v in mean_dict.items():
        if k == "eval question num":
            continue
        mean_dict[k] /= len(result)
    # print(json.dumps(mean_dict, indent=4))
    return mean_dict


def avg_metric(results):
    if not results:
        return {}
    keys = ["bleu", "meteor", "f_measure", "precision", "recall", "edit_dist"]
    avg = {}
    for k in keys:
        values = [r[k] for r in results if k in r]
        avg[k] = float(np.mean(values)) if values else None
    avg["eval question num"] = int(np.sum([r["eval question num"] for r in results]))
    return avg


def run_eval(eval_type, results):
    """评估单个结果文件"""
    print(f"\n📊 Evaluating {eval_type} ...")
    results_single_domain = doc_text_eval(eval_type, results)
    results_single_domain["eval_type"] = eval_type
    print(f"✅ {eval_type} evaluation done.")
    return results_single_domain


def run_eval_wrapper(args):
    eval_type, result_list = args
    print(f"🚀 启动评测: {eval_type}, 样本数={len(result_list)}")
    res = run_eval(eval_type, result_list)
    return eval_type, res


def oceanocr_evaluate(tsv_path, eval_file):
    # pass
    print("\nStarting evaluation Ocean-OCR Bench...")
    df = pd.read_excel(eval_file)
    if "id" in df.iloc[0].values:
        print("⚠️ 检测到首行是header，跳过该行")
        df = df.iloc[1:].reset_index(drop=True)
    eval_types = ["document_en", "document_zh", "scene_text_rec", "handwritten_en", "handwritten_zh"]
    # eval_types = ["document_en"]
    results = {name: [] for name in eval_types}
    for i, row in df.iterrows():
        try:
            id_name = row["id"]
            answer_json = json.loads(row["answer"])
            label_value = answer_json[1]["value"]
            prediction = row.get("prediction", "")
            results[id_name].append({
                "label": label_value,
                "answer": prediction
            })
        except Exception as e:
            print(f"⚠️ 第{i}行解析失败: {e}")
            continue
    # single evaluator
    # eval_results = []
    # for eval_type in eval_types:
    #     results_single_domain = run_eval(eval_type, results[eval_type])
    #     eval_results.append(results_single_domain)

    tasks = [(eval_type, results[eval_type]) for eval_type in eval_types]
    eval_results = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(run_eval_wrapper, task) for task in tasks]
        for future in as_completed(futures):
            eval_type, res = future.result()
            eval_results.append(res)
            print(f"✅ {eval_type} 完成")

    print(eval_results)
    print("\n🎯 All inference + evaluation finished!")

    # 按中英文分类
    scene_results = [r for r in eval_results if r["eval_type"] == "scene_text_rec"]
    zh_results = [r for r in eval_results if r["eval_type"].endswith("_zh")]
    en_results = [r for r in eval_results if r["eval_type"].endswith("_en")]

    zh_avg = avg_metric(zh_results)
    en_avg = avg_metric(en_results)
    scene_avg = avg_metric(scene_results)

    print("\n================== Average (ZH) ==================")
    for k, v in zh_avg.items():
        print(f"{k:<20}: {v}")

    print("\n================== Average (EN) ==================")
    for k, v in en_avg.items():
        print(f"{k:<20}: {v}")

    print("\n================== Average (scene_text_rec) ==================")
    for k, v in en_avg.items():
        print(f"{k:<20}: {v}")

    csv_path = os.path.join(os.path.dirname(eval_file), "OceanOCRBench_eval_summary.csv")
    rows = []
    for name, dct in [
        ("ZH", zh_avg),
        ("EN", en_avg),
        ("scene_text_rec", scene_avg)
    ]:
        for k, v in dct.items():
            rows.append({"Category": name, "Metric": k, "Value": v})

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(csv_path, index=False)
    print(f"\n✅ Results save to: {csv_path}")


class OceanOCRBench(ImageBaseDataset):
    MODALITY = 'IMAGE'
    TYPE = 'QA'

    DATASET_URL = {'OceanOCRBench':'https://opencompass.openxlab.space/utils/VLMEval/OceanOCRBench.tsv'}
    DATASET_MD5 = {'OceanOCRBench': '482553a1bcf5cbcadcb31d847e6e01c8'}  # 测试版本的tsv文件  50243d62799e3467149195980cd7d660

    system_prompt = "Can you pull all textual information from the image?"

    def __init__(self,dataset='OceanOCRBench',**kwargs):
        super().__init__(dataset,**kwargs)
        print(f'self.img_root:{self.img_root}')

    def build_prompt(self, line):

        image_path = self.dump_image(line)[0]
        msg = [
            dict(type='image', value=image_path),
            dict(type='text', value=self.system_prompt)
        ]
        return msg

    def evaluate(self, eval_file, **judge_kwargs):
        try:
            import nltk
            from nltk.metrics import precision, recall, f_measure
            import jieba
            from nltk.translate import meteor_score
            from concurrent.futures import ProcessPoolExecutor, as_completed
        except ImportError as e:
            logging.critical(
                "Please follow the requirements (see vlmeval/dataset/utils/oceanoctbench/eval_req.txt) \
                             to install dependency package for chartmimic evaluation."
            )
            raise e
        tsv_path = self.data_path
        oceanocr_evaluate(tsv_path, eval_file)
