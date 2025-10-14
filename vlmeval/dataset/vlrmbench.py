from ast import literal_eval
from collections import defaultdict
import re
import numpy as np
from sklearn.metrics import f1_score

from .image_base import ImageBaseDataset
from ..smp import *


def format_model_answer_tolist(model_answer, task_gt):
    """
    从模型答案中提取0/1列表

    参数:
        model_answer: 模型的预测答案（字符串）
        task_gt: ground truth列表，用于确定期望的长度

    返回:
        list: 0/1列表
    """
    numbers = re.findall(r'\d+', str(model_answer))

    result = [int(num) for num in numbers]

    # 将非0/1的数字转换为1
    result = [num if num == 0 or num == 1 else 1 for num in result]

    # 调整长度以匹配task_gt
    if len(result) >= len(task_gt):
        return result[:len(task_gt)]
    else:
        return result + [0] * (len(task_gt) - len(result))


def get_F1Score(gathered_model_answer, gathered_task_gt):
    """
    计算F1分数

    参数:
        gathered_model_answer: 所有模型答案的列表
        gathered_task_gt: 所有ground truth的列表

    返回:
        tuple: (F1_pos, F1_neg, F1_w) - 正类F1、负类F1、加权F1
    """
    model_answer = np.array(gathered_model_answer)
    task_gt = np.array(gathered_task_gt)

    pos_count = np.sum(task_gt == 1)
    neg_count = np.sum(task_gt == 0)

    F1_pos = f1_score(task_gt, model_answer, pos_label=1, zero_division=0)
    F1_neg = f1_score(task_gt, model_answer, pos_label=0, zero_division=0)

    w_pos = neg_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
    w_neg = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0

    F1_w = w_neg * F1_neg + w_pos * F1_pos

    return F1_pos, F1_neg, F1_w


class VLRMBench(ImageBaseDataset):
    """
    VLRMBench Dataset - Visual Language Reasoning Model Benchmark

    A comprehensive benchmark for evaluating visual reasoning capabilities including:
    - step_correctness: 步骤正确性检测
    - redundant_det: 冗余检测
    - most_confidence: 最高置信度判断
    - attribute_hallucination: 属性幻觉检测
    - existence_hallucination: 存在性幻觉检测
    - detail_error: 细节错误检测
    - image_ref_error: 图像引用错误检测
    - location_error: 位置错误检测
    """

    TYPE = 'VQA'
    DATASET_URL = {
        'VLRMBench': 'https://huggingface.co/datasets/Winston-Yuan/VLRMBench/resolve/main/VLRMBench.tsv?download=true'
    }
    DATASET_MD5 = {
        'VLRMBench': None  # 可以后续添加MD5校验
    }

    def build_prompt(self, line):
        """
        构建提示信息

        参数:
            line: 数据行，可以是int索引或pd.Series

        返回:
            list: 多模态消息列表，格式为 [dict(type='image', value=path), dict(type='text', value=text), ...]
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # 创建line的副本避免SettingWithCopyWarning
        line = line.copy()

        # 使用父类方法保存图片（从base64解码并保存到本地）
        tgt_path = self.dump_image(line)

        # 构建消息
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        # 添加问题文本
        question = line.get('question', '')
        if question:
            msgs.append(dict(type='text', value=question))

        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        评估模型预测结果

        参数:
            eval_file: 模型预测结果文件路径
            **judge_kwargs: 其他评估参数

        返回:
            pd.DataFrame: 评估结果，包含各类别的F1分数
        """
        # 加载预测数据
        data = load(eval_file)

        # 确保必要的字段存在
        assert 'answer' in data.columns, "评估文件缺少 'answer' 字段"
        assert 'prediction' in data.columns, "评估文件缺少 'prediction' 字段"
        assert 'category' in data.columns, "评估文件缺少 'category' 字段"

        # 按类别收集模型答案和ground truth
        category_model_answers = defaultdict(list)
        category_task_gts = defaultdict(list)
        category_total = defaultdict(int)

        for idx in range(len(data)):
            item = data.iloc[idx]
            category = item['category']

            try:
                # 解析task_gt（answer字段）
                task_gt = item['answer']
                if isinstance(task_gt, str):
                    # 尝试将字符串解析为列表
                    task_gt = literal_eval(task_gt)

                # 获取模型答案（prediction字段）
                model_answer = item.get('prediction', '')

                # 使用format_model_answer_tolist格式化模型答案
                formatted_model_answer = format_model_answer_tolist(model_answer, task_gt)

                # 收集每个类别的答案
                category_task_gts[category].extend(task_gt)
                category_model_answers[category].extend(formatted_model_answer)
                category_total[category] += 1
            except Exception as e:
                # 如果解析失败，记录并跳过该样本
                print(f"处理样本失败 (idx={idx}, category={category}): {e}")
                continue

        # 计算各类别的F1分数
        results = {}
        for category in category_task_gts:
            gathered_task_gt = category_task_gts[category]
            gathered_model_answer = category_model_answers[category]

            if len(gathered_task_gt) > 0:
                F1_pos, F1_neg, F1_w = get_F1Score(gathered_model_answer, gathered_task_gt)

                results[f'{category}_F1_pos'] = F1_pos
                results[f'{category}_F1_neg'] = F1_neg
                results[f'{category}_F1_weighted'] = F1_w
                results[f'{category}_count'] = category_total[category]
            else:
                results[f'{category}_F1_pos'] = 0.0
                results[f'{category}_F1_neg'] = 0.0
                results[f'{category}_F1_weighted'] = 0.0
                results[f'{category}_count'] = 0

        # 计算总体F1分数（所有类别合并）
        all_task_gts = []
        all_model_answers = []
        for category in category_task_gts:
            all_task_gts.extend(category_task_gts[category])
            all_model_answers.extend(category_model_answers[category])

        if len(all_task_gts) > 0:
            F1_pos_overall, F1_neg_overall, F1_w_overall = get_F1Score(all_model_answers, all_task_gts)
            results['Overall_F1_pos'] = F1_pos_overall
            results['Overall_F1_neg'] = F1_neg_overall
            results['Overall_F1_weighted'] = F1_w_overall
            results['Overall_count'] = sum(category_total.values())
        else:
            results['Overall_F1_pos'] = 0.0
            results['Overall_F1_neg'] = 0.0
            results['Overall_F1_weighted'] = 0.0
            results['Overall_count'] = 0

        # 转换为DataFrame格式
        results_df = pd.DataFrame([results])

        # 保存结果
        score_file = eval_file.replace('.xlsx', '_scores.csv')
        dump(results_df, score_file)

        return results_df
