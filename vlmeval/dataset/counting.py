from .image_vqa import ImageVQADataset
from .utils.judge_util import build_judge
from .utils.extractor import LLM_Extractor
from .utils.multiple_choice import report_acc
import os.path as osp
import numpy as np
from vlmeval.smp import load, dump, get_intermediate_file_path, d2df, track_progress_rich
import re
import json
import shapely


class CountBenchQA(ImageVQADataset):
    TYPE = "VQA"
    DATASET_URL = {
        "CountBenchQA":
        "https://opencompass.openxlab.space/utils/VLMEval/CountBenchQA.tsv"
    }
    DATASET_MD5 = {"CountBenchQA": "fc73c8d4ffa665431448753f094d56ff"}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        ques = line['question']
        question = f'{ques} Note that: answer with a number directly e.g. 3. Do not include any additional text.'
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        answers = [str(x) for x in data['answer']]
        correct_count = 0
        total_count = len(predictions)

        for pred, ans in zip(predictions, answers):
            if ans in pred:
                correct_count += 1
        accuracy = correct_count / total_count if total_count > 0 else 0

        result = {'accuracy': accuracy * 100}
        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(d2df(result), result_file)
        return result

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        rating_file = cls.RATING_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
        rating_file = osp.join(root, rating_file)
        rating = load(rating_file)
        res = {'overall': rating.iloc[0]['accuracy']}
        return res


class FSC147(ImageVQADataset):
    TYPE = "VQA"
    DATASET_URL = {
        "FSC147":
        "https://opencompass.openxlab.space/utils/VLMEval/FSC147.tsv"
    }
    DATASET_MD5 = {"FSC147": "577a89387b1a99cf33e0623e67f01f2b"}
    DEFAULT_JUDGE = 'gpt-4o-mini'
    EXTRACT_PROMPT = 'Please extract an integer from the response and directly output it. '

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        ques = line['question']
        question = f'{ques} Note that: answer with a number directly e.g. 3. Do not include any additional text.'
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        judge = build_judge(**judge_kwargs)
        judge_file = get_intermediate_file_path(eval_file, '_judge', 'tsv')
        if not osp.exists(judge_file):
            extractor = LLM_Extractor(judge, self.EXTRACT_PROMPT, int)
            extracted = track_progress_rich(extractor.extract, list(data['prediction']), nproc=16, desc='Extracting')
            extracted = [x if x is not None else -1 for x in extracted]
            data['extracted'] = extracted
            dump(data, judge_file)

        predictions = [int(x) for x in data['extracted']]
        answers = [int(x) for x in data['answer']]
        hit, error = [], []
        for pred, ans in zip(predictions, answers):
            hit.append(pred == ans)
            error.append(min(np.abs(pred - ans), ans))

        accuracy = np.mean(hit)
        mae = np.mean(error)
        rmse = np.sqrt(np.mean(np.array(error) ** 2))
        result = {'accuracy': accuracy * 100, 'mae': mae, 'rmse': rmse}
        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(d2df(result), result_file)
        return result

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        rating_file = cls.RATING_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
        rating_file = osp.join(root, rating_file)
        rating = load(rating_file)
        rating = dict(rating.iloc[0])
        res = {'overall': rating['mae']}
        if verbose:
            res['rating'] = rating
        return res


def calculate_detection_metrics(predicted_points: list[list[float]], polygons_list: list[list[list[float]]]):
    from shapely.geometry import Point, Polygon
    """
    根据目标检测的思路，计算预测点相对于一组多边形（真实目标）的精确率和召回率。
    指标定义:
    - TP (True Positives): 被至少一个预测点命中的多边形数量。
    - FP (False Positives): 落在所有多边形外部的预测点数量。
    - FN (False Negatives): 完全没有被任何预测点命中的多边形数量。
    公式:
    - 精确率 (Precision) = (落在多边形内的点数) / (总点数)
    - 召回率 (Recall) = (被命中的多边形数) / (总多边形数)
    Args:
        predicted_points (list[list[float]]): 预测点列表，格式为 [[x1, y1], ...]。
        polygons_list (list[list[list[float]]]): 多个多边形（真实目标）的列表，
                                                格式为 [[[poly1_x1, poly1_y1], ...], ...]。
    Returns:
        dict: 一个包含精确率和召回率的字典。
              例如: {'precision': 0.8, 'recall': 0.75}
    """
    num_predictions = len(predicted_points)
    num_polygons = len(polygons_list)
    if num_predictions == 0:
        return {
            'precision': 0.0,
            # 如果没有预测，则一个目标也找不到，召回率为0
            'recall': 0.0
        }
    if num_polygons == 0:
        return {
            # 如果没有真实目标，所有预测点都是FP，精确率为0
            'precision': 0.0,
            # 如果没有真实目标，不存在“找回”的概念，召回率可以定义为1.0或0.0，这里定义为1.0表示没有需要找的目标，任务已完成
            'recall': 1.0
        }
    shapely_polygons = [Polygon(p) for p in polygons_list]
    points_in_polygons = 0
    hit_polygon_indices = set()  # 使用集合来存储被命中的多边形的索引，自动去重
    # 遍历每个预测点，计算精确率所需的数据
    for point_coords in predicted_points:
        point = Point(point_coords)
        is_inside_any = False
        for i, poly in enumerate(shapely_polygons):
            if poly.contains(point):
                is_inside_any = True
                hit_polygon_indices.add(i)  # 记录下这个被命中的多边形的索引
        if is_inside_any:
            points_in_polygons += 1
    # 计算精确率 (Precision)
    # (落在多边形内的点数) / (总点数)
    precision = points_in_polygons / num_predictions
    # 计算召回率 (Recall)
    # (被命中的多边形数) / (总多边形数)
    num_hit_polygons = len(hit_polygon_indices)
    recall = num_hit_polygons / num_polygons
    return precision, recall


class PointBench(ImageVQADataset):

    TYPE = "VQA"
    DATASET_URL = {
        "PointBench": "https://opencompass.openxlab.space/utils/VLMEval/PointBench.tsv",
        "PointBench_SEED": "https://opencompass.openxlab.space/utils/VLMEval/PointBench.tsv",
        "PointBench_GEMINI": "https://opencompass.openxlab.space/utils/VLMEval/PointBench.tsv",
    }
    DATASET_MD5 = {
        "PointBench": "e79cb7f883a58754f4e1ccb55db27d22",
        "PointBench_SEED": "e79cb7f883a58754f4e1ccb55db27d22",
        "PointBench_GEMINI": "e79cb7f883a58754f4e1ccb55db27d22",
    }

    def build_prompt(self, line):
        assert self.dataset_name == 'PointBench_SEED', self.dataset_name
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        ques = line['question']
        if self.dataset_name == 'PointBench_SEED':
            question = f"{ques}\nYou don't need to output anything except point(s)."
#         elif self.dataset_name == 'PointBench':
#             question_text = ques.split('Your output should be ')[0].strip()
#             extra_info = json.loads(line['extra_info'])
#             question = f"""\
# {question_text}
# The image dimensions are width={extra_info['width']}px, height={extra_info['height']}px.
# The answer should follow the json format [{{"point": <point>}}].
# IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format \
# where x is the horizontal position (left-to-right) \
# and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
# Example: For a point in the center of the image, return [width/2, height/2].
# """
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate_single(self, item):
        gt_list = json.loads(item['answer'])
        match_pattern = r"<point>.*?</point>"
        width, height, category, count = item['width'], item['height'], item['category'], item['count']

        pattern = re.compile(match_pattern, re.IGNORECASE | re.DOTALL)
        matched_points = pattern.findall(item['prediction'])
        pred_grounding_infos = []
        for match in matched_points:
            match = match.replace("<point>", "").replace("</point>", "")
            try:
                x, y = map(float, match.split(" "))
                pred_grounding_infos.append([x / 999 * width, y / 999 * height])
            except ValueError:
                continue

        if category != "counting":
            pred_grounding_infos = pred_grounding_infos[:1]
        precision, _ = calculate_detection_metrics(pred_grounding_infos, gt_list)
        if precision == 1 and len(pred_grounding_infos) == count:
            score = 1
        else:
            score = 0
        return score

    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        items = [data.iloc[i] for i in range(len(data))]
        scores = track_progress_rich(self.evaluate_single, [dict(item=x) for x in items])
        data['hit'] = scores
        dump(data, get_intermediate_file_path(eval_file, '_judge', 'tsv'))
        acc = report_acc(data)
        dump(acc, get_intermediate_file_path(eval_file, '_acc', 'csv'))
        return acc
