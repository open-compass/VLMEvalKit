import re
import json
from collections import defaultdict

from .image_base import ImageBaseDataset
from vlmeval.smp import *
from ast import literal_eval


PROMPT_TEMPLATE = """All spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the bounding box coordinate of the object the following statement describes:
{description}
Ensure that all details mentioned about the object are accurate. Provide at most one bounding box. If a matching object is found, provide its bounding box as a JSON in the format {{"bbox_2d": [x1, y1, x2, y2]}}. If no matching object is found, output {{"bbox_2d": null}}."""  # noqa: E501


def parse_bbox(text):
    """Extract bounding box from model response."""
    try:
        match = re.search(r'\{.*"bbox_2d".*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            bbox = data["bbox_2d"]
            if bbox is None:
                return [0, 0, 0, 0]
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(coord) for coord in bbox]
    except:
        pass

    # Fallback: try to extract four numbers
    pattern = r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, text)
    if matches:
        return [float(coord) for coord in matches[-1]]

    return [0, 0, 0, 0]


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    if box1 == [0, 0, 0, 0]:
        return 1 if box2 == [0, 0, 0, 0] else 0

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def normalize_bbox(bbox, width, height):
    """Convert normalized or 0-999 range bbox to pixel coordinates."""
    if all(coord <= 1 for coord in bbox):
        return [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
    return [bbox[0] / 999 * width, bbox[1] / 999 * height, bbox[2] / 999 * width, bbox[3] / 999 * height]


class GroundingME(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'GroundingME': 'https://huggingface.co/datasets/lirang04/GroundingME/resolve/main/groundingme.tsv'
    }
    DATASET_MD5 = {
        'GroundingME': '69780c69665a9e087dad1278b0bcdf49'
    }
    JUDGE_FORMAT = '{model_name}_{dataset_name}_detailed.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_score.json'

    def build_prompt(self, line):
        """Build prompt for a single sample."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Save image to local
        tgt_path = self.dump_image(line)

        # Get question/description
        description = line['question']

        # Build prompt
        prompt = PROMPT_TEMPLATE.format(description=description)

        # Build messages
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        msgs.append(dict(type='text', value=prompt))

        return msgs

    @classmethod
    def evaluate_single_item(cls, item):
        response = str(item['prediction'])
        pred_bbox = parse_bbox(response)
        # Get ground truth bbox from answer field
        try:
            gt_bbox = literal_eval(item['answer'])
            if not isinstance(gt_bbox, list):
                gt_bbox = [0, 0, 0, 0]
        except:
            gt_bbox = [0, 0, 0, 0]

        # Get image dimensions
        width = int(item['width'])
        height = int(item['height'])

        # Get subtask info
        subtask_l1 = item['subtask_l1']

        # Handle Rejection category
        if subtask_l1 == 'Rejection':
            gt_bbox = [0, 0, 0, 0]

        # Try different coordinate formats and pick the best one
        pred_candidates = [
            pred_bbox,
            normalize_bbox(pred_bbox, width, height),
        ]

        ious = [compute_iou(gt_bbox, pred) for pred in pred_candidates]
        best_idx = ious.index(max(ious))
        best_iou = ious[best_idx]

        return best_iou

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate predictions and compute metrics."""
        # Load data
        data = load(eval_file)
        judge_file = cls.get_judge_file_path(eval_file)
        rating_file = cls.get_rating_file_path(eval_file)

        if not osp.exists(judge_file):
            ious = track_progress_rich(
                cls.evaluate_single_item,
                [row for _, row in data.iterrows()],
                nproc=32,
                desc='GroundingME Evaluation')
            data['best_iou'] = ious
        else:
            data = load(judge_file)

        # Aggregate overall metrics
        metrics = {
            'IoU': data['best_iou'].mean(),
            'ACC@0.5': np.mean([x >= 0.5 for x in data['best_iou']]),
            'ACC@0.75': np.mean([x >= 0.75 for x in data['best_iou']]),
            'ACC@0.9': np.mean([x >= 0.9 for x in data['best_iou']]),
        }

        # Per-category metrics (subtask_l1)
        categories_l1 = data.groupby('subtask_l1')
        for cat_name, cat_data in categories_l1:
            metrics[f'{cat_name}_ACC@0.5'] = np.mean([x >= 0.5 for x in cat_data['best_iou']])
            metrics[f'{cat_name}_ACC@0.75'] = np.mean([x >= 0.75 for x in cat_data['best_iou']])
            metrics[f'{cat_name}_ACC@0.9'] = np.mean([x >= 0.9 for x in cat_data['best_iou']])

        # Per-subcategory metrics (subtask_l2)
        results_with_l2 = data[data['subtask_l2'] != '']
        if len(results_with_l2) > 0:
            categories_l2 = results_with_l2.groupby(['subtask_l1', 'subtask_l2'])
            for (cat_l1, cat_l2), cat_data in categories_l2:
                cat_name = f'{cat_l1}_{cat_l2}'
                metrics[f'{cat_name}_ACC@0.5'] = np.mean([x >= 0.5 for x in cat_data['best_iou']])

        dump(metrics, rating_file)
        return metrics
