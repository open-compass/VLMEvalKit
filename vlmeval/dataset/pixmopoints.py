import json

import numpy as np
import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset
from .utils.spatial_bench.tools.utils import Point2DParser


class PixmoPointsDataset(ImageBaseDataset):
    """Point localization evaluation using Hungarian matching."""

    TYPE = 'VQA'
    DATASET_URL = {'PixmoPoints': ''}
    DATASET_MD5 = {}

    DISTANCE_THRESHOLD = 0.05  # 5% of normalized image size

    PROMPT_SUFFIX = (
        ' Output the point coordinates in JSON format.\n'
        'For example:\n'
        '[\n'
        '  {"point_2d": [x, y], "label": "point_1"}\n'
        ']'
    )

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += self.PROMPT_SUFFIX
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        data = data.sort_values(by='index')

        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')
        data['index'] = data['index'].astype(str)

        from scipy.optimize import linear_sum_assignment

        details = []
        precision_sum, recall_sum, f1_sum, total = 0, 0, 0, 0
        for _, row in data.iterrows():
            meta_row = meta.loc[row['index']] if row['index'] in meta.index else row
            width = int(float(meta_row.get('width', row.get('width', 1)) or 1))
            height = int(float(meta_row.get('height', row.get('height', 1)) or 1))

            pred_pts = Point2DParser.parse(str(row['prediction']), width, height, output='norm')
            gt_pts = self._parse_points(str(meta_row.get('answer', row.get('answer', ''))))
            pred_pts = pred_pts.tolist() if pred_pts is not None else []

            if len(gt_pts) == 0:
                precision, recall, f1 = (1.0, 1.0, 1.0) if len(pred_pts) == 0 else (0.0, 1.0, 0.0)
            elif len(pred_pts) == 0:
                precision, recall, f1 = 0.0, 0.0, 0.0
            else:
                pred_arr = np.array(pred_pts)
                gt_arr = np.array(gt_pts)
                dists = np.linalg.norm(pred_arr[:, None] - gt_arr[None, :], axis=2)
                row_ind, col_ind = linear_sum_assignment(dists)

                matches = sum(
                    dists[i, j] < self.DISTANCE_THRESHOLD
                    for i, j in zip(row_ind, col_ind)
                )
                precision = matches / len(pred_pts)
                recall = matches / len(gt_pts)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            total += 1
            details.append({
                'index': row['index'],
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })

        result = {
            'precision': precision_sum / total if total > 0 else 0,
            'recall': recall_sum / total if total > 0 else 0,
            'f1': f1_sum / total if total > 0 else 0,
        }
        dump(pd.DataFrame(details), get_intermediate_file_path(eval_file, '_detail'))
        dump(result, get_intermediate_file_path(eval_file, '_score', 'json'))
        return result

    @staticmethod
    def _parse_points(s):
        try:
            pts = json.loads(s)
            if not isinstance(pts, list):
                return []
            result = []
            for p in pts:
                point = None
                if isinstance(p, list) and len(p) == 2:
                    point = p
                elif isinstance(p, dict) and 'point_2d' in p and isinstance(p['point_2d'], list) and len(p['point_2d']) == 2:
                    point = p['point_2d']
                elif isinstance(p, dict) and 'point' in p and isinstance(p['point'], list) and len(p['point']) == 2:
                    point = p['point']
                if point is None:
                    continue
                try:
                    x, y = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    continue
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    result.append([x, y])
            return result
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            return []
