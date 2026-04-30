import json
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load
from .refcoco import RefCOCODataset


class RefL4Dataset(RefCOCODataset):
    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'
    DATASET_URL = {
        'Ref-L4_test': ''
    }
    DATASET_MD5 = {}

    IOU_THRESHOLDS = [0.5, 0.75, 0.9]
    MACC_THRESHOLDS = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    SIZE_BUCKETS = (
        ('small', lambda side: side < 128),
        ('medium', lambda side: 128 <= side <= 256),
        ('large', lambda side: side > 256),
    )

    @classmethod
    def supported_datasets(cls):
        return ['Ref-L4_test']

    def evaluate(self, eval_file, **judge_kwargs):
        self._ensure_metadata_ready()
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')

        data['index'] = data['index'].astype(str)

        thresholds = self.IOU_THRESHOLDS
        macc_thresholds = self.MACC_THRESHOLDS

        ann_hits = {thr: [] for thr in thresholds}
        ann_macc_hits = []
        size_hits = {
            bucket: {
                'acc_0.5': [],
                'macc_0.5:0.95': [],
            }
            for bucket, _ in self.SIZE_BUCKETS
        }
        class_hits = defaultdict(lambda: {
            'acc_0.5': [],
            'macc_0.5:0.95': [],
        })

        details: List[Dict[str, object]] = []
        for record in data.to_dict('records'):
            idx = record['index']
            pred_text = str(record.get('prediction', ''))

            if idx not in meta.index:
                raise KeyError(f'Prediction index {idx} not found in Ref-L4 metadata.')

            meta_row = meta.loc[idx]
            width = float(meta_row.get('width', np.nan))
            height = float(meta_row.get('height', np.nan))
            gt_bbox = self._extract_gt_bbox(meta_row)
            pred_bbox_abs, pred_format = self._parse_ref_l4_prediction(pred_text, width, height)

            if pred_bbox_abs is None or gt_bbox is None:
                iou = 0.0
            else:
                iou = float(self._compute_iou(pred_bbox_abs, gt_bbox))

            threshold_hits = {thr: int(iou >= thr) for thr in thresholds}
            macc_scores = [int(iou >= thr) for thr in macc_thresholds]
            ann_macc = float(np.mean(macc_scores)) if macc_scores else 0.0

            for thr in thresholds:
                ann_hits[thr].append(threshold_hits[thr])
            ann_macc_hits.append(ann_macc)

            size_bucket = self._size_bucket(meta_row)
            if size_bucket is not None:
                size_hits[size_bucket]['acc_0.5'].append(threshold_hits[0.5])
                size_hits[size_bucket]['macc_0.5:0.95'].append(ann_macc)

            class_key = self._class_key(meta_row)
            class_hits[class_key]['acc_0.5'].append(threshold_hits[0.5])
            class_hits[class_key]['macc_0.5:0.95'].append(ann_macc)

            details.append({
                'index': idx,
                'pred_bbox': self._format_bbox(pred_bbox_abs),
                'pred_format': pred_format,
                'gt_bbox': self._format_bbox(gt_bbox),
                'iou': iou,
                'acc_iou_0.5': threshold_hits[0.5],
                'acc_iou_0.75': threshold_hits[0.75],
                'acc_iou_0.9': threshold_hits[0.9],
                'macc_iou_0.5:0.95': ann_macc,
                'size_bucket': size_bucket or '',
                'ori_category_id': meta_row.get('ori_category_id', ''),
                'class_key': class_key,
                'is_rewrite': meta_row.get('is_rewrite', ''),
                'split': meta_row.get('split', self.dataset_name),
            })

        detail_df = pd.DataFrame(details)
        dump(detail_df, get_intermediate_file_path(eval_file, '_detail'))

        summary = {
            'Ann-level acc iou 0.5': self._mean_pct(ann_hits[0.5]),
            'Ann-level acc iou 0.75': self._mean_pct(ann_hits[0.75]),
            'Ann-level acc iou 0.9': self._mean_pct(ann_hits[0.9]),
            'Ann-level macc iou 0.5:0.95': self._mean_pct(ann_macc_hits),
        }
        summary['Ann-level accs for copy'] = ', '.join(
            f'{summary[key]:.2f}'
            for key in [
                'Ann-level acc iou 0.5',
                'Ann-level acc iou 0.75',
                'Ann-level acc iou 0.9',
                'Ann-level macc iou 0.5:0.95',
            ]
        )

        for bucket, _ in self.SIZE_BUCKETS:
            summary[f'{bucket.capitalize()} acc iou 0.5'] = self._mean_pct(size_hits[bucket]['acc_0.5'])
            summary[f'{bucket.capitalize()} macc iou 0.5:0.95'] = self._mean_pct(size_hits[bucket]['macc_0.5:0.95'])
        summary['Size level accs for copy'] = ', '.join(
            f'{summary[key]:.2f}'
            for key in [
                'Small acc iou 0.5',
                'Small macc iou 0.5:0.95',
                'Medium acc iou 0.5',
                'Medium macc iou 0.5:0.95',
                'Large acc iou 0.5',
                'Large macc iou 0.5:0.95',
            ]
        )

        class_acc = [
            float(np.mean(values['acc_0.5'])) * 100
            for values in class_hits.values()
            if values['acc_0.5']
        ]
        class_macc = [
            float(np.mean(values['macc_0.5:0.95'])) * 100
            for values in class_hits.values()
            if values['macc_0.5:0.95']
        ]
        summary['Average class-level acc iou 0.5'] = float(np.mean(class_acc)) if class_acc else 0.0
        summary['Average class-level macc iou 0.5:0.95'] = float(np.mean(class_macc)) if class_macc else 0.0
        summary['Avg class-level accs for copy'] = ', '.join(
            f'{summary[key]:.2f}'
            for key in [
                'Average class-level acc iou 0.5',
                'Average class-level macc iou 0.5:0.95',
            ]
        )

        summary_df = pd.DataFrame({
            'Metric': list(summary.keys()),
            'Value': list(summary.values()),
        })
        dump(summary_df, get_intermediate_file_path(eval_file, '_acc'))
        return summary_df

    @classmethod
    def _parse_ref_l4_prediction(cls, text: str, width: float, height: float) -> tuple[Optional[np.ndarray], str]:
        if not isinstance(text, str):
            return None, ''

        stripped = text.strip()
        if not stripped:
            return None, ''

        parsed_json = cls._safe_json_load(stripped)
        if isinstance(parsed_json, dict):
            bbox = parsed_json.get('pred_bbox') or parsed_json.get('bbox')
            fmt = str(parsed_json.get('format', 'xyxy')).lower()
            pred = cls._normalize_prediction_bbox(bbox, fmt, width, height)
            if pred is not None:
                return pred, fmt

        if isinstance(parsed_json, list) and parsed_json:
            first = parsed_json[0]
            if isinstance(first, dict):
                bbox = first.get('pred_bbox') or first.get('bbox')
                fmt = str(first.get('format', 'xyxy')).lower()
                pred = cls._normalize_prediction_bbox(bbox, fmt, width, height)
                if pred is not None:
                    return pred, fmt

        inferred_format = 'xywh' if 'xywh' in stripped.lower() else 'xyxy'
        pred = cls._normalize_prediction_bbox(cls._parse_prediction(stripped), inferred_format, width, height)
        return pred, inferred_format

    @classmethod
    def _normalize_prediction_bbox(
        cls, bbox: object, bbox_format: str, width: float, height: float
    ) -> Optional[np.ndarray]:
        if bbox is None:
            return None

        if isinstance(bbox, str):
            coords = cls._parse_prediction(bbox)
        else:
            try:
                coords = np.array(list(bbox), dtype=float)
            except Exception:
                return None

        if coords is None or len(coords) < 4:
            return None

        coords = coords.astype(float)[:4]
        fmt = (bbox_format or 'xyxy').lower()
        if fmt == 'xywh':
            coords = np.array([coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]], dtype=float)

        return cls._to_absolute(coords, width, height)

    @staticmethod
    def _safe_json_load(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    def _size_bucket(self, meta_row: pd.Series) -> Optional[str]:
        bbox = self._extract_gt_bbox(meta_row)
        if bbox is None:
            return None

        side = float(np.sqrt(max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0)))
        for bucket, predicate in self.SIZE_BUCKETS:
            if predicate(side):
                return bucket
        return None

    @staticmethod
    def _class_key(meta_row: pd.Series) -> str:
        for key in ['mapped_category_id', 'ori_category_id', 'category_id', 'category']:
            value = meta_row.get(key, '')
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
        return 'unknown'

    @staticmethod
    def _mean_pct(values: List[float]) -> float:
        return float(np.mean(values)) * 100 if values else 0.0
