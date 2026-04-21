"""
LICA-Bench: graphic design VLM evaluation (layout, typography, SVG, templates, temporal, Lottie, category).

Benchmark code: https://github.com/purvanshi/lica-bench
Dataset: https://github.com/purvanshi/lica-dataset

Data preparation:
    python vlmeval/dataset/lica_bench.py --dataset-root /path/to/lica-benchmarks-dataset

This creates TSV files under $LMUDataRoot/LICABench/ that VLMEvalKit can consume.
"""

import json
import os
import os.path as osp
import warnings
from pathlib import Path

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset

LICA_BENCH_TASK_IDS = [
    'category-1', 'category-2',
    'layout-1', 'layout-2', 'layout-3', 'layout-4',
    'layout-5', 'layout-6', 'layout-7', 'layout-8',
    'svg-1', 'svg-2', 'svg-3', 'svg-4',
    'svg-5', 'svg-6', 'svg-7', 'svg-8',
    'template-1', 'template-2', 'template-3', 'template-4', 'template-5',
    'temporal-1', 'temporal-2', 'temporal-3',
    'temporal-4', 'temporal-5', 'temporal-6',
    'typography-1', 'typography-2', 'typography-3', 'typography-4',
    'typography-5', 'typography-6', 'typography-7', 'typography-8',
    'lottie-1', 'lottie-2',
]


def _vlmeval_name(task_id: str) -> str:
    return 'LICABench_' + task_id.replace('-', '_')


def _serialize_gt(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def prepare_lica_bench_tsv(dataset_root: str, task_ids=None, out_dir=None):
    """Convert lica-bench tasks into VLMEvalKit TSV files.

    Each TSV has columns: index, image_path, question, answer, category.
    Images stay on disk; ``image_path`` is absolute.
    """
    try:
        from design_benchmarks import BenchmarkRegistry
        from design_benchmarks.models.base import Modality, ModelInput
    except ImportError:
        raise ImportError(
            'lica-bench is required: pip install "lica-bench @ git+https://github.com/purvanshi/lica-bench.git"'
        )

    registry = BenchmarkRegistry()
    registry.discover()

    if task_ids is None:
        task_ids = LICA_BENCH_TASK_IDS

    if out_dir is None:
        out_dir = osp.join(LMUDataRoot(), 'LICABench')
    os.makedirs(out_dir, exist_ok=True)

    created = {}
    for tid in task_ids:
        try:
            bench = registry.get(tid)
        except KeyError:
            warnings.warn(f'lica-bench task {tid} not found in registry, skipping.')
            continue

        try:
            data_dir = bench.resolve_data_dir(dataset_root)
            samples = bench.load_data(data_dir, dataset_root=dataset_root)
        except Exception as e:
            warnings.warn(f'Failed to load data for {tid}: {e}')
            continue

        rows = []
        for i, sample in enumerate(samples):
            model_input = bench.build_model_input(sample, modality=Modality.TEXT_AND_IMAGE)
            if not isinstance(model_input, ModelInput):
                continue

            image_paths = []
            for img in (model_input.images or []):
                if isinstance(img, (str, Path)):
                    p = str(Path(img).expanduser().resolve())
                    if osp.isfile(p):
                        image_paths.append(p)

            question = model_input.text or ''
            meta = model_input.metadata or {}
            if meta:
                meta_str = json.dumps(meta, ensure_ascii=False, default=str)
                if len(meta_str) > 100_000:
                    meta_str = meta_str[:100_000] + '...[truncated]'
                question = f'{question}\n\n[metadata]\n{meta_str}' if question else f'[metadata]\n{meta_str}'

            gt = _serialize_gt(sample.get('ground_truth', ''))

            row = {
                'index': i,
                'question': question,
                'answer': gt,
                'category': bench.meta.domain,
            }
            if image_paths:
                if len(image_paths) == 1:
                    row['image_path'] = image_paths[0]
                else:
                    row['image_path'] = json.dumps(image_paths)
            rows.append(row)

        if not rows:
            warnings.warn(f'No valid samples for {tid}.')
            continue

        df = pd.DataFrame(rows)
        vname = _vlmeval_name(tid)
        tsv_path = osp.join(out_dir, f'{vname}.tsv')
        df.to_csv(tsv_path, sep='\t', index=False)
        created[vname] = tsv_path
        print(f'  {vname}: {len(rows)} samples -> {tsv_path}')

    return created


class LICABenchDataset(ImageBaseDataset):
    """VLMEvalKit dataset class for LICA-Bench tasks."""

    TYPE = 'VQA'

    DATASET_URL = {_vlmeval_name(tid): '' for tid in LICA_BENCH_TASK_IDS}
    DATASET_MD5 = {}

    def __init__(self, dataset='LICABench_category_1', **kwargs):
        self.lica_task_id = dataset.replace('LICABench_', '').replace('_', '-')
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return [_vlmeval_name(tid) for tid in LICA_BENCH_TASK_IDS]

    def load_data(self, dataset):
        tsv_path = osp.join(LMUDataRoot(), 'LICABench', f'{dataset}.tsv')
        if not osp.exists(tsv_path):
            raise FileNotFoundError(
                f'{tsv_path} not found. Run the data preparation script first:\n'
                '  python vlmeval/dataset/lica_bench.py --dataset-root /path/to/lica-benchmarks-dataset'
            )
        return load(tsv_path)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        msgs = []
        if 'image_path' in line and pd.notna(line.get('image_path')):
            if self.meta_only:
                tgt_path = toliststr(line['image_path'])
            else:
                tgt_path = self.dump_image(line)

            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs.append(dict(type='image', value=tgt_path))

        question = str(line.get('question', ''))
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if 'prediction' not in data.columns or 'answer' not in data.columns:
            warnings.warn('Missing prediction or answer column.')
            return {}

        predictions = data['prediction'].tolist()
        answers = data['answer'].tolist()

        correct = 0
        total = len(predictions)
        for pred, ans in zip(predictions, answers):
            pred_s = str(pred).strip().lower()
            ans_s = str(ans).strip().lower()
            if pred_s == ans_s:
                correct += 1
            elif ans_s in pred_s:
                correct += 1

        accuracy = correct / total * 100 if total > 0 else 0.0

        result = {
            'Overall': round(accuracy, 2),
            'Total': total,
            'Correct': correct,
        }

        if 'category' in data.columns:
            cats = data['category'].unique()
            for cat in cats:
                mask = data['category'] == cat
                cat_preds = data.loc[mask, 'prediction'].tolist()
                cat_ans = data.loc[mask, 'answer'].tolist()
                cat_correct = sum(
                    1 for p, a in zip(cat_preds, cat_ans)
                    if str(p).strip().lower() == str(a).strip().lower()
                    or str(a).strip().lower() in str(p).strip().lower()
                )
                cat_total = len(cat_preds)
                result[f'{cat}|Acc'] = round(cat_correct / cat_total * 100, 2) if cat_total else 0.0

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(result, score_file)
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare LICA-Bench TSV files for VLMEvalKit')
    parser.add_argument('--dataset-root', required=True, help='Path to lica-benchmarks-dataset/')
    parser.add_argument('--tasks', nargs='*', default=None, help='Task IDs (default: all)')
    parser.add_argument('--out-dir', default=None, help='Output directory (default: $LMUDataRoot/LICABench/)')
    args = parser.parse_args()

    created = prepare_lica_bench_tsv(
        dataset_root=args.dataset_root,
        task_ids=args.tasks,
        out_dir=args.out_dir,
    )
    print(f'\nPrepared {len(created)} TSV files.')
