import json
import os
import os.path as osp
import shutil
import sys

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, load
from .image_base import ImageBaseDataset


class FinixDocBench(ImageBaseDataset):
    MODALITY = 'IMAGE'
    TYPE = 'QA'

    REPO_ID = 'inclusionAI/FinixDocBench'
    DATASET_URL = {
        'FinixDocBench': '',
        'FinixDocBench_FinixDigital': '',
        'FinixDocBench_FinixPhoto': '',
        'FinixDocBench_FinixHuge_Long': '',
        'FinixDocBench_FinixHuge_Table': '',
    }
    DATASET_MD5 = {}

    DATASET_TRACKS = {
        'FinixDocBench': None,
        'FinixDocBench_FinixDigital': 'FinixDigital',
        'FinixDocBench_FinixPhoto': 'FinixPhoto',
        'FinixDocBench_FinixHuge_Long': 'FinixHuge-Long',
        'FinixDocBench_FinixHuge_Table': 'FinixHuge-Table',
    }

    TRACK_DIRS = {
        'FinixDigital': 'track1_finixdigital_242_insurance_terms',
        'FinixPhoto': 'track2_finixphoto_300',
        'FinixHuge-Long': 'track3_finixhuge_100_long',
        'FinixHuge-Table': 'track3_finixhuge_100_table',
    }

    system_prompt = (
        'Convert the document image into page-level Markdown. Preserve the original reading '
        'order, headings, paragraphs, lists, and tables. Use Markdown or HTML for tables '
        'when needed. '
        'Return only the Markdown content without explanations.'
    )

    def _snapshot_dir(self, dataset):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError('Please install huggingface_hub to use FinixDocBench.') from e

        track = self.DATASET_TRACKS[dataset]
        allow_patterns = ['dataset_manifest.jsonl', 'FinixDocBench_Eval_for_Markdown/**']
        if track is None:
            allow_patterns.extend([
                f'{subset_dir}/images/**' for subset_dir in self.TRACK_DIRS.values()
            ])
            allow_patterns.extend([
                f'{subset_dir}/mds/**' for subset_dir in self.TRACK_DIRS.values()
            ])
        else:
            subset_dir = self.TRACK_DIRS[track]
            allow_patterns.extend([f'{subset_dir}/images/**', f'{subset_dir}/mds/**'])

        cache_dir = osp.join(LMUDataRoot(), 'FinixDocBench_hf_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return snapshot_download(
            self.REPO_ID,
            repo_type='dataset',
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
        )

    def load_data(self, dataset):
        root = self._snapshot_dir(dataset)
        self.repo_root = root
        manifest_file = osp.join(root, 'dataset_manifest.jsonl')
        self.data_path = manifest_file
        target_track = self.DATASET_TRACKS[dataset]

        rows = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                meta = json.loads(line)
                track = meta['track']
                if target_track is not None and track != target_track:
                    continue

                markdown_path = osp.join(root, meta['markdown_path'])
                with open(markdown_path, 'r', encoding='utf-8') as mf:
                    answer = mf.read()

                row = dict(meta)
                row['index'] = meta['id']
                row['image_path'] = osp.join(root, meta['image_path'])
                row['markdown_path'] = markdown_path
                row['answer'] = answer
                row['question'] = self.system_prompt
                row['tasks'] = json.dumps(meta.get('tasks', []), ensure_ascii=False)
                rows.append(row)

        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = self.dump_image(line)[0]
        return [
            dict(type='image', value=image_path),
            dict(type='text', value=self.system_prompt),
        ]

    def _materialize_md_dirs(self, eval_file):
        data = load(eval_file)
        result_root = osp.splitext(eval_file)[0] + '_finixdocbench_eval'
        pred_dir = osp.join(result_root, 'pred')
        gt_dir = osp.join(result_root, 'gt')
        if osp.exists(pred_dir):
            shutil.rmtree(pred_dir)
        if osp.exists(gt_dir):
            shutil.rmtree(gt_dir)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        for _, row in data.iterrows():
            md_name = osp.basename(row['markdown_path'])
            gt_path = osp.join(gt_dir, md_name)
            pred_path = osp.join(pred_dir, md_name)
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(str(row['answer']) if not pd.isna(row['answer']) else '')
            with open(pred_path, 'w', encoding='utf-8') as f:
                if 'prediction' in row and not pd.isna(row['prediction']):
                    prediction = row['prediction']
                else:
                    prediction = ''
                f.write(str(prediction))

        return gt_dir, pred_dir, result_root

    def evaluate(self, eval_file, **judge_kwargs):
        gt_dir, pred_dir, result_root = self._materialize_md_dirs(eval_file)
        evaluator_root = osp.join(self.repo_root, 'FinixDocBench_Eval_for_Markdown')
        if not osp.isdir(evaluator_root):
            self.repo_root = self._snapshot_dir(self.dataset_name)
            evaluator_root = osp.join(self.repo_root, 'FinixDocBench_Eval_for_Markdown')

        sys.path.insert(0, evaluator_root)
        try:
            from finixdoc_md_eval.omnidocbench_adapter import evaluate_md_dirs
        except ImportError as e:
            raise ImportError(
                'Please install FinixDocBench evaluator requirements from '
                'FinixDocBench_Eval_for_Markdown/requirements.txt.'
            ) from e
        finally:
            if sys.path and sys.path[0] == evaluator_root:
                sys.path.pop(0)

        metrics = evaluate_md_dirs(gt_dir, pred_dir)
        metrics = {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
        metrics['score'] = metrics['overall']

        result = {
            'success': True,
            'metrics': metrics,
            'inputs': {
                'gt_files': len([x for x in os.listdir(gt_dir) if x.endswith('.md')]),
                'pred_files': len([x for x in os.listdir(pred_dir) if x.endswith('.md')]),
            },
        }
        output_json = osp.join(result_root, 'finixdocbench_result.json')
        dump(result, output_json)
        return pd.DataFrame([metrics])
