"""MemLens: Multimodal Long-Context Conversational Memory benchmark.

Paper: https://arxiv.org/abs/2605.14906
HF:    xiyuRenBill/MEMLENS
"""
import json
import os
import os.path as osp
import re
import tarfile

import pandas as pd
from huggingface_hub import hf_hub_download

from .image_base import ImageBaseDataset
from .utils import build_judge
from .utils.memlens_utils import evaluate as memlens_judge_eval
from ..smp import dump, get_logger, load, toliststr
from ..smp.file import get_intermediate_file_path, LMUDataRoot


def _safe_extract(tar, path):
    root = osp.abspath(path)
    for member in tar.getmembers():
        target = osp.abspath(osp.join(root, member.name))
        if not (target == root or target.startswith(root + os.sep)):
            raise RuntimeError(f'Unsafe path in tar archive: {member.name}')
    tar.extractall(root)


class MemLens(ImageBaseDataset):
    """MemLens benchmark across four context lengths (32K / 64K / 128K / 256K)."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE = 'deepseek'

    HF_REPO_ID = 'xiyuRenBill/MEMLENS'
    SOURCE_HF_REPO_ID = HF_REPO_ID
    DATASET_URL = {
        'MemLens_32K': 'https://huggingface.co/datasets/xiyuRenBill/MEMLENS/resolve/main/vlmevalkit/MemLens_32K.tsv',
        'MemLens_64K': 'https://huggingface.co/datasets/xiyuRenBill/MEMLENS/resolve/main/vlmevalkit/MemLens_64K.tsv',
        'MemLens_128K': 'https://huggingface.co/datasets/xiyuRenBill/MEMLENS/resolve/main/vlmevalkit/MemLens_128K.tsv',
        'MemLens_256K': 'https://huggingface.co/datasets/xiyuRenBill/MEMLENS/resolve/main/vlmevalkit/MemLens_256K.tsv',
    }
    DATASET_MD5 = {}

    # All splits share the original MemLens image archive.
    IMAGE_ARCHIVE = 'release_images.tar.gz'

    ABSTENTION_TYPES = frozenset({'answer_refusal'})

    def __init__(self, dataset='MemLens_32K', **kwargs):
        super().__init__(dataset=dataset, skip_noimg=False)
        if os.environ.get('MEMLENS_AUTO_DOWNLOAD_IMAGES', '1') == '1':
            self._prepare_images()

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @classmethod
    def _tsv_path(cls, dataset):
        root = os.environ.get('MEMLENS_TSV_ROOT', LMUDataRoot())
        return osp.join(root, f'{dataset}.tsv')

    def load_data(self, dataset):
        tsv_path = self._tsv_path(dataset)
        if osp.exists(tsv_path):
            data = load(tsv_path)
        else:
            url = self.DATASET_URL.get(dataset, '')
            if not url:
                raise FileNotFoundError(
                    f'MemLens TSV not found at {tsv_path} and no DATASET_URL configured.')
            data = self.prepare_tsv(url, self.DATASET_MD5.get(dataset))

        # Normalise image_path to list[str]
        if 'image_path' in data.columns:
            data['image_path'] = data['image_path'].apply(toliststr)
        return data

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------

    def _img_root(self):
        env = os.environ.get('MEMLENS_IMAGE_ROOT', '')
        if env:
            return env
        return osp.join(LMUDataRoot(), 'images', 'memlens_images')

    def _resolve_image_path(self, rel_path):
        if osp.isabs(rel_path):
            return rel_path
        img_root = self._img_root()
        candidates = [
            osp.join(img_root, rel_path),
            osp.join(img_root, 'release_images', rel_path),
            osp.join(LMUDataRoot(), 'images', 'memlens_images', rel_path),
        ]
        for c in candidates:
            if osp.exists(c):
                return c
        return candidates[0]

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        images = toliststr(line['image_path']) if 'image_path' in line else []
        return [self._resolve_image_path(p) for p in images]

    def _image_ready(self):
        if 'image_path' not in self.data.columns:
            return True
        checked = 0
        for _, row in self.data.iterrows():
            for p in toliststr(row['image_path']):
                checked += 1
                if not osp.exists(self._resolve_image_path(p)):
                    return False
                if checked >= 32:
                    return True
        return True

    def _prepare_images(self):
        if self._image_ready():
            return
        logger = get_logger('MemLens')
        img_root = self._img_root()
        os.makedirs(img_root, exist_ok=True)

        logger.info(f'Downloading MemLens image archive: {self.IMAGE_ARCHIVE}')
        tar_path = hf_hub_download(
            repo_id=self.SOURCE_HF_REPO_ID,
            filename=self.IMAGE_ARCHIVE,
            repo_type='dataset',
        )
        logger.info(f'Extracting MemLens images to {img_root}')
        with tarfile.open(tar_path, 'r:gz') as tar:
            _safe_extract(tar, img_root)

        if not self._image_ready():
            raise FileNotFoundError(
                'MemLens images are still missing after extraction. '
                f'Please check MEMLENS_IMAGE_ROOT={os.environ.get("MEMLENS_IMAGE_ROOT", "")} '
                f'or {self.SOURCE_HF_REPO_ID}/{self.IMAGE_ARCHIVE}.')

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        images = self.dump_image(line)
        text = str(line['question'])

        token_pattern = r'(<image token>|<image>)'
        parts = re.split(token_pattern, text)
        msgs = []
        image_idx = 0

        for part in parts:
            if part in ['<image token>', '<image>']:
                if image_idx < len(images):
                    msgs.append(dict(type='image', value=images[image_idx]))
                    image_idx += 1
            elif part:
                msgs.append(dict(type='text', value=part))

        if image_idx != len(images):
            raise ValueError(
                f'MemLens image mismatch: consumed {image_idx} image tokens, '
                f'but image_path has {len(images)} images.'
            )

        return [m for m in msgs if m.get('value')]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        data = load(eval_file)

        judge_name = judge_kwargs.pop('model', cls.DEFAULT_JUDGE)
        nproc = judge_kwargs.pop('nproc', 8)
        use_cache = judge_kwargs.pop('use_cache', True)
        max_samples = judge_kwargs.pop('max_samples', None)
        judge_kwargs.setdefault('max_tokens', 2048)
        judge_model = build_judge(model=judge_name, **judge_kwargs)
        if hasattr(judge_model, 'working') and not judge_model.working():
            raise RuntimeError(f'MemLens LLM judge {judge_name} is not working.')

        # Map VLMEvalKit column names -> MemLens judge expected keys.
        def cell_str(row, key, default=''):
            value = row.get(key, default)
            if pd.isna(value):
                return default
            return str(value)

        records = []
        for _, row in data.iterrows():
            records.append({
                'question_id': cell_str(row, 'question_id', cell_str(row, 'index', '')),
                'question': cell_str(row, 'question', ''),
                'question_type': cell_str(row, 'question_type', 'unknown'),
                'question_subtype': cell_str(row, 'question_subtype', ''),
                'old_answer': cell_str(row, 'old_answer', ''),
                'reference_answer': cell_str(row, 'answer', ''),
                'prediction': cell_str(row, 'prediction', ''),
                'parsed_output': cell_str(row, 'parsed_output', ''),
                'output_len': row.get('output_len', 0),
            })

        jsonl_file = get_intermediate_file_path(eval_file, f'_{judge_name}_memlens_judge', 'jsonl')
        metrics, details = memlens_judge_eval(
            data=records,
            jsonl_path=jsonl_file,
            use_cache=use_cache,
            max_samples=max_samples,
            num_workers=nproc,
            judge_model=judge_model,
            judge_name=judge_name,
        )

        # Save per-sample detail
        detail_file = get_intermediate_file_path(eval_file, f'_{judge_name}_memlens_eval', 'xlsx')
        dump(pd.DataFrame(details), detail_file)

        # Keep the main score table compact; diagnostics stay in the detail file.
        o = metrics['overall']
        rows = [
            {'question_type': 'overall', 'metric': 'accuracy', 'score': o['accuracy'], 'num': o['total']},
        ]

        for qtype, s in sorted(metrics['by_question_type'].items()):
            rows.append({'question_type': qtype, 'metric': 'accuracy', 'score': s['accuracy'], 'num': s['count']})

        score = pd.DataFrame(rows)
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        logger.info(f'MemLens LLM judge evaluation done -> {score_file}')
        return score
