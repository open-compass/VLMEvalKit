"""CAPEval dataset for VLMEvalKit.

Caption inference uses the official prompt. Checklist judging uses VLMEvalKit
``build_judge`` with the official CAPEval single-pass protocol (same prompt /
parser / C-P formulas as the paper).

Data is auto-downloaded from Hugging Face to ``$LMUData/CAPEval/``
(``checklist.jsonl`` + ``image/``). No extra clone or ``CAPEVAL_HOME`` is required.
"""

from __future__ import annotations
import os.path as osp
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils import DEBUG_MESSAGE, build_judge
from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load, modelscope_flag_set
from vlmeval.utils import track_progress_rich
from .utils.capeval.evaluator import (CAPEval_atomeval, assemble_records, captions_from_eval_file,
                                      load_cached_answers, pending_units, prepare_units,
                                      records_to_score_df)
from .utils.capeval.prompts import (CAPTION_PROMPT, PROMPT_VERSION, SINGLE_PASS_SYSTEM,
                                    build_single_pass_user_prompt)
from .utils.capeval.schema import EVALUATOR_VERSION, load_jsonl

HF_REPO = 'LiuzhipengUCAS/CAPEval'
EXPECTED_N_IMAGES = 300
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}


def _data_root() -> Path:
    return Path(LMUDataRoot()) / 'CAPEval'


def _count_images(image_dir: Path) -> int:
    if not image_dir.is_dir():
        return 0
    return sum(
        1 for p in image_dir.iterdir()
        if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in IMAGE_SUFFIXES
    )


def locate_capeval_files(root: Path) -> Optional[Tuple[Path, Path]]:
    """Return ``(checklist.jsonl, image_dir)`` if a complete copy exists under ``root``."""
    checklist_cands = [
        root / 'checklist.jsonl',
        root / 'data' / 'checklist.jsonl',
    ]
    image_cands = [
        root / 'image',
        root / 'images',
        root / 'data' / 'image',
        root / 'data' / 'images',
    ]
    checklist = next((p for p in checklist_cands if p.is_file()), None)
    image_dir = next((p for p in image_cands if p.is_dir()), None)
    if checklist is None or image_dir is None:
        return None
    if _count_images(image_dir) < EXPECTED_N_IMAGES:
        return None
    return checklist, image_dir


def _snapshot_download(local_dir: Path) -> str:
    """Same HF / ModelScope pattern as ViewSpatialBench, VsiBench, Video-MME, etc."""
    repo = HF_REPO
    if modelscope_flag_set():
        from modelscope import dataset_snapshot_download
        return dataset_snapshot_download(dataset_id=repo, local_dir=str(local_dir))
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            'CAPEval auto-download needs huggingface_hub (already in VLMEvalKit '
            'requirements.txt). Install it, or place checklist.jsonl + image/ under '
            f'$LMUData/CAPEval/ ({local_dir}).'
        ) from e
    return snapshot_download(repo_id=repo, repo_type='dataset', local_dir=str(local_dir))


def ensure_capeval_data() -> Tuple[Path, Path]:
    """Download CAPEval to ``$LMUData/CAPEval`` if needed.

    Layout (Hugging Face ``LiuzhipengUCAS/CAPEval``)::

        $LMUData/CAPEval/checklist.jsonl
        $LMUData/CAPEval/image/*.jpg|jpeg|png|webp
    """
    root = _data_root()
    found = locate_capeval_files(root)
    if found is not None:
        return found

    root.mkdir(parents=True, exist_ok=True)
    _snapshot_download(root)
    found = locate_capeval_files(root)
    if found is None:
        raise FileNotFoundError(
            f'Failed to prepare CAPEval under {root}. '
            f'Expected checklist.jsonl and {EXPECTED_N_IMAGES} images '
            f'(Hugging Face dataset {HF_REPO}).'
        )
    return found


def _safe_model_name(model_name: str) -> str:
    return str(model_name).replace('/', '_').replace(':', '_')


class CAPEval(ImageBaseDataset):
    """Caption → checklist judge. Dataset name for ``run.py --data``: ``CAPEval``."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE = 'qwen-72b'
    # Upstream hook in inference.py (same as BabyVision / MMESCI / SUPERChem):
    # skip model.build_prompt so the official caption instruction is not replaced.
    force_use_dataset_prompt = True
    DATASET_URL = {'CAPEval': f'https://huggingface.co/datasets/{HF_REPO}'}
    DATASET_MD5 = {}

    def __init__(self, dataset='CAPEval', **kwargs):
        super().__init__(dataset=dataset, skip_noimg=False)

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ['CAPEval']

    def load_data(self, dataset: str = 'CAPEval'):
        checklist_path, image_root = ensure_capeval_data()
        self.checklist_path = str(checklist_path)
        self.image_root = str(image_root)

        rows = []
        for i, obj in enumerate(load_jsonl(str(checklist_path), checklist_rows_only=True)):
            img_path = str(obj.get('img_path') or '').strip()
            if not img_path:
                continue
            abs_img = image_root / Path(img_path).name
            stem = Path(img_path).stem
            prefix = ''.join(ch for ch in stem if ch.isalpha()) or 'unknown'
            rows.append({
                'index': i,
                'image_id': img_path,
                'image_path': str(abs_img),
                'question': CAPTION_PROMPT,
                'category': prefix.upper(),
            })
        if len(rows) < EXPECTED_N_IMAGES:
            warnings.warn(
                f'CAPEval checklist has {len(rows)} image rows; expected {EXPECTED_N_IMAGES}.'
            )
        return pd.DataFrame(rows)

    def build_prompt(self, line) -> List[Dict[str, Any]]:
        # ImageBaseDataset.build_prompt: meta_only → line['image_path']; else dump_image().
        # CAPEval TSV-less rows have image_path only, so meta_only stays True.
        msgs = super().build_prompt(line)
        for m in msgs:
            if m.get('type') == 'image' and not osp.isfile(str(m.get('value', ''))):
                raise FileNotFoundError(f'CAPEval image not found: {m.get("value")}')
        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs) -> Any:
        """Score captions with ``build_judge`` + official CAPEval checklist protocol."""
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('use_vllm', None)
        judge_kwargs.pop('use_verifier', None)

        judge_model = judge_kwargs.pop('model', None) or self.DEFAULT_JUDGE
        safe_model = _safe_model_name(judge_model)
        cache_tag = f'{safe_model}_{PROMPT_VERSION}_{EVALUATOR_VERSION}'

        storage = get_intermediate_file_path(eval_file, f'_{cache_tag}_judge', 'json')
        tmp_file = get_intermediate_file_path(eval_file, f'_{cache_tag}_judge_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{cache_tag}_score', 'csv')

        data = load(eval_file)
        captions = captions_from_eval_file(data)
        if not captions:
            raise RuntimeError(
                f'No captions found in {eval_file}. '
                'Expected a `prediction` column plus `image_id` / `image_path`.'
            )

        checklist_path = getattr(self, 'checklist_path', None)
        if not checklist_path or not osp.isfile(checklist_path):
            checklist_path, _ = ensure_capeval_data()
            checklist_path = str(checklist_path)
        units = prepare_units(captions, checklist_path)

        ans = load_cached_answers(tmp_file, storage)
        todo = pending_units(units, ans)
        if todo:
            model = build_judge(
                model=judge_model,
                temperature=0.0,
                system_prompt=SINGLE_PASS_SYSTEM,
                max_tokens=8192,
                **judge_kwargs,
            )
            assert model.working(), (
                'CAPEval evaluation requires a working judge API '
                f'(default --judge {self.DEFAULT_JUDGE}).\n' + DEBUG_MESSAGE
            )

            tups = [
                (model, build_single_pass_user_prompt(u['caption'], u['checklist_items']),
                 len(u['checklist_items']))
                for u in todo
            ]
            keys = [u['image_id'] for u in todo]
            track_progress_rich(
                CAPEval_atomeval,
                tups,
                nproc=nproc,
                chunksize=nproc,
                keys=keys,
                save=tmp_file,
            )
            ans = load_cached_answers(tmp_file, storage)

        records = assemble_records(units, ans)
        for rec in records:
            rec['judge_model'] = judge_model
            rec['prompt_version'] = PROMPT_VERSION
            rec['evaluator_version'] = EVALUATOR_VERSION
        dump(records, storage)
        ret = records_to_score_df(
            records,
            judge_model=judge_model,
            prompt_version=PROMPT_VERSION,
            evaluator_version=EVALUATOR_VERSION,
        )
        n_error = int(ret.iloc[0]['n_error'])
        n_ok = int(ret.iloc[0]['n_images'])
        if n_error:
            warnings.warn(
                f'CAPEval partial evaluation: {n_error} image(s) failed the judge; '
                f'C/P are aggregated over {n_ok} successful image(s) only, '
                f'not a full {EXPECTED_N_IMAGES}-image score. See eval_status=partial '
                f'and n_error in {score_file}.'
            )
        dump(ret, score_file)
        return ret
