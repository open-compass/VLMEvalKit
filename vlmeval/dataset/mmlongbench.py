import json
import os
import os.path as osp
import re
import tarfile

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge
from .utils.mmlongbench_metrics import calculate_metrics, parse_output
from ..smp import dump, get_logger, load, toliststr
from ..smp.file import get_intermediate_file_path
from ..smp.file import LMUDataRoot


class MMLongBench(ImageBaseDataset):
    """MMLongBench full benchmark downsampled to selected context lengths."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE = 'gpt-5.5-2026-04-24'
    JUDGE_FORMAT = None
    RATING_FORMAT = '{model_name}_{dataset_name}_score.csv'
    HF_REPO_ID = 'ZhaoweiWang/MMLongBench'
    DATASET_URL = {
        'MMLongBench_32K': 'https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/resolve/main/vlmevalkit/MMLongBench_32K.tsv',  # noqa: E501
        'MMLongBench_128K': 'https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/resolve/main/vlmevalkit/MMLongBench_128K.tsv',  # noqa: E501
        'MMLongBench_256K': 'https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/resolve/main/vlmevalkit/MMLongBench_256K.tsv',  # noqa: E501
        'MMLongBench_512K': 'https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/resolve/main/vlmevalkit/MMLongBench_512K.tsv',  # noqa: E501
    }
    DATASET_MD5 = {}

    DATASET_SETS = {
        'MMLongBench_32K': ['vrag_32', 'NIAH_32', 'ICL_32', 'summ_32', 'documentQA_32'],
        'MMLongBench_128K': ['vrag_128', 'NIAH_128', 'ICL_128', 'summ_128', 'documentQA_128'],
        'MMLongBench_256K': ['vrag_256_sr50', 'NIAH_256_sr25', 'ICL_256_sr50', 'summ_256_sr50', 'documentQA_256_sr50'],  # noqa: E501
        'MMLongBench_512K': ['vrag_512_sr50', 'NIAH_512_sr25', 'ICL_512_sr50', 'summ_512_sr50', 'documentQA_512_sr50'],  # noqa: E501
    }

    DEFAULT_ANSWER = [0, 1, 1, 0]
    IMAGE_ARCHIVES = {
        'MMLongBench_32K': ['vlmevalkit/MMLongBench_32K_images.tar.gz'],
        'MMLongBench_128K': ['vlmevalkit/MMLongBench_128K_images.tar.gz'],
        'MMLongBench_256K': ['vlmevalkit/MMLongBench_256K_images.tar.gz'],
        'MMLongBench_512K': ['vlmevalkit/MMLongBench_512K_images.tar.gz'],
    }

    def __init__(self, dataset='MMLongBench_32K', **kwargs):
        super().__init__(dataset=dataset, skip_noimg=False)
        if os.environ.get('MMLB_AUTO_DOWNLOAD_IMAGES', '1') == '1':
            self.prepare_images()

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SETS)

    @staticmethod
    def _parse_extra_info(extra_info):
        if isinstance(extra_info, dict):
            return extra_info
        if isinstance(extra_info, str) and extra_info.strip():
            try:
                return json.loads(extra_info)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _json_dumps(value):
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _standard_tsv_path(cls, dataset):
        root = os.environ.get('MMLB_TSV_ROOT', LMUDataRoot())
        return osp.join(root, f'{dataset}.tsv')

    @classmethod
    def _normalize_tsv_data(cls, data):
        if 'image_path' in data:
            data['image_path'] = [toliststr(x) for x in data['image_path']]
        if 'extra_info' in data:
            data['extra_info'] = [
                cls._json_dumps(cls._parse_extra_info(x)) for x in data['extra_info']
            ]
        return data

    def load_data(self, dataset):
        if dataset not in self.DATASET_SETS:
            raise KeyError(f'Unsupported MMLongBench split: {dataset}')

        tsv_path = self._standard_tsv_path(dataset)
        if osp.exists(tsv_path):
            return self._normalize_tsv_data(load(tsv_path))

        url = self.DATASET_URL.get(dataset, '')
        if url:
            file_md5 = self.DATASET_MD5.get(dataset, None)
            return self._normalize_tsv_data(self.prepare_tsv(url, file_md5))

        raise FileNotFoundError(
            f'MMLongBench requires a standard TSV file for {dataset}. '
            f'Expected local file: {tsv_path}. '
            'Alternatively, publish the TSV and set MMLongBench.DATASET_URL/MD5.'
        )

    @staticmethod
    def _safe_extract(tar, path):
        root = osp.abspath(path)
        for member in tar.getmembers():
            target = osp.abspath(osp.join(root, member.name))
            if not (target == root or target.startswith(root + os.sep)):
                raise RuntimeError(f'Unsafe path in tar archive: {member.name}')
        tar.extractall(root)

    @staticmethod
    def _tar_has_top_level_mmlb_image(tar_path):
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                name = member.name.lstrip('./')
                if not name:
                    continue
                return name.split('/', 1)[0] == 'mmlb_image'
        return False

    @classmethod
    def _extract_image_archive(cls, tar_path, output_root):
        has_root = cls._tar_has_top_level_mmlb_image(tar_path)
        extract_root = output_root if has_root else osp.join(output_root, 'mmlb_image')
        os.makedirs(extract_root, exist_ok=True)
        with tarfile.open(tar_path, 'r:gz') as tar:
            cls._safe_extract(tar, extract_root)

    @classmethod
    def _download_image_archive(cls, filename, download_dir):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise ImportError(
                'huggingface_hub is required to auto-download MMLongBench images. '
                'Install it with `pip install huggingface_hub`, or set '
                'MMLB_AUTO_DOWNLOAD_IMAGES=0 and prepare $LMUData/images/mmlb_image manually.'
            ) from err

        return hf_hub_download(
            repo_id=cls.HF_REPO_ID,
            filename=filename,
            repo_type='dataset',
            local_dir=download_dir,
            token=os.environ.get('HF_TOKEN'),
        )

    def _image_ready(self):
        if 'image_path' not in self.data:
            return True
        checked = 0
        for _, line in self.data.iterrows():
            for image_path in toliststr(line['image_path']):
                checked += 1
                if not osp.exists(self._resolve_image_path(line, image_path)):
                    return False
                if checked >= 32:
                    return True
        return True

    def prepare_images(self):
        if self._image_ready():
            return

        logger = get_logger('MMLongBench')
        output_root = osp.join(LMUDataRoot(), 'images')
        download_dir = osp.join(output_root, 'mmlb_image_downloads')
        os.makedirs(download_dir, exist_ok=True)
        logger.warning(
            'MMLongBench images are missing. Downloading official image archives from Hugging Face. '
            'This can take a long time and requires substantial disk space.'
        )
        for archive in self.IMAGE_ARCHIVES.get(self.dataset_name, []):
            tar_path = self._download_image_archive(archive, download_dir)
            self._extract_image_archive(tar_path, output_root)

        if not self._image_ready():
            raise FileNotFoundError(
                f'MMLongBench images are still missing after extraction. '
                f'Expected images under {osp.join(output_root, "mmlb_image")}.'
            )

    def _resolve_image_path(self, line, image_path):
        if osp.isabs(image_path):
            return image_path
        subset = line.get('mmlb_subset', '')
        length = self.dataset_name.rsplit('_', 1)[-1]
        env_img_root = os.environ.get('MMLB_IMAGE_ROOT', '')
        candidates = [
            osp.join(env_img_root, length, image_path) if env_img_root else '',
            osp.join(env_img_root, image_path) if env_img_root else '',
            osp.join(self.img_root, length, image_path),
            osp.join(self.img_root, image_path),
            osp.join(LMUDataRoot(), 'images', 'mmlb_image', length, image_path),
            osp.join(LMUDataRoot(), 'images', 'mmlb_image', image_path),
            osp.join(LMUDataRoot(), 'images', self.dataset_name, image_path),
            osp.join(LMUDataRoot(), 'images', subset, image_path),
            osp.join(LMUDataRoot(), 'images', 'MMLongBench', subset, image_path),
            osp.join(LMUDataRoot(), 'images', 'MMLongBench', image_path),
        ]
        for candidate in candidates:
            if candidate and osp.exists(candidate):
                return candidate
        return candidates[1]

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        images = toliststr(line['image_path']) if 'image_path' in line else []
        return [self._resolve_image_path(line, p) for p in images]

    @staticmethod
    def _append_text(msgs, text):
        if text:
            msgs.append(dict(type='text', value=text))

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
            else:
                self._append_text(msgs, part)

        if image_idx == 0 and images:
            msgs = [dict(type='image', value=p) for p in images] + msgs
        elif image_idx < len(images):
            msgs.extend(dict(type='image', value=p) for p in images[image_idx:])

        return [m for m in msgs if m['value']]

    @staticmethod
    def _metric_for(row):
        task = row['task']
        source = row.get('source_dataset', '')
        if task == 'vrag':
            return 'sub_em'
        if task == 'NIAH':
            if source.startswith('vh_'):
                return 'binary_acc'
            if 'counting' in source:
                return 'soft_acc'
            if 'retrieval' in source or 'reasoning' in source:
                # image variants are multiple-choice, text variants are open-ended
                return 'mc_acc' if 'image' in source else 'sub_em'
        if task == 'ICL':
            return 'cls_acc'
        if task == 'summ':
            return 'rouge'
        if task == 'documentQA':
            return 'doc_qa_llm'
        raise KeyError(f'Cannot infer MMLongBench metric for task={task}, source_dataset={source}')

    @staticmethod
    def _prefix_for(task):
        if task == 'summ':
            return 'Summary:'
        if task == 'ICL':
            return 'label:'
        return 'Answer:'

    @classmethod
    def _score_one(cls, row, row_id, judge_model=None):
        prediction = str(row.get('prediction', ''))
        extra = cls._parse_extra_info(row.get('extra_info'))
        # extra_info keeps the typed answer (int / list / str); the TSV column is stringified.
        answer = extra.get('answer', row['answer'])
        task = row['task']
        prefix = cls._prefix_for(task)

        # documentQA is always scored by the LLM judge (doc_qa_llm).
        if task == 'documentQA':
            if judge_model is None:
                raise ValueError('documentQA requires an LLM judge for doc_qa_llm scoring.')
            parsed = parse_output(prediction, prefix=prefix)
            parsed_pred = parsed if parsed is not None else prediction
            answer_format = extra.get('answer_format', 'String')
            metric = 'doc_qa_llm'
            extra_info = {
                'llm_judge_client': judge_model,
                'answer_format': answer_format,
                'question': extra.get('question', ''),
            }
            try:
                mets = calculate_metrics(parsed_pred, answer, 'doc_qa_llm', extra_info=extra_info)
                judge_result = mets['doc_qa_llm']['judge_result']
                mets = {
                    'doc_qa_llm': float(mets['doc_qa_llm']['final_score']),
                    'judge_raw': judge_result.get('raw_output', ''),
                }
            except Exception as err:
                mets = {'doc_qa_llm': 0.0, 'metric_error': f'{type(err).__name__}: {err}'}

            return metric, parsed_pred, mets

        metric = cls._metric_for(row)

        # binary_acc (visual haystack): uses the RAW prediction + a rotating default answer.
        if metric == 'binary_acc':
            default_answer = cls.DEFAULT_ANSWER[row_id % len(cls.DEFAULT_ANSWER)]
            try:
                mets = calculate_metrics((prediction, default_answer), answer, 'binary_acc')
            except Exception as err:
                mets = {'acc': 0, 'metric_error': f'{type(err).__name__}: {err}'}
            return metric, prediction, mets

        # default_post_process: score both the raw and parsed prediction, keep the max.
        try:
            mets = calculate_metrics(prediction, answer, metric)
            parsed = parse_output(prediction, prefix=prefix)
            if parsed is not None:
                new_mets = calculate_metrics(parsed, answer, metric)
                mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
            parsed_pred = parsed if parsed is not None else prediction
        except Exception as err:
            mets = {metric: 0.0, 'metric_error': f'{type(err).__name__}: {err}'}
            parsed_pred = prediction
        return metric, parsed_pred, mets

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _score_key_for(metric):
        if metric == 'rouge':
            return 'rougeLsum_f1'
        if metric == 'binary_acc':
            return 'acc'
        return metric

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        data = load(eval_file)

        # documentQA is scored only by an LLM judge (doc_qa_llm).
        judge_model = None
        has_docqa = 'task' in data.columns and (data['task'] == 'documentQA').any()
        if has_docqa:
            judge_name = judge_kwargs.get('model', None)
            if judge_name is None:
                raise ValueError('MMLongBench documentQA requires a judge model for doc_qa_llm scoring.')
            else:
                judge_model = build_judge(max_tokens=1024, **judge_kwargs)
                if judge_model is None or (hasattr(judge_model, 'working') and not judge_model.working()):
                    raise RuntimeError(f'Judge {judge_name} is not available for doc_qa_llm scoring.')

        detail_rows = []
        for row_id, (_, row) in enumerate(data.iterrows()):
            jm = judge_model if row['task'] == 'documentQA' else None
            metric, parsed_prediction, metric_res = cls._score_one(row, row_id, judge_model=jm)
            detail = row.to_dict()
            detail.pop('image_path', None)
            detail['metric'] = metric
            detail['parsed_prediction'] = parsed_prediction
            for key, value in metric_res.items():
                detail[key] = value
            score_key = cls._score_key_for(metric)
            detail['score'] = metric_res.get(score_key, 0.0)
            detail_rows.append(detail)

        detail = pd.DataFrame(detail_rows)
        detail_file = get_intermediate_file_path(eval_file, '_mmlb_eval', 'tsv')
        dump(detail, detail_file)

        overall_scores = [float(x) for x in detail['score']]
        overall_score = cls._mean(overall_scores)
        rows = [{
            'task': 'overall',
            'metric': 'avg',
            'num': len(overall_scores),
            'score': overall_score,
        }]
        task_order = ['vrag', 'NIAH', 'ICL', 'summ', 'documentQA']
        for task in task_order:
            task_detail = detail[detail['task'] == task]
            if len(task_detail) == 0:
                continue
            scores = [float(x) for x in task_detail['score']]
            rows.append({
                'task': task,
                'metric': 'avg',
                'num': len(scores),
                'score': cls._mean(scores),
            })

        score = pd.DataFrame(rows)
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        logger.info(f'MMLongBench evaluation finished for {eval_file}; scores saved to {score_file}')
        return score
