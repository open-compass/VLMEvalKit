import json
import os
import os.path as osp
import tarfile
import warnings

import pandas as pd

from .image_base import ImageBaseDataset
from .utils.judge_util import DEBUG_MESSAGE, build_judge
from .utils.longdocurl import build_extraction_prompt, eval_score, parse_extracted_answer
from ..smp import dump, get_intermediate_file_path, get_logger, load, toliststr
from ..smp.file import LMUDataRoot
from ..utils import track_progress_rich


FAIL_MSG = 'Failed to obtain answer via API.'


def _get_line_value(line, key, default=''):
    if key not in line:
        return default
    value = line[key]
    if _is_missing(value):
        return default
    return value


def _is_missing(value):
    return value is None or (isinstance(value, float) and pd.isna(value))


def LongDocURL_auxeval(model, line, system_prompt):
    question = _get_line_value(line, 'question')
    analysis = _get_line_value(line, 'detailed_response') or _get_line_value(line, 'prediction')
    prompt = build_extraction_prompt(question, analysis, system_prompt)
    response = model.generate(prompt, temperature=0)
    if response is None or FAIL_MSG in str(response):
        return {
            'detailed_response': analysis,
            'pred': 'Fail to extract',
            'extracted_answer_format': 'None',
            'extraction_response': response,
            'log': 'Failed to extract due to judge API failure.',
        }
    pred, answer_format = parse_extracted_answer(str(response))
    return {
        'detailed_response': analysis,
        'pred': pred,
        'extracted_answer_format': answer_format,
        'extraction_response': response,
        'log': 'Succeed' if pred != 'Fail to extract' else 'Failed to parse judge response.',
    }


class LongDocURL(ImageBaseDataset):
    """LongDocURL long-document VQA benchmark."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE = 'gpt-5.4-2026-03-05'
    SYSTEM_PROMPT = 'You are an expert in visual document question-answering, please answer our questions based on the given images.\n'

    HF_REPO_ID = 'dengchao/LongDocURL'
    DATA_FILE = 'LongDocURL_public_with_subtask_category.jsonl'
    IMAGE_ARCHIVES = ('png_files_p1.tar.gz', 'png_files_p2.tar.gz')
    DATASET_URL = {
        'LongDocURL': f'https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{DATA_FILE}',
    }
    DATASET_MD5 = {}
    _IMAGE_PREPARE_ATTEMPTED = False

    def __init__(self, dataset='LongDocURL', **kwargs):
        super().__init__(dataset=dataset, skip_noimg=True)

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    @classmethod
    def _tsv_path(cls, dataset):
        root = os.environ.get('LONGDOCURL_TSV_ROOT', LMUDataRoot())
        return osp.join(root, f'{dataset}.tsv')

    @staticmethod
    def _json_dumps(value):
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _parse_jsonish(value, default=None):
        if isinstance(value, (list, dict)):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return default
        return default

    @staticmethod
    def _relative_image_path(path):
        path = str(path)
        marker = '/pdf_pngs/'
        if marker in path:
            return path.split(marker, 1)[1]
        return path.lstrip('/')

    @classmethod
    def _row_from_sample(cls, sample, idx):
        images = [cls._relative_image_path(p) for p in sample.get('images', [])]
        return {
            'index': idx,
            'question_id': sample.get('question_id', ''),
            'question': sample.get('question', ''),
            'answer': cls._json_dumps(sample.get('answer')) if isinstance(sample.get('answer'), list) else sample.get('answer', ''),  # noqa: E501
            'image_path': cls._json_dumps(images),
            'doc_no': sample.get('doc_no', ''),
            'total_pages': sample.get('total_pages', ''),
            'start_end_idx': cls._json_dumps(sample.get('start_end_idx', [])),
            'question_type': sample.get('question_type', ''),
            'answer_format': sample.get('answer_format', ''),
            'task_tag': sample.get('task_tag', ''),
            'evidence_pages': cls._json_dumps(sample.get('evidence_pages', [])),
            'evidence_sources': cls._json_dumps(sample.get('evidence_sources', [])),
            'subTask': cls._json_dumps(sample.get('subTask', [])),
            'detailed_evidences': sample.get('detailed_evidences', ''),
            'pdf_path': sample.get('pdf_path', ''),
        }

    @classmethod
    def _load_jsonl(cls, path):
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @classmethod
    def _download_jsonl(cls):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise ImportError(
                'huggingface_hub is required to download LongDocURL. '
                'Install it with `pip install huggingface_hub`, or set LONGDOCURL_TSV_ROOT.'
            ) from err
        return hf_hub_download(repo_id=cls.HF_REPO_ID, filename=cls.DATA_FILE, repo_type='dataset')

    @classmethod
    def _download_hf_file(cls, filename, local_dir):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise ImportError(
                'huggingface_hub is required to download LongDocURL. '
                'Install it with `pip install huggingface_hub`.'
            ) from err
        os.makedirs(local_dir, exist_ok=True)
        return hf_hub_download(
            repo_id=cls.HF_REPO_ID,
            filename=filename,
            repo_type='dataset',
            local_dir=local_dir,
        )

    def load_data(self, dataset):
        tsv_path = self._tsv_path(dataset)
        if osp.exists(tsv_path):
            data = load(tsv_path)
        else:
            jsonl_path = os.environ.get('LONGDOCURL_JSONL', '') or self._download_jsonl()
            samples = self._load_jsonl(jsonl_path)
            data = pd.DataFrame([self._row_from_sample(sample, i) for i, sample in enumerate(samples)])
            os.makedirs(osp.dirname(tsv_path), exist_ok=True)
            dump(data, tsv_path)

        if 'image_path' in data.columns:
            data['image_path'] = data['image_path'].apply(toliststr)
        return data

    def _img_root(self):
        env = os.environ.get('LONGDOCURL_IMAGE_ROOT', '')
        if env:
            return env
        return osp.join(LMUDataRoot(), 'images', 'LongDocURL', 'pdf_pngs')

    def _image_cache_root(self):
        img_root = self._img_root().rstrip(osp.sep)
        if osp.basename(img_root) == 'pdf_pngs':
            return osp.dirname(img_root)
        return img_root

    def _resolve_image_path(self, rel_path):
        if osp.isabs(rel_path):
            return rel_path
        img_root = self._img_root()
        candidates = [
            osp.join(img_root, rel_path),
            osp.join(LMUDataRoot(), 'images', 'LongDocURL', rel_path),
            osp.join(LMUDataRoot(), 'images', rel_path),
        ]
        for candidate in candidates:
            if osp.exists(candidate):
                return candidate
        return candidates[0]

    @staticmethod
    def _auto_download_images():
        value = os.environ.get('LONGDOCURL_AUTO_DOWNLOAD_IMAGES', '1').lower()
        return value not in {'0', 'false', 'no'}

    @staticmethod
    def _safe_extract(tar_path, target_dir, keep_from=None):
        os.makedirs(target_dir, exist_ok=True)
        target_dir = osp.realpath(target_dir)
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = []
            for member in tar.getmembers():
                name = member.name.lstrip('./')
                if keep_from is not None:
                    parts = name.split('/')
                    if keep_from not in parts:
                        continue
                    name = '/'.join(parts[parts.index(keep_from):])
                    member.name = name
                members.append(member)
            for member in members:
                member_path = osp.realpath(osp.join(target_dir, member.name))
                if member_path != target_dir and not member_path.startswith(target_dir + osp.sep):
                    raise RuntimeError(f'Unsafe path in LongDocURL archive: {member.name}')
            tar.extractall(target_dir, members=members)

    def _extract_target_for_archive(self, tar_path):
        cache_root = self._image_cache_root()
        with tarfile.open(tar_path, 'r:gz') as tar:
            names = [member.name.lstrip('./') for member in tar.getmembers()[:20]]
        if any('pdf_pngs' in name.split('/') for name in names):
            return cache_root, 'pdf_pngs'
        return self._img_root(), None

    def _ensure_images(self, rel_paths):
        if not self._auto_download_images():
            return

        missing = [p for p in rel_paths if not osp.exists(self._resolve_image_path(p))]
        if not missing:
            return

        logger = get_logger('Dataset')
        if self.__class__._IMAGE_PREPARE_ATTEMPTED:
            logger.warning(
                f'LongDocURL images are still missing after the previous preparation attempt. '
                f'Example missing file: {missing[0]}'
            )
            return

        self.__class__._IMAGE_PREPARE_ATTEMPTED = True
        cache_root = self._image_cache_root()
        download_root = osp.join(cache_root, 'downloads')
        logger.warning(
            'LongDocURL images are missing. Downloading public PNG archives from Hugging Face '
            f'to {download_root}. Set LONGDOCURL_AUTO_DOWNLOAD_IMAGES=0 to disable this.'
        )
        for archive in self.IMAGE_ARCHIVES:
            archive_path = self._download_hf_file(archive, download_root)
            extract_target, keep_from = self._extract_target_for_archive(archive_path)
            logger.warning(f'Extracting {archive_path} to {extract_target}')
            self._safe_extract(archive_path, extract_target, keep_from=keep_from)
            missing = [p for p in rel_paths if not osp.exists(self._resolve_image_path(p))]
            if not missing:
                break

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        images = toliststr(line['image_path']) if 'image_path' in line else []
        paths = [self._resolve_image_path(p) for p in images]
        if any(not osp.exists(p) for p in paths):
            self._ensure_images(images)
            paths = [self._resolve_image_path(p) for p in images]
        return paths

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = str(line['question'])
        images = self.dump_image(line)
        text = self.SYSTEM_PROMPT + 'Following is our question: \n' + f'<question>{question}</question>' + '\n'
        msgs = [dict(type='text', value=text)]
        for idx, image in enumerate(images):
            msgs.append(dict(type='text', value=f'Below is the {idx + 1}-th image (total {len(images)} images).\n'))
            msgs.append(dict(type='image', value=image))
        return msgs

    @staticmethod
    def _prediction_for_score(row):
        for key in ['pred', 'parsed_prediction', 'parsed_output', 'prediction']:
            if key in row and not _is_missing(row[key]):
                pred = row[key]
                if isinstance(pred, str) and '<concise_answer>' in pred:
                    pred, _ = parse_extracted_answer(pred)
                return pred
        return ''

    @classmethod
    def _extract_predictions(cls, eval_file, data, model_name, judge_kwargs):
        logger = get_logger('Evaluation')
        if model_name == 'exact_matching' or 'pred' in data:
            data = data.copy()
            if 'pred' not in data:
                data['pred'] = [cls._prediction_for_score(data.iloc[i]) for i in range(len(data))]
            if 'detailed_response' not in data and 'prediction' in data:
                data['detailed_response'] = data['prediction']
            if 'extracted_answer_format' not in data:
                data['extracted_answer_format'] = [''] * len(data)
            if model_name == 'exact_matching':
                logger.warning('LongDocURL uses direct rule-based scoring without answer extraction.')
            return data

        storage = get_intermediate_file_path(eval_file, f'_{model_name}_extract')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model_name}_extract', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if osp.exists(storage):
            logger.warning(f'LongDocURL extraction file {storage} already exists, will reuse it. ')
            return load(storage)

        judge = build_judge(model=model_name, max_tokens=4096, **judge_kwargs)
        if hasattr(judge, 'working') and not judge.working():
            warnings.warn('Judge is not working. LongDocURL answer extraction requires a working judge.\n' + DEBUG_MESSAGE)

        lines = [data.iloc[i] for i in range(len(data))]
        indices = [line['index'] for line in lines]
        ans = load(tmp_file) if osp.exists(tmp_file) else {}
        tasks = [(judge, line, cls.SYSTEM_PROMPT) for line in lines]
        todo_tasks = [x for x, i in zip(tasks, indices) if i not in ans]
        todo_idx = [i for i in indices if i not in ans]
        if len(todo_idx):
            track_progress_rich(
                LongDocURL_auxeval,
                todo_tasks,
                nproc=nproc,
                chunksize=nproc,
                keys=todo_idx,
                save=tmp_file,
            )
            ans = load(tmp_file)

        data = data.copy()
        data['detailed_response'] = [ans[idx]['detailed_response'] for idx in indices]
        data['pred'] = [ans[idx]['pred'] for idx in indices]
        data['extracted_answer_format'] = [ans[idx]['extracted_answer_format'] for idx in indices]
        data['extraction_response'] = [ans[idx]['extraction_response'] for idx in indices]
        data['extraction_log'] = [ans[idx]['log'] for idx in indices]
        dump(data, storage)
        return data

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        data = load(eval_file)
        model_name = judge_kwargs.pop('model', cls.DEFAULT_JUDGE)
        data = cls._extract_predictions(eval_file, data, model_name, judge_kwargs)
        scores = []
        detail_rows = []

        for _, row in data.iterrows():
            pred = cls._prediction_for_score(row)
            answer = row['answer']
            answer_format = row.get('answer_format', '')
            try:
                score = 0.0 if pred == 'Fail to extract' else eval_score(answer, pred, answer_format)
            except Exception as err:
                score = 0.0
                row = row.copy()
                row['metric_error'] = f'{type(err).__name__}: {err}'
            detail = row.to_dict()
            detail['pred_for_eval'] = pred
            detail['score_v3'] = score
            scores.append(float(score))
            detail_rows.append(detail)

        detail = pd.DataFrame(detail_rows)
        detail_file = get_intermediate_file_path(eval_file, '_longdocurl_eval', 'tsv')
        dump(detail, detail_file)

        rows = [{
            'category': 'overall',
            'metric': 'generalized_accuracy',
            'score': sum(scores) / len(scores) if scores else 0.0,
            'num': len(scores),
        }]
        task_tags = []
        if 'task_tag' in data:
            task_tags = sorted([str(x) for x in data['task_tag'].dropna().unique()])
        for task_tag in task_tags:
            task_detail = detail[detail['task_tag'] == task_tag]
            task_scores = [float(x) for x in task_detail['score_v3']]
            rows.append({
                'category': task_tag,
                'metric': 'generalized_accuracy',
                'score': sum(task_scores) / len(task_scores) if task_scores else 0.0,
                'num': len(task_scores),
            })

        score = pd.DataFrame(rows)
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        logger.info(f'LongDocURL evaluation finished for {eval_file}; scores saved to {score_file}')
        return score
