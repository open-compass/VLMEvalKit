import os
import os.path as osp
import re
import shutil
import string
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import numpy as np
import pandas as pd

from vlmeval.smp import (LMUDataRoot, atomic_write_audio_file, dump, get_intermediate_file_path,
                         get_logger, load)
from .audio_base import AudioBaseDataset, audio_read_ok, audio_reuse_ok
from .utils import DEBUG_MESSAGE, build_judge

logger = get_logger(__name__)

MMSU_AUDIO_EXTENSIONS = {
    '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.oga', '.opus', '.aac'
}


def _clean_text(value):
    if value is None or pd.isna(value):
        return ''
    return str(value).replace('\t', ' ').strip()


def _safe_name(value, fallback):
    value = _clean_text(value) or str(fallback)
    value = re.sub(r'[^0-9A-Za-z_.-]+', '_', value).strip('._')
    return value or str(fallback)


def _canonical_text(value):
    value = _clean_text(value).lower()
    value = re.sub(r'\s+', ' ', value)
    value = re.sub(r'^[a-d][\).:\s-]+', '', value)
    return value.strip()


def _normalise_answer(answer, choices):
    answer = _clean_text(answer)
    upper = answer.upper()
    if upper in choices:
        return upper

    match = re.match(r'^\(?([A-D])\)?(?:[\).:\s-]|$)', answer, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in choices:
            return candidate

    match = re.match(r'^choice[_\s-]*([A-D])$', answer, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in choices:
            return candidate

    if answer in {'0', '1', '2', '3'}:
        candidate = string.ascii_uppercase[int(answer)]
        if candidate in choices:
            return candidate

    norm_answer = _canonical_text(answer)
    matches = [letter for letter, text in choices.items() if _canonical_text(text) == norm_answer]
    if len(matches) == 1:
        return matches[0]
    return ''


def _audio_extension(audio):
    if isinstance(audio, dict) and audio.get('path'):
        suffix = osp.splitext(str(audio['path']))[1].lower()
        if suffix in MMSU_AUDIO_EXTENSIONS:
            return suffix
    if isinstance(audio, str):
        suffix = osp.splitext(audio)[1].lower()
        if suffix in MMSU_AUDIO_EXTENSIONS:
            return suffix
    return '.wav'


def _write_wav_from_array(array, sampling_rate, target):
    from scipy.io import wavfile

    arr = np.asarray(array)
    if arr.dtype.kind == 'f':
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * np.iinfo(np.int16).max).astype(np.int16)
    atomic_write_audio_file(target, lambda f: wavfile.write(f, int(sampling_rate), arr))


def _copy_audio_file(source, target):
    def writer(output):
        with open(source, 'rb') as input_file:
            shutil.copyfileobj(input_file, output)

    atomic_write_audio_file(target, writer)


def _resolve_source_path(path, source_root=None):
    path = str(path)
    if osp.isabs(path):
        return path
    if source_root is not None:
        return osp.join(source_root, path)
    return path


def _export_hf_audio(audio, audio_dir, audio_name, source_root=None):
    if audio is None:
        raise ValueError('MMSU sample has empty audio field.')

    if isinstance(audio, dict):
        suffix = _audio_extension(audio)
        target = osp.join(audio_dir, audio_name + suffix)
        if audio_read_ok(target):
            return osp.basename(target)

        audio_bytes = audio.get('bytes')
        if audio_bytes:
            atomic_write_audio_file(target, lambda f: f.write(audio_bytes))
            return osp.basename(target)

        audio_path = audio.get('path')
        if audio_path and osp.isfile(_resolve_source_path(audio_path, source_root)):
            _copy_audio_file(_resolve_source_path(audio_path, source_root), target)
            return osp.basename(target)

        if audio.get('array') is not None and audio.get('sampling_rate') is not None:
            target = osp.join(audio_dir, audio_name + '.wav')
            if not audio_read_ok(target):
                _write_wav_from_array(audio['array'], audio['sampling_rate'], target)
            return osp.basename(target)

    if isinstance(audio, (bytes, bytearray)):
        target = osp.join(audio_dir, audio_name + '.wav')
        if not audio_read_ok(target):
            atomic_write_audio_file(target, lambda f: f.write(audio))
        return osp.basename(target)

    if isinstance(audio, str):
        source_path = _resolve_source_path(audio, source_root)
        if osp.isfile(source_path):
            suffix = _audio_extension(audio)
            target = osp.join(audio_dir, audio_name + suffix)
            if not audio_read_ok(target):
                _copy_audio_file(source_path, target)
            return osp.basename(target)

    raise ValueError(f'Unsupported MMSU audio payload type: {type(audio)}')


class MMSUDataset(AudioBaseDataset):
    TYPE = 'MCQ'
    MODALITY = 'AUDIO'
    HF_DATASET = 'ddwang2000/MMSU'
    SPLIT = 'train'
    BREAKDOWN_FIELDS = [
        'category',
        'sub-category',
        'sub-sub-category',
        'task_name',
        'linguistics_sub_discipline',
    ]

    @classmethod
    def supported_datasets(cls):
        return ['MMSU']

    def load_data(self, dataset):
        assert dataset == 'MMSU', dataset
        data_root = LMUDataRoot()
        tsv_path = osp.join(data_root, f'{dataset}.tsv')
        self.data_path = tsv_path
        if osp.exists(tsv_path):
            logger.info(f'Loading MMSU from {tsv_path}')
            return load(tsv_path)
        return self._download_and_convert(tsv_path)

    def _download_and_convert(self, tsv_path):
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError(
                'Please install datasets to download MMSU: pip install datasets'
            ) from err

        data_root = LMUDataRoot()
        audio_dir = osp.join(data_root, 'audios', 'MMSU')
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(osp.dirname(tsv_path), exist_ok=True)

        logger.info(f'Downloading MMSU from Hugging Face dataset {self.HF_DATASET}...')
        hf_data, source_root = self._load_hf_data(load_dataset)

        records = []
        bad_answers = []
        used_audio_names = set()
        for i, sample in enumerate(self._iter_samples(hf_data)):
            index = _clean_text(sample.get('id')) or str(i)
            audio_name = _safe_name(index, i)
            if audio_name in used_audio_names:
                audio_name = f'{audio_name}_{i}'
            used_audio_names.add(audio_name)

            choices = {
                'A': _clean_text(sample.get('choice_a')),
                'B': _clean_text(sample.get('choice_b')),
                'C': _clean_text(sample.get('choice_c')),
                'D': _clean_text(sample.get('choice_d')),
            }
            answer_raw = _clean_text(sample.get('answer_gt'))
            answer = _normalise_answer(answer_raw, choices)
            if answer == '':
                bad_answers.append((index, answer_raw))

            record = {
                'index': index,
                'audio_path': _export_hf_audio(
                    sample.get('audio'), audio_dir, audio_name, source_root=source_root),
                'question': _clean_text(sample.get('question')),
                'A': choices['A'],
                'B': choices['B'],
                'C': choices['C'],
                'D': choices['D'],
                'answer': answer,
                'answer_raw': answer_raw,
                'task_name': _clean_text(sample.get('task_name')),
                'category': _clean_text(sample.get('category')),
                'sub-category': _clean_text(sample.get('sub-category')),
                'sub-sub-category': _clean_text(sample.get('sub-sub-category')),
                'linguistics_sub_discipline': _clean_text(
                    sample.get('linguistics_sub_discipline')),
            }
            records.append(record)

        if bad_answers:
            preview = ', '.join([f'{idx}: {ans}' for idx, ans in bad_answers[:5]])
            raise ValueError(
                f'Failed to map {len(bad_answers)} MMSU answers to A/B/C/D. '
                f'Examples: {preview}'
            )

        data = pd.DataFrame(records)
        dump(data, tsv_path)
        logger.info(f'Saved MMSU metadata to {tsv_path} ({len(data)} samples)')
        return data

    def _load_hf_data(self, load_dataset):
        try:
            hf_data = load_dataset(self.HF_DATASET, split=self.SPLIT)
            if 'audio' in hf_data.column_names:
                try:
                    from datasets import Audio
                    hf_data = hf_data.cast_column('audio', Audio(decode=False))
                except Exception as err:
                    logger.warning(f'Failed to disable HF audio decoding for MMSU: {err}')
            return hf_data, None
        except Exception as err:
            logger.warning(f'Failed to load MMSU with datasets.load_dataset: {err}')

        try:
            from huggingface_hub import HfApi
        except ImportError as err:
            raise ImportError(
                'Please install huggingface_hub to list MMSU fallback files.'
            ) from err

        repo_dir = self._download_hf_files_direct(HfApi)
        data_dir = osp.join(repo_dir, 'data')
        parquet_files = [
            osp.join(data_dir, name)
            for name in sorted(os.listdir(data_dir))
            if name.endswith('.parquet')
        ]
        if not parquet_files:
            raise FileNotFoundError(f'No MMSU parquet files found under {data_dir}')
        frames = [pd.read_parquet(path) for path in parquet_files]
        return pd.concat(frames, ignore_index=True), repo_dir

    def _download_hf_files_direct(self, HfApi):
        import requests
        from tqdm import tqdm

        endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
        repo_dir = osp.join(
            LMUDataRoot(),
            '.cache',
            'huggingface',
            'manual',
            self.HF_DATASET.replace('/', '__'),
        )
        api = HfApi(endpoint=endpoint)
        all_files = api.list_repo_files(self.HF_DATASET, repo_type='dataset')
        files = [
            path for path in all_files
            if self._should_download_repo_file(path)
        ]
        todo = [
            path for path in files
            if not self._repo_file_exists(repo_dir, path)
        ]
        logger.info(
            f'Downloading {len(todo)} / {len(files)} MMSU repo files via direct resolve URLs...')
        workers = int(os.environ.get('MMSU_DOWNLOAD_WORKERS', 16))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._download_hf_file, requests, endpoint, path, repo_dir)
                for path in todo
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Downloading MMSU files',
            ):
                future.result()
        return repo_dir

    @staticmethod
    def _repo_file_exists(repo_dir, repo_path):
        target = osp.join(repo_dir, repo_path)
        if osp.splitext(repo_path)[1].lower() in MMSU_AUDIO_EXTENSIONS:
            return audio_reuse_ok(target)
        return osp.exists(target) and osp.getsize(target) > 0

    @staticmethod
    def _should_download_repo_file(path):
        suffix = osp.splitext(path)[1].lower()
        return (
            path.startswith('data/') and suffix == '.parquet'
        ) or (
            path.startswith('audio/') and suffix in MMSU_AUDIO_EXTENSIONS
        )

    def _download_hf_file(self, requests, endpoint, repo_path, repo_dir):
        target = osp.join(repo_dir, repo_path)
        if self._repo_file_exists(repo_dir, repo_path):
            return

        os.makedirs(osp.dirname(target), exist_ok=True)
        url = (
            f'{endpoint}/datasets/{self.HF_DATASET}/resolve/main/'
            f'{quote(repo_path, safe="/")}'
        )
        headers = {}
        token = os.environ.get('HF_TOKEN', '').strip()
        if token:
            headers['Authorization'] = f'Bearer {token}'

        def stream_to(output):
            with requests.get(url, headers=headers, stream=True, timeout=(10, 300)) as response:
                response.raise_for_status()
                response_headers = getattr(response, 'headers', {})
                content_length = response_headers.get('Content-Length')
                expected_size = None
                if content_length is not None:
                    try:
                        expected_size = int(content_length)
                    except (TypeError, ValueError):
                        logger.warning(
                            f'Ignoring invalid Content-Length for {repo_path!r}: '
                            f'{content_length!r}'
                        )
                    if expected_size is not None and expected_size < 0:
                        expected_size = None
                written = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        written += len(chunk)
                        if expected_size is not None and written > expected_size:
                            raise IOError(
                                f'MMSU download exceeded Content-Length for {repo_path!r}: '
                                f'expected {expected_size} bytes, received at least {written}.'
                            )
                if expected_size is not None and written != expected_size:
                    raise IOError(
                        f'Incomplete MMSU download for {repo_path!r}: expected '
                        f'{expected_size} bytes, received {written}.'
                    )
                if written == 0:
                    raise IOError(f'MMSU download returned an empty body for {repo_path!r}.')

        if osp.splitext(repo_path)[1].lower() in MMSU_AUDIO_EXTENSIONS:
            atomic_write_audio_file(target, stream_to)
            return

        tmp_target = target + '.tmp'
        try:
            with open(tmp_target, 'wb') as output:
                stream_to(output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(tmp_target, target)
        finally:
            if osp.exists(tmp_target):
                try:
                    os.unlink(tmp_target)
                except OSError:
                    pass

    @staticmethod
    def _iter_samples(hf_data):
        if isinstance(hf_data, pd.DataFrame):
            for _, row in hf_data.iterrows():
                yield row.to_dict()
        else:
            for sample in hf_data:
                yield sample

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        audio_paths = self.dump_audio(line)
        options = {
            cand: line[cand]
            for cand in 'ABCD'
            if cand in line and not pd.isna(line[cand]) and _clean_text(line[cand]) != ''
        }
        options_prompt = ''.join([f'{key}. {value}\n' for key, value in options.items()])
        prompt = (
            'Listen to the audio and answer the following multiple-choice question.\n'
            f'Question: {line["question"]}\n'
            'Options:\n'
            f'{options_prompt}'
            'Respond with only the letter of the correct option.'
        )

        msgs = [dict(type='audio', value=path) for path in audio_paths]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import mcq_vanilla_eval

        nproc = judge_kwargs.pop('nproc', 4)
        model_name = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model_name] if model_name in name_str_map else model_name

        if model_name == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('Judge API is not working properly; falling back to exact matching.')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')
        data = load(eval_file).sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for key in list(data.keys()):
            data[key.lower() if key not in list(string.ascii_uppercase) else key] = data.pop(key)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        for idx in data['index']:
            assert idx in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        for field in self.BREAKDOWN_FIELDS:
            if field not in data and field in meta:
                field_map = {idx: value for idx, value in zip(meta['index'], meta[field])}
                data[field] = [field_map[idx] for idx in data['index']]
        eval_record = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, eval_record)

        acc = self._report_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)
        return acc

    def _report_accuracy(self, data):
        rows = [
            dict(
                dimension='Overall',
                group='Overall',
                num=len(data),
                acc=float(data['hit'].mean()),
            )
        ]
        for field in self.BREAKDOWN_FIELDS:
            if field not in data:
                continue
            values = [_clean_text(x) for x in data[field]]
            for value in sorted(set(values)):
                if value == '':
                    continue
                sub_data = data[[x == value for x in values]]
                rows.append(dict(
                    dimension=field,
                    group=value,
                    num=len(sub_data),
                    acc=float(sub_data['hit'].mean()),
                ))
        return pd.DataFrame(rows)
