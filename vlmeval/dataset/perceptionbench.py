import base64
import json
import os
import os.path as osp
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from vlmeval.smp import LMUDataRoot, download_file, dump, get_intermediate_file_path, load, read_ok
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import build_judge

PERCEPTIONBENCH_DATASET = 'PerceptionBench'
PERCEPTIONBENCH_URL = (
    'https://huggingface.co/datasets/moonshotai/PerceptionBench/'
    'resolve/main/PerceptionBench.jsonl'
)
PH = re.compile(r'<\|image_(\d+)\|>')
_HERE = osp.dirname(osp.abspath(__file__))
JUDGE_TEMPLATE = open(
    osp.join(_HERE, 'utils', 'perceptionbench_judge_prompt.txt'),
    encoding='utf-8',
).read()


def _safe_model_name(model_name):
    return str(model_name).replace('/', '_').replace(':', '_')


def _json_safe(value):
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _coerce_images(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith('[') and value.endswith(']'):
            try:
                data = json.loads(value)
            except json.JSONDecodeError:
                return []
            return [str(x) for x in data] if isinstance(data, list) else []
        return [value]
    return []


def _line_value(line, key, default=''):
    if isinstance(line, dict):
        return line.get(key, default)
    return line[key] if key in line else default


def _reference_answer(value):
    if isinstance(value, dict):
        return value.get('answer', '')
    if isinstance(value, str) and value.strip().startswith('{'):
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return data.get('answer', value)
        except json.JSONDecodeError:
            pass
    return value


def _normalize_escape(s):
    wl = ' 0123456789-'
    s = re.sub(rf'\\+[{wl}]', r'\\\\' + ' ', s)
    s = re.sub(rf'\\+(?![{wl}])', r'\\', s)
    return re.sub(r'\\+n', '\n', s)


def _decode_judge(resp):
    resp = str(resp).strip()
    if '[reason]' not in resp or '[judge]' not in resp:
        return False, 'No [reason] or [judge] in output'
    reason = resp.split('[judge]')[0].split('[reason]')[-1].strip()
    verdict = resp.split('[judge]')[-1].strip()
    return 'true' in verdict.lower(), reason


def _is_failed(pred):
    if pred is None:
        return True
    if not isinstance(pred, str):
        return False
    pred = pred.strip()
    return not pred or pred.startswith('[error]') or pred.startswith('Failed to obtain answer')


def _build_judge_prompt(question, pred, reference):
    ref = str(reference if reference is not None else '').strip()
    return (
        JUDGE_TEMPLATE
        .replace('{problem}', _normalize_escape(str(question)).strip())
        .replace('{reference_answer}', _normalize_escape(ref).strip())
        .replace('{assistant_answer}', _normalize_escape(str(pred)).strip())
    )


def _data_url_parts(url):
    url = str(url)
    if not (url.startswith('data:') and ';base64,' in url):
        return None, None
    header, payload = url.split(',', 1)
    return header[len('data:'):].split(';')[0], base64.b64decode(payload)


class PerceptionBench(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DATASET_URL = {PERCEPTIONBENCH_DATASET: PERCEPTIONBENCH_URL}
    DEFAULT_JUDGE_MODEL = 'gpt-oss-120b'
    force_use_dataset_prompt = True

    def __init__(self, dataset=PERCEPTIONBENCH_DATASET, skip_noimg=False):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    @classmethod
    def supported_datasets(cls):
        return [PERCEPTIONBENCH_DATASET]

    def load_data(self, dataset):
        assert dataset == PERCEPTIONBENCH_DATASET, dataset
        jsonl_path = self._find_jsonl(dataset)
        records = []
        with open(jsonl_path, encoding='utf-8') as f:
            for row_id, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                answer = _reference_answer(rec.get('answer', ''))
                images = rec.get('image') if isinstance(rec.get('image'), list) else []
                image_paths = self.dump_image({
                    'index': rec.get('index', row_id),
                    'image': images,
                })
                records.append({
                    'index': rec.get('index', row_id),
                    'problem': str(rec.get('problem', '') or ''),
                    'question': str(rec.get('problem', '') or ''),
                    'answer': answer,
                    'image_paths': json.dumps(image_paths, ensure_ascii=False),
                    'hint': rec.get('hint', ''),
                    'error_category': rec.get('error_category', ''),
                    'source_bmk': rec.get('source_bmk', ''),
                    'source_idx': rec.get('source_idx', ''),
                })
        return pd.DataFrame(records)

    @staticmethod
    def _data_root():
        return Path(LMUDataRoot())

    @classmethod
    def _find_jsonl(cls, dataset):
        url = cls.DATASET_URL[dataset]
        target = cls._data_root() / 'perception_bench' / f'{dataset}.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        target = str(target)
        if osp.exists(target):
            return target
        download_file(url, target)
        return target

    def _record_from_line(self, line):
        return {
            'index': _json_safe(_line_value(line, 'index')),
            'problem': str(_line_value(line, 'problem', _line_value(line, 'question', '')) or ''),
            'image': _coerce_images(_line_value(line, 'image', '[]')),
            'image_paths': _coerce_images(_line_value(line, 'image_paths', '[]')),
            'answer': _line_value(line, 'answer', ''),
            'error_category': _line_value(line, 'error_category', ''),
            'source_bmk': _line_value(line, 'source_bmk', ''),
        }

    def dump_image(self, line):
        paths = _coerce_images(_line_value(line, 'image_paths', '[]'))
        if paths:
            return paths
        rec = self._record_from_line(line)
        return self._dump_images(rec['index'], rec['image'])

    def _dump_images(self, index, images):
        os.makedirs(self.img_root, exist_ok=True)
        paths = []
        safe_index = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(index))
        for i, img in enumerate(images):
            mime, raw = _data_url_parts(img)
            if raw is None:
                paths.append(img)
                continue
            ext = {
                'image/jpeg': 'jpg',
                'image/jpg': 'jpg',
                'image/png': 'png',
                'image/webp': 'webp',
            }.get(mime, 'jpg')
            path = osp.join(self.img_root, f'{safe_index}_{i}.{ext}')
            if not read_ok(path):
                with open(path, 'wb') as f:
                    f.write(raw)
            paths.append(path)
        return paths

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        rec = self._record_from_line(line)
        imgs = self.dump_image(line)
        prob = rec['problem']
        msgs, last, used = [], 0, set()

        for m in PH.finditer(prob):
            n = int(m.group(1))
            if 1 <= n <= len(imgs):
                used.add(n - 1)
                seg = prob[last:m.start()]
                if seg:
                    msgs.append(dict(type='text', value=seg))
                msgs.append(dict(type='image', value=imgs[n - 1]))
                last = m.end()
        if prob[last:]:
            msgs.append(dict(type='text', value=prob[last:]))
        for k, img in enumerate(imgs):
            if k not in used:
                msgs.append(dict(type='image', value=img))
        if not msgs:
            msgs.append(dict(type='text', value=prob))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) if pd.notna(x) else '' for x in data['prediction']]
        data['answer'] = [_reference_answer(x) for x in data['answer']]

        judge_model_name = judge_kwargs.pop('model', self.DEFAULT_JUDGE_MODEL)
        nproc = int(judge_kwargs.pop('nproc', 16))
        judge_kwargs.setdefault('retry', 4)
        judge_kwargs.setdefault('temperature', 0.3)
        judge_model = build_judge(model=judge_model_name, **judge_kwargs)

        safe_model = _safe_model_name(judge_model_name)
        storage = get_intermediate_file_path(eval_file, f'_{safe_model}_judge')
        tmp_file = get_intermediate_file_path(eval_file, f'_{safe_model}_judge_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{safe_model}_scores', 'json')

        if osp.exists(storage):
            judged = load(storage)
        else:
            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            lines = [data.iloc[i] for i in range(len(data))]
            keys = [str(line['index']) for line in lines if str(line['index']) not in ans]
            tasks = [line for line in lines if str(line['index']) not in ans]
            if tasks:
                task_inputs = [
                    dict(judge_model=judge_model, line=line)
                    for line in tasks
                ]
                track_progress_rich(
                    self._evaluate_single,
                    task_inputs,
                    nproc=nproc,
                    save=tmp_file,
                    keys=keys,
                )
                ans = load(tmp_file)
            judged = data.copy()
            judged['judge_result'] = [ans[str(idx)]['judge_result'] for idx in judged['index']]
            judged['judge_reason'] = [ans[str(idx)]['judge_reason'] for idx in judged['index']]
            judged['judge_response'] = [ans[str(idx)]['judge_response'] for idx in judged['index']]
            dump(judged, storage)

        metrics = self.compute_metrics(judged)
        dump(metrics, score_file)
        return metrics

    def _evaluate_single(self, judge_model, line):
        prediction = str(line['prediction'])
        rec = self._record_from_line(line)
        if _is_failed(prediction):
            return {
                'judge_result': 0,
                'judge_reason': 'failed to obtain answer',
                'judge_response': '',
            }

        prompt = _build_judge_prompt(rec['problem'], prediction, _reference_answer(rec['answer']))
        try:
            raw = judge_model.generate(prompt)
            if isinstance(raw, dict) and 'prediction' in raw:
                raw = raw['prediction']
            ok, reason = _decode_judge(raw)
            return {
                'judge_result': int(ok),
                'judge_reason': reason,
                'judge_response': raw,
            }
        except Exception as err:
            return {
                'judge_result': 0,
                'judge_reason': f'judge error: {err}',
                'judge_response': '',
            }

    @staticmethod
    def compute_metrics(results):
        tot, cor = Counter(), Counter()
        for _, row in results.iterrows():
            c = row['error_category'] or '?'
            tot[c] += 1
            cor[c] += int(row['judge_result'])
        overall = sum(cor.values()) / max(1, sum(tot.values()))
        return {
            'overall': overall,
            'per_category': {c: cor[c] / tot[c] for c in tot},
        }

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'overall' in metrics:
            return {'overall': metrics['overall']}
        return super().report_primary_metric(metrics)
