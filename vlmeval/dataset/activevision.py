import base64
import json
import os
import os.path as osp
import re

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, get_logger, load, toliststr
from .image_base import ImageBaseDataset

# ActiveVision — An Exam for Active Observers (arXiv:2607.16165)
# 85 items / 17 tasks / 3 categories, single image + question, exact-match scoring.
# Grading below is a faithful port of the official eval/lib/scoring.py
# (github.com/saccharomycetes/ActiveVision); question text already instructs the
# model to answer in <answer>...</answer> tags, so build_prompt sends it verbatim.

_INT_PATTERN = re.compile(r'-?\d+')
_ANSWER_TAG_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.S | re.I)


def normalise_answer(ans):
    return re.sub(r'[^A-Z0-9]', '', str(ans or '').upper())


def gt_type_for(ans):
    if isinstance(ans, bool):
        return 'string'
    if isinstance(ans, (int, np.integer)):
        return 'int'
    if isinstance(ans, str) and ans.strip().lstrip('-').isdigit():
        return 'int'
    return 'string'


def extract_predicted_answer(model_text, gt_type):
    text = str(model_text or '').strip()
    tags = _ANSWER_TAG_PATTERN.findall(text)
    if tags:
        text = tags[-1].strip()
    if gt_type == 'int':
        nums = _INT_PATTERN.findall(text)
        return nums[-1] if nums else ''
    marker = re.search(r'(?im)(?:final answer|answer)\s*[:=]\s*(.+?)\s*$', text)
    if marker:
        return marker.group(1).strip()
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ''


def is_correct(pred, gt):
    p = normalise_answer(pred)
    g = normalise_answer(gt)
    if not p:
        return False
    if p == g:
        return True
    try:
        return int(p) == int(g)
    except (ValueError, TypeError):
        return False


class ActiveVisionDataset(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {'ActiveVision': ''}
    DATASET_MD5 = {}
    HF_REPO = 'activevisionai/ActiveVision'

    def load_data(self, dataset):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        tsv_path = osp.join(data_root, f'{dataset}.tsv')

        if osp.exists(tsv_path):
            return load(tsv_path)

        logger = get_logger('ActiveVision')
        logger.info(f'Downloading ActiveVision from HuggingFace: {self.HF_REPO}')

        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=self.HF_REPO, repo_type='dataset', allow_patterns=['data/*'])
        meta_path = osp.join(local_dir, 'data', 'metadata.jsonl')

        rows = []
        with open(meta_path) as f:
            for idx, line in enumerate(raw for raw in f if raw.strip()):
                item = json.loads(line)
                img_path = osp.join(local_dir, 'data', item['file_name'])
                with open(img_path, 'rb') as img_f:
                    image_b64 = base64.b64encode(img_f.read()).decode('ascii')
                rows.append({
                    'index': idx,
                    'id': item['id'],
                    'image': image_b64,
                    'image_path': item['file_name'],
                    'task': item['task'],
                    'category': item['category'],
                    'question': item['question'],
                    'answer': str(item['answer']),
                })

        data = pd.DataFrame(rows)
        dump(data, tsv_path)
        logger.info(f'ActiveVision data saved to {tsv_path} ({len(data)} samples)')
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        msgs = [dict(type='image', value=p) for p in (tgt_path if isinstance(tgt_path, list) else [tgt_path])]
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        hits, preds = [], []
        for i in range(len(data)):
            item = data.iloc[i]
            gt = str(item['answer'])
            pred = extract_predicted_answer(item['prediction'], gt_type_for(gt))
            preds.append(pred)
            hits.append(int(is_correct(pred, gt)))
        data['extracted'] = preds
        data['hit'] = hits

        storage = get_intermediate_file_path(eval_file, '_score', 'xlsx')
        detail = data.copy()
        if 'image' in detail:
            detail = detail.drop(columns=['image'])
        dump(detail, storage)

        result = {'Overall': np.mean(data['hit']) * 100}
        for cat in sorted(data['category'].unique()):
            result[cat] = np.mean(data[data['category'] == cat]['hit']) * 100
        for task in sorted(data['task'].unique()):
            result[task] = np.mean(data[data['task'] == task]['hit']) * 100

        score_df = pd.DataFrame([result])
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(score_df, score_file)
        return score_df
