import glob
import json
import os
import os.path as osp
import string

import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import LMUDataRoot, dump, get_cache_path, load, toliststr
from .image_mcq import ImageMCQDataset
from .utils.spatial_bench.cal_scores import eval_mcq_score
from .utils.spatial_bench.matching_func import can_match_option

MV_ROBOBENCH_REPO_ID = 'AaronFengZY24/MV_Robobench'

MV_ROBOBENCH_CATEGORIES = [
    'robotic_action_planning',
    'robotic_affordance_recognition',
    'robotic_step_execution',
    'robotic_trajectory_selection',
    'spatial_3d_spatial_consistency',
    'spatial_cross_view_matching',
    'spatial_distance_judgement',
    'spatial_viewpoint_identification',
]

MV_ROBOBENCH_SOURCES = ['agiworld', 'bridgev2']


def _build_records(snapshot_path):
    # Flatten the official QA.json tree (source/category/qa_id/QA.json + view*.png)
    # into TSV rows. Image paths are stored relative to the snapshot root so that
    # the TSV stays valid wherever the snapshot is cached.
    records = []
    qa_files = sorted(glob.glob(osp.join(snapshot_path, '*', '*', '*', 'QA.json')))
    assert len(qa_files) > 0, f'No QA.json found under {snapshot_path}'

    for qa_file in qa_files:
        with open(qa_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        rel_dir = osp.relpath(osp.dirname(qa_file), snapshot_path)
        source, category, qa_id = rel_dir.split(osp.sep)

        image_paths, view_labels = [], []
        for img in meta.get('images', []):
            image_paths.append(osp.normpath(osp.join(rel_dir, str(img['path']))))
            view_labels.append(str(img.get('illustration', '')))

        row = {
            'index': len(records),
            'question': meta['question'],
            'image_path': str(image_paths),
            'image_view_labels': str(view_labels),
            'category': category,
            'source': source,
            'qa_id': qa_id,
            'task_type': str(meta.get('type', '')),
            'answer': str(meta['correct']).strip().upper(),
        }
        for letter in string.ascii_uppercase[:5]:
            row[letter] = str(meta.get('options', {}).get(letter, ''))
        records.append(row)

    return records


def _extract_option(prediction):
    # Extract an A-E letter from the model output; return False (counted wrong) on failure.
    text = '' if prediction is None else str(prediction)
    option = can_match_option(text, choices=list('ABCDE'))
    return option if option in set('ABCDE') else False


def _score_mvrobobench(data: pd.DataFrame) -> pd.DataFrame:
    scored = data.copy()
    predictions, answers, hits = [], [], []
    for _, row in scored.iterrows():
        pred = _extract_option(row.get('prediction', ''))
        answer = str(row.get('answer', '')).strip().upper()
        predictions.append(pred)
        answers.append(answer)
        hits.append(int(pred == answer))
    scored['pred_extracted'] = predictions
    scored['answer_normalized'] = answers
    scored['hit'] = hits
    return scored


def _build_score_fn():

    def score_fn(data: pd.DataFrame) -> pd.DataFrame:
        return _score_mvrobobench(data)

    # eval_mcq_score relies on these attributes to name / cache the scoring artifacts
    score_fn.judge_mode = 'rule'
    score_fn.judge_model = 'extract_matching'
    return score_fn


class MVRoboBench(ImageMCQDataset):
    """MV-RoboBench: multi-view spatial reasoning of VLMs in robotic scenes.

    Reference:
      Seeing Across Views: Benchmarking Spatial Reasoning of
      Vision-Language Models in Robotic Scenes
      https://arxiv.org/abs/2510.19400 (ICLR 2026)
      https://github.com/microsoft/MV-RoboBench

    1,708 MCQ questions over 8 subtasks (4 spatial + 4 robotic), built from
    AgiWorld and BridgeV2 robot demonstrations, each with synchronized
    multi-view images. The TSV is built on first use from the official
    QA.json tree hosted on HuggingFace; images are referenced from the
    local snapshot.
    """

    TYPE = 'MCQ'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['MVRoboBench']

    @staticmethod
    def _require_snapshot(repo_id):
        snapshot_path = get_cache_path(repo_id)
        if snapshot_path is None:
            snapshot_path = snapshot_download(repo_id=repo_id, repo_type='dataset', revision='main')
        return snapshot_path

    def prepare_tsv(self, url, file_md5=None, repo_id=MV_ROBOBENCH_REPO_ID):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        tsv_file = osp.join(data_root, f'{self.dataset_name}.tsv')

        if not osp.isfile(tsv_file):
            records = _build_records(self._require_snapshot(repo_id))
            dump(pd.DataFrame(records), tsv_file)

        self.data_path = tsv_file
        data = load(tsv_file)
        # Localize snapshot-relative image paths to absolute ones. TSVs whose
        # image paths are already absolute (e.g. prepared out of band) are
        # left untouched, and no snapshot download is triggered.
        if 'image_path' in data.columns and len(data) > 0:
            first = toliststr(data['image_path'].iloc[0])[0]
            if not osp.isabs(first):
                snapshot_path = self._require_snapshot(repo_id)
                data['image_path'] = data['image_path'].map(
                    lambda p: [osp.join(snapshot_path,
                                        str(x).strip()) for x in toliststr(p)])
        return data

    def build_prompt(self, line):
        # Mirror the official evaluation branch (build_messages_vision): the text
        # (instruction + question + options) comes first, and every image is
        # preceded by an `Image context: {view label}` line, view by view.
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_paths = toliststr(line['image_path'])
        labels = toliststr(line['image_view_labels'])

        options = []
        for letter in string.ascii_uppercase[:5]:
            value = line.get(letter, '')
            if pd.notna(value) and str(value).strip():
                options.append(f'{letter}. {str(value).strip()}')

        prompt = ('Choose the single best option that answers the question (A-E).\n'
                  "Your response must end with a single line: 'Answer: X'.\n\n"
                  f'Question:\n{str(line["question"]).strip()}\n\n'
                  'Options:\n' + '\n'.join(options))

        messages = [{'type': 'text', 'value': prompt}]
        for pos, path in enumerate(image_paths):
            label = labels[pos] if pos < len(labels) else ''
            if label:
                messages.append({'type': 'text', 'value': f'Image context: {label}'})
            messages.append({'type': 'image', 'value': path})
        return messages

    def evaluate(self, eval_file, **judge_kwargs):
        # Every question has a unique gold letter, so rule-based scoring suffices
        # and no external judge is required. Unlike the official repo, which falls
        # back to option A when extraction fails, failures are counted as wrong
        # (stricter). Results are grouped by category (8 subtasks) and source
        # (agiworld / bridgev2), following the paper's reporting protocol.
        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=_build_score_fn(),
            group_col=['category', 'source'],
            order={
                'category': MV_ROBOBENCH_CATEGORIES,
                'source': MV_ROBOBENCH_SOURCES,
            },
            dataset_name=self.dataset_name,
        )
