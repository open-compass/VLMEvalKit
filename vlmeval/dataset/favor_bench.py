import ast
import json
import os.path as osp
from pathlib import Path

import pandas as pd

from vlmeval.smp import download_file, dump, get_intermediate_file_path, load
from vlmeval.smp.file import LMUDataRoot
from .video_base import VideoBaseDataset

FAVOR_PROMPT = (
    "Carefully watch the video and pay attention to temporal dynamics in this video, "
    "focusing on the camera motions, actions, activities, and interactions. Based on "
    "your observations, select the best option that accurately addresses the question.\n"
    "{question}\n"
    "You can only response with the answer among {options}"
)


class FavorBench(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE_MODEL = None
    HF_REPO_ID = 'zl2048/FAVOR'
    ANNOTATION_FILENAME = 'video_perspective.json'
    ANNOTATION_URL = f'https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{ANNOTATION_FILENAME}'
    HF_VIDEO_PREFIX = 'videos/FAVOR-Bench'

    @classmethod
    def supported_datasets(cls):
        return ['FAVOR-Bench']

    def __init__(self, dataset='FAVOR-Bench', pack=False, nframe=0, fps=1.0):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def _annotation_file(cls, bench_root):
        path = bench_root / cls.ANNOTATION_FILENAME
        if path.exists():
            return path

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                'Please install huggingface_hub to download FavorBench annotations.'
            ) from exc

        try:
            downloaded = hf_hub_download(
                repo_id=cls.HF_REPO_ID,
                repo_type='dataset',
                filename=cls.ANNOTATION_FILENAME,
                local_dir=str(bench_root),
            )
            return Path(downloaded)
        except Exception:
            download_file(cls.ANNOTATION_URL, str(path))
            return path

    @staticmethod
    def _options_from_line(line):
        options = line['options']
        if isinstance(options, str):
            return ast.literal_eval(options)
        return list(options)

    @classmethod
    def _build_tsv(cls, annotation_file, data_file):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for item in data:
            video_name = item['video_name']
            shared = {
                'video': video_name,
                'video_name': video_name,
                'caption': item.get('caption', ''),
                'camera_motion': item.get('camera_motion', ''),
                'subject_attributes': item.get('subject_attributes', ''),
                'motion_list': item.get('motion_list', ''),
                'chronological_motion_list': item.get('chronological_motion_list', ''),
            }
            for question_idx, question in enumerate(item['questions']):
                records.append({
                    'index': len(records),
                    'question_id': question_idx,
                    'question': question['question'],
                    'options': repr(question['options']),
                    'answer': question['correct_answer'],
                    'correct_answer': question['correct_answer'],
                    'task_type': question['task_type'],
                    **shared,
                })

        dump(pd.DataFrame(records), str(data_file))

    @staticmethod
    def _ensure_video_aliases(video_root, data_file):
        data = load(str(data_file))
        for video_name in sorted(set(data['video'])):
            video_name = str(video_name)
            direct = video_root / video_name
            nested = video_root / 'FAVOR-Bench' / video_name
            if not direct.exists() and not direct.is_symlink() and nested.exists():
                direct.symlink_to(Path('FAVOR-Bench') / video_name)

            target = video_root / (video_name + '.mp4')
            if target.exists() or target.is_symlink() or not direct.exists():
                continue
            target.symlink_to(direct.name)

    @staticmethod
    def _videos_ready(video_root, data_file):
        data = load(str(data_file))
        return all((video_root / (str(video_name) + '.mp4')).exists() for video_name in set(data['video']))

    @classmethod
    def _download_missing_videos(cls, bench_root, data_file):
        from urllib.parse import quote

        data = load(str(data_file))
        for video_name in sorted(set(data['video'])):
            video_name = str(video_name)
            target = bench_root / cls.HF_VIDEO_PREFIX / video_name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = (
                f'https://huggingface.co/datasets/{cls.HF_REPO_ID}/resolve/main/'
                f'{cls.HF_VIDEO_PREFIX}/{quote(video_name, safe="")}'
            )
            download_file(url, str(target))

    @classmethod
    def _download_videos(cls, bench_root, data_file):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                'Please install huggingface_hub to download FavorBench videos.'
            ) from exc

        try:
            snapshot_download(
                repo_id=cls.HF_REPO_ID,
                repo_type='dataset',
                allow_patterns=f'{cls.HF_VIDEO_PREFIX}/*.mp4',
                local_dir=str(bench_root),
            )
        except Exception:
            cls._download_missing_videos(bench_root, data_file)

    def prepare_dataset(self, dataset):
        lmu_root = Path(LMUDataRoot())
        bench_root = lmu_root / dataset
        bench_root.mkdir(parents=True, exist_ok=True)

        data_file = bench_root / f'{dataset}.tsv'
        annotation_file = self._annotation_file(bench_root)
        if (
            not data_file.exists()
            or annotation_file.stat().st_mtime > data_file.stat().st_mtime
        ):
            self._build_tsv(annotation_file, data_file)

        video_root = bench_root / 'videos'
        video_root.mkdir(parents=True, exist_ok=True)
        self._ensure_video_aliases(video_root, data_file)
        if not self._videos_ready(video_root, data_file):
            self._download_videos(bench_root, data_file)
            self._ensure_video_aliases(video_root, data_file)
        if not self._videos_ready(video_root, data_file):
            raise FileNotFoundError(
                f'FavorBench videos are incomplete under {video_root}. '
                f'Expected files are listed in {data_file}.'
            )
        return dict(root=str(video_root), data_file=str(data_file))

    def build_prompt(self, line, video_llm=False, **kwargs):
        if isinstance(line, int):
            line = self.data.iloc[line]

        message = []
        video_name = str(line['video'])
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, video_name + '.mp4')))
        else:
            frames = self.save_video_frames(video_name)
            for frame in frames:
                message.append(dict(type='image', value=frame))

        prompt = FAVOR_PROMPT.format(question=line['question'], options=line['options'])
        message.append(dict(type='text', value=prompt))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        scores = {'ALL': [0, len(data)]}

        judges = []
        for _, row in data.iterrows():
            task_type = row['task_type']
            scores.setdefault(task_type, [0, 0])
            scores[task_type][1] += 1

            correct_answer = row['correct_answer']
            output = str(row.get('prediction', ''))
            options = self._options_from_line(row)
            containing_options = [
                opt for opt in options
                if opt != correct_answer and correct_answer in opt
            ]

            judge = False
            if correct_answer.lower() in output.lower():
                judge = not any(opt.lower() in output.lower() for opt in containing_options)

            judges.append(judge)
            if judge:
                scores[task_type][0] += 1
                scores['ALL'][0] += 1

        data['judge'] = judges
        detail_file = get_intermediate_file_path(eval_file, '_judge', 'xlsx')
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(data, detail_file)

        rating = {key: round(value[0] / value[1] * 100, 2) if value[1] else 0.0
                  for key, value in scores.items()}
        dump(rating, score_file)
        return rating

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'ALL' in metrics:
            return {'ALL': metrics['ALL']}
        return {}
