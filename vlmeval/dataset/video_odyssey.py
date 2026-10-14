import ast
import json
import os
import os.path as osp
import re
import shutil
import subprocess
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .video_base import VideoBaseDataset


def _list_value(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]


def _parse_time_seconds(value):
    try:
        parts = [int(part) for part in str(value).strip().split(':')]
    except (TypeError, ValueError):
        return 0
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def _certificate_seconds(value):
    if not value or '-' not in str(value):
        return 0
    start, end = str(value).split('-', 1)
    return max(0, _parse_time_seconds(end) - _parse_time_seconds(start))


def _duration_bin(minutes):
    try:
        minutes = float(minutes)
    except (TypeError, ValueError):
        minutes = 0
    if 60 <= minutes < 90:
        return '[60, 90) min'
    if 90 <= minutes < 120:
        return '[90, 120) min'
    if 120 <= minutes < 180:
        return '[120, 180) min'
    if 180 <= minutes < 270:
        return '[180, 270) min'
    return 'Other/Short'


def _certificate_bin(seconds):
    minutes = seconds / 60
    if minutes < 0.5:
        return '[0, 0.5) min'
    if minutes < 3:
        return '[0.5, 3) min'
    if minutes < 15:
        return '[3, 15) min'
    if minutes < 60:
        return '[15, 60) min'
    return '[60, inf) min'


def _extract_answer(prediction):
    text = str(prediction).strip()
    if not text or text.lower() == 'nan':
        return ''

    if '</think>' in text:
        final_response = text.rsplit('</think>', 1)[1].strip()
        if final_response:
            text = final_response

    fenced = re.findall(r'```(?:json)?\s*(.*?)```', text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced[-1].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            value = parsed.get('predicted_answer', parsed.get('answer', ''))
            match = re.search(r'\b([A-D])\b', str(value).upper())
            if match:
                return match.group(1)
    except (TypeError, json.JSONDecodeError):
        pass

    match = re.search(r'(?:^|\n)\s*[\[(]?([A-D])[\])]?[.。]?\s*$', text)
    if match:
        return match.group(1)

    explicit_patterns = [
        r'(?:final\s+)?(?:predicted_)?answer["\']?\s*(?:is|[:=])\s*["\'`*(\[]*([A-D])\b',
        r'(?:final\s+)?(?:option|choice)\s*(?:is|[:=])?\s*["\'`*(\[]*([A-D])\b',
    ]
    candidates = []
    for pattern in explicit_patterns:
        candidates.extend(
            (match.start(1), match.group(1).upper())
            for match in re.finditer(pattern, text, re.IGNORECASE)
        )

    # Keep standalone matching case-sensitive so ordinary words such as the
    # article "a" are not treated as option A. Prefer the last candidate to
    # avoid selecting an intermediate choice from the reasoning.
    candidates.extend(
        (match.start(1), match.group(1))
        for match in re.finditer(r'(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])', text)
    )
    if candidates:
        return max(candidates)[1]

    stripped = re.sub(r'[\s\W_]+', '', text.upper())
    return stripped[-1] if stripped and stripped[-1] in 'ABCD' else ''


class VideoOdysseyV(VideoBaseDataset):

    REPO_ID = 'maifoundations/VideoOdyssey'
    ANNOTATION_FILE = 'VideoOdyssey_V.json'
    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE_MODEL = 'exact_matching'
    JPEG_MAX_SIDE = 512
    JPEG_QUALITY = 70
    DOWNLOAD_WORKERS = 32

    def __init__(self, dataset='VideoOdyssey-V', nframe=64, fps=-1, pack=False):
        if fps > 0:
            raise ValueError('VideoOdyssey-V reproduces the official fixed-frame sampler; fps mode is unsupported.')
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, pack=pack)

    @classmethod
    def supported_datasets(cls):
        return ['VideoOdyssey-V']

    def prepare_dataset(self, dataset='VideoOdyssey-V'):
        if dataset not in self.supported_datasets():
            raise ValueError(f'Unsupported dataset: {dataset}')

        data_root = osp.join(LMUDataRoot(), dataset)
        os.makedirs(data_root, exist_ok=True)
        dataset_path = snapshot_download(
            repo_id=self.REPO_ID,
            repo_type='dataset',
            local_dir=data_root,
            allow_patterns=[self.ANNOTATION_FILE, 'videos/*.mp4'],
            max_workers=self.DOWNLOAD_WORKERS,
        )
        annotation_path = osp.join(dataset_path, self.ANNOTATION_FILE)
        with open(annotation_path, 'r', encoding='utf-8') as file:
            annotation = json.load(file)

        rows = []
        for video in annotation:
            relative_video_path = f"videos/{video['video_path']}"
            questions = video.get('questions', [])
            for question in questions:
                rows.append({
                    'index': len(rows),
                    'video': str(video['video_id']),
                    'video_path': relative_video_path,
                    'video_id': str(video['video_id']),
                    'question_id': str(question['question_id']),
                    'question': str(question['question']),
                    'options': json.dumps(question['options'], ensure_ascii=False),
                    'answer': str(question['answer']).strip().upper(),
                    'question_type': json.dumps(question.get('question_type', []), ensure_ascii=False),
                    'time_reference': str(question.get('time_reference', '')),
                    'video_category': str(video.get('video_category', 'Unknown')),
                    'duration_minutes': video.get('duration_minutes', 0),
                })

        if not rows:
            raise RuntimeError('VideoOdyssey-V preparation produced no questions.')
        data_file = osp.join(data_root, f'{dataset}.tsv')
        pd.DataFrame(rows).to_csv(data_file, sep='\t', index=False)
        return {'root': data_root, 'data_file': data_file}

    @staticmethod
    def _fourcc(video):
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
        return ''.join(chr((fourcc >> (8 * idx)) & 255) for idx in range(4))

    @staticmethod
    def _ffmpeg_frame(video_path, timestamp):
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                return None
        command = [
            ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-ss', f'{max(timestamp, 0):.3f}', '-i', video_path,
            '-frames:v', '1', '-q:v', '4', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1',
        ]
        result = subprocess.run(command, capture_output=True, check=False, timeout=60)
        if result.returncode != 0 or not result.stdout:
            return None
        return cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _save_frame(self, frame, path):
        height, width = frame.shape[:2]
        if max(height, width) > self.JPEG_MAX_SIDE:
            scale = self.JPEG_MAX_SIDE / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        if not cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY]):
            raise RuntimeError(f'Failed to save sampled frame: {path}')

    def save_video_frames(self, video):
        video_path = osp.join(self.data_root, 'videos', f'{video}.mp4')
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open VideoOdyssey video: {video_path}')
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total_frames <= 0 or fps <= 0:
            capture.release()
            raise RuntimeError(f'Invalid VideoOdyssey video metadata: {video_path}')

        interval = max(1, total_frames // self.nframe)
        indices = list(range(0, total_frames, interval))[:self.nframe]
        frame_paths = self.frame_paths(video)[:len(indices)]
        if all(osp.exists(path) for path in frame_paths):
            capture.release()
            return frame_paths

        use_ffmpeg = self._fourcc(capture).upper() == 'AV01'
        lock_path = osp.join(self.frame_root, f'{video}.lock')
        with portalocker.Lock(lock_path, 'w', timeout=600):
            if all(osp.exists(path) for path in frame_paths):
                capture.release()
                return frame_paths
            for frame_index, frame_path in zip(indices, frame_paths):
                if osp.exists(frame_path):
                    continue
                frame = None
                if not use_ffmpeg:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, frame = capture.read()
                    if not success:
                        frame = None
                if frame is None:
                    frame = self._ffmpeg_frame(video_path, frame_index / fps)
                if frame is None:
                    capture.release()
                    raise RuntimeError(f'Failed to extract frame {frame_index} from {video_path}')
                self._save_frame(frame, frame_path)
        capture.release()
        return frame_paths

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            if line >= len(self.data):
                raise IndexError(line)
            line = self.data.iloc[line]

        options = '\n'.join(str(option) for option in _list_value(line['options']))
        prompt = (
            'Based on the video only, please answer the question. '
            'You MUST choose the most appropriate answer from the given options (A, B, C, or D). '
            'Even if uncertain, make your best guess. Please directly output the option letter.\n\n'
            f"Question: {line['question']}\nOptions:\n{options}"
        )
        message = [{'type': 'text', 'value': prompt}]
        if video_llm:
            message.append({
                'type': 'video',
                'value': osp.join(self.data_root, 'videos', f"{line['video']}.mp4"),
            })
        else:
            message.extend(
                {'type': 'image', 'value': frame, 'detail': 'low'}
                for frame in self.save_video_frames(str(line['video']))
            )
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        del judge_kwargs
        data = load(eval_file)
        if 'prediction' not in data:
            raise ValueError(f'Prediction column is missing from {eval_file}')

        dimension_stats = {
            'Video Category': defaultdict(lambda: [0, 0]),
            'Question Type': defaultdict(lambda: [0, 0]),
            'Video Duration': defaultdict(lambda: [0, 0]),
            'Continuous Certificate Length': defaultdict(lambda: [0, 0]),
        }
        extracted_answers = []
        scores = []
        invalid = 0
        correct = 0

        for _, row in data.iterrows():
            prediction = _extract_answer(row['prediction'])
            extracted_answers.append(prediction)
            ground_truth = str(row['answer']).strip().upper()
            is_valid = prediction in {'A', 'B', 'C', 'D'}
            is_correct = is_valid and prediction == ground_truth
            scores.append(int(is_correct))
            invalid += int(not is_valid)
            correct += int(is_correct)

            labels = {
                'Video Category': [str(row.get('video_category', 'Unknown'))],
                'Question Type': _list_value(row.get('question_type')) or ['Unknown'],
                'Video Duration': [_duration_bin(row.get('duration_minutes', 0))],
                'Continuous Certificate Length': [
                    _certificate_bin(_certificate_seconds(row.get('time_reference', '')))
                ],
            }
            for dimension, dimension_labels in labels.items():
                for label in dimension_labels:
                    stats = dimension_stats[dimension][str(label)]
                    stats[1] += 1
                    stats[0] += int(is_correct)

        scored_data = data.copy()
        scored_data['extracted_answer'] = extracted_answers
        scored_data['score'] = scores
        score_file = get_intermediate_file_path(eval_file, '_score')
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(scored_data, score_file)

        metrics = {
            'Overall': (correct / len(data) * 100) if len(data) else 0.0,
            'Evaluated Questions': len(data),
            'Invalid Predictions': invalid,
        }
        for dimension, details in dimension_stats.items():
            for label, (label_correct, label_total) in sorted(details.items()):
                metrics[f'{dimension}|{label}'] = label_correct / label_total * 100
        dump(metrics, rating_file)
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'Overall' in metrics:
            return {'Overall Score': metrics['Overall']}
        return super().report_primary_metric(metrics)
