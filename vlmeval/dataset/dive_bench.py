"""DIVE-Bench: educational text QA and sampled high-motion grid trajectories."""

import hashlib
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import portalocker

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .utils.dive_bench import (
    _compute_cer, _compute_exact_match, _compute_token_f1, _compute_wer,
    _grid_sequence_metrics, _highmotion_count_text, parse_grid_sequence,
)
from .video_base import VideoBaseDataset


EDUCATIONAL = 'dive_bench_educational_high_fps'
HIGH_MOTION = 'dive_bench_high_motion_high_fps'
PREVIEW = 'dive_bench_high_motion_high_fps_preview1000'
EDUCATIONAL_POST_PROMPT = (
    ' Provide the most complete answer possible. For subtitle or OCR questions, '
    'reproduce the relevant text from the video instead of answering with only the video id.'
)


def sample_indices(frame_count, nframe):
    if frame_count <= 0 or nframe <= 0:
        raise ValueError('frame_count and nframe must be positive')
    return np.linspace(0, frame_count - 1, min(nframe, frame_count), dtype=int).tolist()


def high_motion_prompt(doc, nframe):
    count = min(int(doc['frame_count']), nframe)
    count_text = _highmotion_count_text(count)
    comma_text = _highmotion_count_text(max(count - 1, 0))
    question = str(doc.get('question', '')).strip()
    action = re.split(r'\n\nWe consider all\s+\d+\s+frames', question, maxsplit=1)[0].strip()
    if not action:
        action = 'Track the visible right-hand palm center in the video.'
    return (
        f'{action}\n\n'
        f'The video input contains exactly {count_text} frames uniformly sampled in temporal order '
        'from the original clip, including its first and last frames. '
        'On each sampled frame, divide the image into top, middle, and bottom rows and left, center, '
        'and right columns, then locate the visible right-hand palm center. '
        'Use topleft, top, or topright for the top row; left, middle, or right for the middle row; '
        'and bottomleft, bottom, or bottomright for the bottom row. '
        'For every sampled frame in order, select one region; repeat a name when the hand stays '
        'in the same region.\n\n'
        f'Answer with exactly {count_text} labels in sampled-frame order (exactly {comma_text} commas) '
        "and no extra text. Return one item per input frame—not one item per possible region—and stop "
        "after the final frame's label. "
        f'Do not stop early: if uncertain, repeat your best region choice until all {count_text} '
        'frame slots are filled. Before answering, verify the comma count. Use commas only; '
        'do not use vertical bars or brackets.'
    )


class DIVEBench(VideoBaseDataset):
    TYPE = 'VQA'
    # The old high-motion alias is explicitly the published 1,000-row preview.
    ALIASES = {'densevideo': EDUCATIONAL, 'densevideo_highmotion': PREVIEW}
    SPECS = {
        EDUCATIONAL: {
            'repo': 'haichaozhang/DenseVideoEvaluation',
            'filename': 'LPM_videos.parquet',
            'revision': '5cc61a045c8e5e95d1d9c87e22ccd0f699575aea',
            'sha256': '9ab09ea35a66ca86fc7fdce4e539171ea5eef38f8162874f74dd1b2809dafb66',
            'rows': 634,
        },
        HIGH_MOTION: {
            'repo': 'haichaozhang/highmotion_densevideounderstand',
            'filename': 'Egodex_traj.parquet',
            'revision': None,  # Public access/revision not established; exact bytes are checked below.
            'sha256': '39f9da7aca9020d79f383953646a5893f09c6f8e5f60433560011280ee987b2d',
            'rows': 3243,
        },
    }

    def __init__(self, dataset=EDUCATIONAL, nframe=8, fps=-1, pack=False,
                 annotation_file=None, data_root=None):
        if dataset not in self.supported_datasets():
            raise ValueError(f'Unknown DIVE-Bench task: {dataset}')
        if pack or fps > 0 or not isinstance(nframe, int) or isinstance(nframe, bool) or nframe <= 0:
            raise ValueError('DIVE-Bench requires pack=False, fps<=0 and a positive integer nframe')
        self.canonical_name = self.ALIASES.get(dataset, dataset)
        self.is_high_motion = self.canonical_name != EDUCATIONAL
        self.annotation_file = annotation_file
        self.requested_root = data_root or os.environ.get('DIVE_BENCH_DATA_ROOT')
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, pack=False)

    @classmethod
    def supported_datasets(cls):
        return [EDUCATIONAL, HIGH_MOTION, PREVIEW, *cls.ALIASES]

    def prepare_dataset(self, dataset):
        spec = self.SPECS[HIGH_MOTION if self.is_high_motion else EDUCATIONAL]
        if not self.requested_root:
            raise ValueError('Set DIVE_BENCH_DATA_ROOT or data_root to the extracted video root. '
                             'Video archives are not downloaded/extracted automatically.')
        root = Path(self.requested_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'DIVE-Bench video root does not exist: {root}')
        annotation = self.annotation_file
        if annotation is None:
            from huggingface_hub import hf_hub_download
            try:
                annotation = hf_hub_download(
                    repo_id=spec['repo'], repo_type='dataset', filename=spec['filename'],
                    revision=spec['revision'],
                )
            except Exception as exc:
                raise RuntimeError(
                    f'Cannot retrieve {spec["repo"]}/{spec["filename"]}. Educational data requires '
                    'accepted Hub access terms and authentication; high-motion public access is '
                    'not yet established. Supply an authorized local annotation_file instead.'
                ) from exc
        annotation = Path(annotation)
        checksum = hashlib.sha256(annotation.read_bytes()).hexdigest()
        if checksum != spec['sha256']:
            raise ValueError('DIVE-Bench annotation SHA-256 mismatch; refusing a different release/order')
        raw = pd.read_parquet(annotation)
        required = ['video_path', 'question', 'answer', 'qid', 'frame_count']
        if any(column not in raw for column in required) or len(raw) != spec['rows']:
            raise ValueError(f'Invalid annotation schema/count; expected {required}, {spec["rows"]} rows')
        if raw[required].isna().any().any():
            raise ValueError('DIVE-Bench annotations contain missing required values')
        if self.canonical_name == PREVIEW:
            raw = raw.iloc[:1000]
        data = raw[required].copy().reset_index(drop=True)
        data.insert(0, 'index', np.arange(len(data)))
        if data[['video_path', 'qid']].duplicated().any():
            raise ValueError('Duplicate (video_path, qid) example identity')
        if any(not isinstance(value, str) or not value.strip() for value in data['answer']):
            raise ValueError('DIVE-Bench reference answers must be nonempty strings')
        for row in data.to_dict('records'):
            count = row['frame_count']
            if count != int(count) or count <= 0:
                raise ValueError('Invalid frame_count')
            if self.is_high_motion and len(parse_grid_sequence(row['answer'])) != int(count):
                raise ValueError('High-motion reference length must match frame_count')
        data['video'] = [self._resolve_video(root, path) for path in data['video_path']]
        cache_root = Path(LMUDataRoot()) / 'DIVE-Bench'
        cache_root.mkdir(parents=True, exist_ok=True)
        root_hash = hashlib.sha256(str(root).encode()).hexdigest()[:12]
        data_file = cache_root / f'{dataset}-{checksum[:12]}-{root_hash}.tsv'
        # Regenerate from the checksummed annotations, never trust a stale cached answer table.
        with portalocker.Lock(str(data_file) + '.lock', 'a', timeout=60):
            # The parent class reads after this lock is released. Publish atomically
            # so another rank can never expose a partially written answer table.
            with tempfile.NamedTemporaryFile(dir=cache_root, suffix='.tsv', delete=False) as stream:
                temporary = Path(stream.name)
            try:
                dump(data, str(temporary))
                os.replace(temporary, data_file)
            finally:
                temporary.unlink(missing_ok=True)
        return {'root': str(root), 'data_file': str(data_file)}

    def _resolve_video(self, root, value):
        relative = Path(str(value))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe annotation video path: {value}')
        candidates = [root / relative]
        if not self.is_high_motion:
            candidates.extend([root / 'videos' / relative.name,
                               root / 'DenseVideoEvaluation' / 'videos' / relative.name,
                               root / 'DenseVideo-LPM' / 'videos' / relative.name])
        else:
            candidates.extend([root / 'highmotion_densevideounderstand' / relative,
                               root / 'highmotion_densevideounderstand' / 'all_test' / relative])
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file():
                return str(resolved)
        # Never resolve high-motion videos by basename: action folders reuse numeric names.
        raise FileNotFoundError(f'DIVE-Bench video missing under {root}: {value}')

    def _row(self, index):
        if isinstance(index, (int, np.integer)):
            return self.data.iloc[int(index)]
        return index

    def _sample_frames(self, row):
        import decord
        from PIL import Image
        video = row['video']
        reader = decord.VideoReader(video)
        if self.is_high_motion and len(reader) != int(row['frame_count']):
            raise ValueError('Decoded high-motion frame count differs from the annotations')
        indices = sample_indices(len(reader), self.nframe)
        key = hashlib.sha256(str(video).encode()).hexdigest()
        frame_root = Path(self.frame_root) / key
        frame_root.mkdir(parents=True, exist_ok=True)
        paths = [frame_root / f'endpoint-{index}.png' for index in indices]
        with portalocker.Lock(str(frame_root / 'frames.lock'), 'a', timeout=60):
            frames = reader.get_batch(indices).asnumpy()
            for path, frame in zip(paths, frames):
                Image.fromarray(frame).save(path)
        return [str(path) for path in paths]

    def build_prompt(self, idx, video_llm=False):
        row = self._row(idx)
        question = (high_motion_prompt(row, self.nframe) if self.is_high_motion
                    else row['question'] + EDUCATIONAL_POST_PROMPT)
        if self.is_high_motion and video_llm:
            raise ValueError('Generic VIDEO_LLM wrappers may resample or require video messages. '
                             'Use the GRT custom adapter or an ordered-image model with VIDEO_LLM=False '
                             'for the aligned High-Motion protocol.')
        if video_llm and not self.is_high_motion:
            visuals = [dict(type='video', value=row['video'])]
        else:
            # High-motion labels require these exact frames, not arbitrary model-side sampling.
            visuals = [dict(type='image', value=path) for path in self._sample_frames(row)]
        return [*visuals, dict(type='text', value=question)]

    def build_grt_prompt(self, row):
        if self.nframe != 8:
            raise ValueError('The released GRT profiles require exactly eight sampled frames')
        row = self._row(row)
        question = (high_motion_prompt(row, self.nframe) if self.is_high_motion
                    else row['question'] + EDUCATIONAL_POST_PROMPT)
        return [dict(type='video', value=row['video']), dict(type='text', value=question)]

    def evaluate(self, eval_file, **judge_kwargs):
        predictions = load(eval_file)
        if 'index' not in predictions or 'prediction' not in predictions:
            raise ValueError('Prediction table must include index and prediction')
        expected = self.data.copy()
        expected['_key'] = expected['index'].astype(str)
        predictions = predictions[['index', 'prediction']].copy()
        predictions['_key'] = predictions['index'].astype(str)
        if predictions['_key'].duplicated().any() or set(predictions['_key']) != set(expected['_key']):
            raise ValueError('Prediction ids must match the complete selected split exactly once')
        merged = expected.merge(predictions[['_key', 'prediction']], on='_key', validate='one_to_one')
        scores = []
        failed = 0
        for row in merged.to_dict('records'):
            prediction = row['prediction']
            if pd.isna(prediction) or any(marker in str(prediction) for marker in self.INFER_FAIL_MARKERS):
                prediction = ''
                failed += 1
            if self.is_high_motion:
                full = parse_grid_sequence(row['answer'])
                target = [full[index] for index in sample_indices(len(full), self.nframe)]
                parsed = parse_grid_sequence(prediction)
                score = _grid_sequence_metrics(parsed, target)
                score['token_f1'] = _compute_token_f1(' '.join(parsed), ' '.join(target))
            else:
                score = {
                    'cer': _compute_cer(prediction, row['answer']),
                    'wer': _compute_wer(prediction, row['answer']),
                    'token_f1': _compute_token_f1(prediction, row['answer']),
                    'exact_match': _compute_exact_match(prediction, row['answer']),
                }
            scores.append(score)
        result = {key: math.fsum(score[key] for score in scores) / len(scores) for key in scores[0]}
        result.update(samples=len(scores), failed_predictions=failed, nframe=self.nframe)
        dump(result, get_intermediate_file_path(eval_file, '_score', 'json'))
        return result
