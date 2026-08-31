import json
import os.path as osp
import re
import tarfile
import warnings
from glob import glob

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import dump, get_cache_path, get_file_extension, get_intermediate_file_path, load
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


class HERBench(VideoBaseDataset):
    """HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering.

    Paper: https://arxiv.org/abs/2512.14870
    Data: https://huggingface.co/datasets/DanBenAmi/HERBench

    Five-way multiple-choice questions over long videos. Each question requires
    aggregating at least 3 distinct, temporally separated visual cues.
    Three subsets are supported:
      - HERBench         (config 'full',    27k+ questions, 335 videos, ~161 GB of videos)
      - HERBench_lite    (config 'lite',    68 videos, ~35 GB of videos)
      - HERBench_lite_v2 (config 'lite_v2', refined lite subset, same 68 videos)
    """

    TYPE = 'Video-MCQ'
    REPO_ID = 'DanBenAmi/HERBench'
    DEFAULT_JUDGE_MODEL = 'exact_matching'
    CHOICES = ('A', 'B', 'C', 'D', 'E')

    # dataset_name -> HuggingFace config / parquet file
    VARIANTS = {
        'HERBench': dict(hf_config='full', parquet='data/herbench_full.parquet'),
        'HERBench_lite': dict(hf_config='lite', parquet='data/herbench_lite.parquet'),
        'HERBench_lite_v2': dict(hf_config='lite_v2', parquet='data/herbench_lite_v2.parquet'),
    }

    # The video archive is split into chunks: videos.tar.part.00 .. videos.tar.part.16.
    # Parts 00-03 form one complete tar with the 68 HERBench-Lite videos (used by both
    # 'lite' and 'lite_v2'), the remaining parts form a second complete tar with the
    # rest of the full set. Extraction must therefore tolerate concatenated archives
    # (handled below via tarfile's ignore_zeros). The lite datasets download only the
    # four lite chunks; the full dataset downloads by pattern, so extra chunks appended
    # to the HF repo later are picked up automatically.
    LITE_PART_IDS = tuple(range(0, 4))

    def __init__(self, dataset='HERBench', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return list(cls.VARIANTS)

    @classmethod
    def _strip_choice_prefix(cls, choice):
        # Choices in the parquet already carry letter prefixes such as 'A. <text>'
        match = re.match(r'^([A-E])\.\s*(.*)$', str(choice), flags=re.DOTALL)
        return match.group(2) if match else str(choice)

    @staticmethod
    def _parse_candidates(value):
        """Choice texts for one line, from the JSON-encoded 'candidates' column.

        JSON (rather than per-letter TSV columns) keeps choices whose literal text
        is 'None'/'NA'/... intact through the TSV/xlsx round-trip, which pandas
        would otherwise parse as NaN (16 such choices exist in the full set).
        """
        return value if isinstance(value, list) else json.loads(value)

    @classmethod
    def _generate_tsv(cls, dataset_path, dataset_name):
        parquet_file = osp.join(dataset_path, cls.VARIANTS[dataset_name]['parquet'])
        data = pd.read_parquet(parquet_file)
        data = data.assign(index=range(len(data)))
        data['video'] = data['video_id']
        data['candidates'] = data['choices'].map(
            lambda choices: json.dumps([cls._strip_choice_prefix(c) for c in choices]))

        columns = [
            'index', 'question_id', 'video', 'video_path', 'question', 'candidates',
            'answer', 'answer_text', 'task_type', 'source_dataset', 'duration', 'resolution'
        ]
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        data[columns].to_csv(data_file, sep='\t', index=False)
        return data_file

    @classmethod
    def _check_integrity(cls, dataset_path, dataset_name):
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        if not osp.isfile(data_file):
            return False
        data = load(data_file)
        required = {'question', 'video', 'video_path', 'task_type', 'answer', 'candidates'}
        if not required.issubset(data.columns):
            return False
        return all(osp.isfile(osp.join(dataset_path, pth)) for pth in data['video_path'].unique())

    @classmethod
    def _extract_videos(cls, dataset_path, dataset_name):
        parquet_file = osp.join(dataset_path, cls.VARIANTS[dataset_name]['parquet'])
        needed = set(pd.read_parquet(parquet_file, columns=['video_path'])['video_path'])
        if all(osp.exists(osp.join(dataset_path, pth)) for pth in needed):
            return

        if dataset_name == 'HERBench':
            # full set: take every downloaded chunk (zero-padded names sort correctly)
            part_files = sorted(glob(osp.join(dataset_path, 'videos', 'videos.tar.part.*')))
            if len(part_files) < len(cls.LITE_PART_IDS):
                raise FileNotFoundError(
                    f'Only {len(part_files)} HERBench video archive chunks found under '
                    f'{osp.join(dataset_path, "videos")}. Re-run to resume the download.')
        else:
            part_files = [
                osp.join(dataset_path, 'videos', f'videos.tar.part.{i:02d}')
                for i in cls.LITE_PART_IDS
            ]
            missing_parts = [p for p in part_files if not osp.exists(p)]
            if missing_parts:
                raise FileNotFoundError(
                    f'Missing HERBench video archive chunks: {missing_parts}. '
                    'Re-run to resume the download, or fetch them manually with huggingface-cli.')

        class _ChainedReader:
            """Read the split tar chunks as one continuous stream (no 170 GB temp file)."""

            def __init__(self, paths):
                self.paths = list(paths)
                self.idx = 0
                self.fp = open(self.paths[0], 'rb') if self.paths else None

            def read(self, size=-1):
                if size is None or size < 0:
                    raise ValueError('unbounded read not supported')
                chunks = []
                remaining = size
                while remaining > 0 and self.fp is not None:
                    chunk = self.fp.read(remaining)
                    if chunk:
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    else:
                        self.fp.close()
                        self.idx += 1
                        self.fp = open(self.paths[self.idx], 'rb') if self.idx < len(self.paths) else None
                return b''.join(chunks)

            def close(self):
                if self.fp is not None:
                    self.fp.close()
                    self.fp = None

        # Tar entries are relative to the videos/ directory (e.g. 'trailers/xxx.mp4'),
        # matching the 'videos/...' paths in the annotations once extracted there.
        videos_root = osp.join(dataset_path, 'videos')
        print(f'Extracting HERBench videos to {videos_root} (this may take a while)...')
        reader = _ChainedReader(part_files)
        extracted = skipped = 0
        try:
            # ignore_zeros lets tarfile read through the end-of-archive markers between
            # the two concatenated tar archives that make up the chunk sequence
            with tarfile.open(fileobj=reader, mode='r|', ignore_zeros=True) as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = member.name.lstrip('./')
                    if name.startswith('/') or '..' in name.split('/'):
                        warnings.warn(f'Skipping suspicious tar member: {member.name}')
                        continue
                    target = osp.join(videos_root, name)
                    if osp.exists(target) and osp.getsize(target) == member.size:
                        skipped += 1
                        continue
                    member.name = name
                    tar.extract(member, path=videos_root)
                    extracted += 1
        finally:
            reader.close()
        print(f'HERBench video extraction done: {extracted} extracted, {skipped} already present.')

        still_missing = [pth for pth in needed if not osp.exists(osp.join(dataset_path, pth))]
        if still_missing:
            raise RuntimeError(
                f'{len(still_missing)} HERBench videos are still missing after extraction, '
                f'e.g. {still_missing[:3]}. If the downloaded archive chunks are corrupted, delete '
                'them and re-run to re-download; otherwise please report this at '
                'https://github.com/DanBenAmi/HERBench/issues.')

    def prepare_dataset(self, dataset_name='HERBench', repo_id=REPO_ID):
        assert dataset_name in self.VARIANTS, f'Unknown HERBench variant: {dataset_name}'
        dataset_path = get_cache_path(repo_id)
        if dataset_path is None or not self._check_integrity(dataset_path, dataset_name):
            if dataset_name == 'HERBench':
                part_patterns = ['videos/videos.tar.part.*']
            else:
                part_patterns = [f'videos/videos.tar.part.{i:02d}' for i in self.LITE_PART_IDS]
            # 'videos/*.mp4' also fetches any loose video files shipped next to the
            # archives (e.g. hotfixes for videos missing from the tar chunks)
            dataset_path = snapshot_download(
                repo_id=repo_id, repo_type='dataset',
                allow_patterns=['data/*.parquet', 'videos/*.mp4'] + part_patterns)
            self._extract_videos(dataset_path, dataset_name)
            self._generate_tsv(dataset_path, dataset_name)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def save_video_frames(self, video_path, video_llm=False):
        import decord
        vid_path = osp.join(self.data_root, video_path)
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        # frame cache key: strip the leading 'videos/' dir and the file extension
        frame_key = osp.splitext(video_path)[0]
        if frame_key.startswith('videos/'):
            frame_key = frame_key[len('videos/'):]

        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(frame_key)
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(frame_key, len(indices))
        else:
            raise ValueError(
                'HERBench frame sampling needs nframe or fps to be set. Use a dataset variant '
                'such as HERBench_lite_v2_16frame / HERBench_1fps, or pass nframe/fps explicitly.')

        flag = np.all([osp.exists(p) for p in frame_paths])
        if not flag and not video_llm:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=300):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        candidates = self._parse_candidates(line['candidates'])
        letters = [self.CHOICES[i] for i in range(len(candidates))]
        options = '\n'.join(f'{letter}. {text}' for letter, text in zip(letters, candidates))
        letter_str = ', '.join(letters[:-1]) + f', or {letters[-1]}'
        # Prompt format follows the official HERBench evaluation protocol
        prompt = (
            f"{line['question']}\n\n{options}\n\n"
            f'Please respond with only the correct answer letter ({letter_str}) '
            'without any explanations or additional text.'
        )

        message = []
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video_path'])))
        else:
            frame_paths, _, _ = self.save_video_frames(line['video_path'])
            for frame_path in frame_paths:
                message.append(dict(type='image', value=frame_path))
        message.append(dict(type='text', value=prompt))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from .utils.herbench import extract_characters_regex, extract_option, get_dimension_rating

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be a supported format (xlsx/json/tsv) file'

        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', cls.DEFAULT_JUDGE_MODEL)
            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

            data = load(eval_file)
            failed = pd.isna(data['prediction']) | data['prediction'].astype(str).str.contains(FAIL_MSG)
            n_missing = int(failed.sum())

            letters_per_row = data['candidates'].map(
                lambda c: ''.join(cls.CHOICES[:len(cls._parse_candidates(c))]))
            extracted = pd.Series(
                [extract_characters_regex(p, letters) for p, letters in zip(data['prediction'], letters_per_row)],
                index=data.index)
            # fall back to the (optional) LLM judge for unparseable predictions;
            # failed / missing predictions always score 0 and never reach the judge
            for pos in np.where((extracted.values == '') & ~failed.values)[0]:
                item = data.iloc[pos].to_dict()
                extracted.iloc[pos] = extract_option(model, item, 'HERBench')
            extracted[failed.values] = ''
            data['extracted_answer'] = extracted
            answers = data['answer'].astype(str).str.strip().str.upper()
            data['score'] = (extracted == answers).astype(int)

            print(
                f'Among {len(data)} questions, '
                f'failed to obtain prediction for {n_missing} questions. '
                'Those questions are counted as wrong in the reported accuracy.'
            )
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and metrics.get('overall') not in (None, ''):
            return {'Overall Acc': float(metrics['overall'])}
        return super().report_primary_metric(metrics)
