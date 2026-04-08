import os
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import portalocker
from PIL import Image

from vlmeval.smp import download_file, file_size, load, md5
from vlmeval.smp.file import (INFER_FAIL_MSG, RUN_STATUS_NAME, LMUDataRoot, _prediction_table,
                              fetch_aux_files)
from vlmeval.smp.log import get_logger
from vlmeval.smp.status_report import is_number, to_number
from .image_base import (_choose_primary_metric_key, _count_markers_in_obj,
                         _score_judge_file_candidate)

logger = get_logger(__name__)


class VideoBaseDataset(metaclass=ABCMeta):

    MODALITY = 'VIDEO'
    DEFAULT_JUDGE: str | list = 'gpt-4o-mini'

    INFER_FAIL_MARKERS = (INFER_FAIL_MSG, )
    JUDGE_FAIL_MARKERS = (INFER_FAIL_MSG, )

    def __init__(self,
                 dataset='MMBench-Video',
                 pack=False,
                 nframe=0,
                 fps=-1):
        try:
            import decord  # noqa: F401
        except Exception as e:
            logger.critical(f'{type(e)}: {e}')
            logger.critical('Please install decord via `pip install decord`.')

        self.dataset_name = dataset
        ret = self.prepare_dataset(dataset)
        assert ret is not None
        lmu_root = LMUDataRoot()
        self.frame_root = osp.join(lmu_root, 'images', dataset)
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = 'frame-{}-of-{}.jpg'
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)
        if 'index' not in self.data:
            self.data['index'] = np.arange(len(self.data))

        assert 'question' in self.data and 'video' in self.data
        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos
        self.pack = pack
        self.nframe = nframe
        self.fps = fps
        if self.fps > 0 and self.nframe > 0:
            raise ValueError('fps and nframe should not be set at the same time')
        if self.fps <= 0 and self.nframe <= 0:
            self.split_frame = False
            logger.info('fps and nframe is not set, disable frame split (Use video file directly.)')
        else:
            self.split_frame = True

    def __len__(self):
        return len(self.videos) if self.pack else len(self.data)

    def __getitem__(self, idx):
        if self.pack:
            assert idx < len(self.videos)
            sub_data = self.data[self.data['video'] == self.videos[idx]]
            return sub_data
        else:
            assert idx < len(self.data)
            return dict(self.data.iloc[idx])

    def frame_paths(self, video):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def frame_paths_fps(self, video, num_frames):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root,
                         self.frame_tmpl_fps.format(i, num_frames, self.fps)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, video):
        import decord
        if self.fps > 0:
            vid_path = osp.join(self.data_root, video + '.mp4')
            vid = decord.VideoReader(vid_path)

            # 计算视频的总帧数和总时长
            total_frames = len(vid)
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps

            # 计算需要提取的总帧数
            required_frames = int(total_duration * self.fps)

            # 计算提取帧的间隔
            step_size = video_fps / self.fps

            # 计算提取帧的索引
            indices = [int(i * step_size) for i in range(required_frames)]

            # 提取帧并保存
            frame_paths = self.frame_paths_fps(video, len(indices))
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths

            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths

        else:
            frame_paths = self.frame_paths(video)
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths
            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths
                vid_path = osp.join(self.data_root, video + '.mp4')
                vid = decord.VideoReader(vid_path)
                step_size = len(vid) / (self.nframe + 1)
                indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return ['MMBench-Video', 'Video-MME', 'MVBench', 'MVBench_MP4',
                'LongVideoBench', 'WorldSense', 'VDC', 'MovieChat1k', 'AV-SpeakerBench']

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @abstractmethod
    def build_prompt(self, idx):
        pass

    @abstractmethod
    def prepare_dataset(self, dataset):
        # The prepare_dataset function should return a dictionary containing:
        # `root` (directory that containing video files)
        # `data_file` (the TSV dataset file)
        pass

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name_legacy = url.split('/')[-1]
        file_name = f"{self.dataset_name}.tsv"
        data_path_legacy = os.path.join(data_root, file_name_legacy)
        data_path = os.path.join(data_root, file_name)

        self.data_path = data_path
        if os.path.exists(data_path):
            if file_md5 is None or md5(data_path) == file_md5:
                pass
            else:
                warnings.warn(f'The tsv file is in {data_root}, but the md5 does not match, will re-download')
                download_file(url, data_path)
                update_flag = True
        else:
            if os.path.exists(data_path_legacy) and (file_md5 is None or md5(data_path_legacy) == file_md5):
                warnings.warn(
                    'Due to a modification in #1055, the local target file name has changed. '
                    f'We detected the tsv file with legacy name {data_path_legacy} exists and will do the rename. '
                )
                import shutil
                shutil.move(data_path_legacy, data_path)
            else:
                download_file(url, data_path)
                update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not os.path.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    @classmethod
    def get_judge_file(cls, eval_file: str | Path, judge_model: str | None = None) -> Path | None:
        eval_path = Path(eval_file)
        aux_files = fetch_aux_files(str(eval_path))
        if not aux_files:
            return None

        prediction_path = eval_path.resolve() if eval_path.exists() else None
        candidates = []
        for aux_file in aux_files:
            aux_path = Path(aux_file)
            if not aux_path.exists():
                continue
            if prediction_path is not None and aux_path.resolve() == prediction_path:
                continue
            if aux_path.name == RUN_STATUS_NAME or aux_path.suffix == '.lock':
                continue
            if aux_path.name.endswith(('_checkpoint.pkl', '_PREV.pkl', '_structs.pkl')):
                continue
            candidates.append(aux_path)

        if not candidates:
            return None

        candidates.sort(key=lambda path: _score_judge_file_candidate(path, judge_model), reverse=True)
        return candidates[0]

    @classmethod
    def report_infer_err(cls, prediction_file: str | Path | None) -> dict[str, int]:
        if prediction_file is None or not Path(prediction_file).exists():
            return dict(failed=0, total=0)

        frame = _prediction_table(str(prediction_file))
        if frame is None:
            return dict(failed=0, total=0)

        predictions = frame['prediction'] if 'prediction' in frame else []
        total = len(predictions)
        failed = sum(
            any(marker in str(prediction) for marker in cls.INFER_FAIL_MARKERS)
            for prediction in predictions
        )
        return dict(failed=failed, total=total)

    @classmethod
    def report_judge_err(
        cls,
        prediction_file: str | Path | None,
        *,
        total_samples: int | None,
        judge_model: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, int | None]:
        if total_samples is None or total_samples <= 0:
            return dict(failed=0, total=total_samples)

        if prediction_file is None or not Path(prediction_file).exists():
            failed = total_samples if error_message else 0
            return dict(failed=failed, total=total_samples)

        judge_file = cls.get_judge_file(prediction_file, judge_model=judge_model)
        if judge_file is None:
            failed = total_samples if error_message else 0
            return dict(failed=failed, total=total_samples)

        try:
            data = load(str(judge_file))
        except Exception:
            data = None

        failed = min(_count_markers_in_obj(data, cls.JUDGE_FAIL_MARKERS), total_samples)
        if error_message and failed == 0:
            failed = total_samples
        return dict(failed=failed, total=total_samples)

    @classmethod
    def report_primary_metric(cls, metrics: dict[str, Any] | None) -> dict[str, float | int]:
        if not isinstance(metrics, dict) or not metrics:
            return {}

        matched_keys = []
        fallback = _choose_primary_metric_key(metrics)
        if fallback is not None:
            matched_keys = [fallback]

        primary_metrics = {}
        for key in matched_keys:
            value = metrics.get(key)
            if is_number(value):
                primary_metrics[str(key)] = to_number(value)
        return primary_metrics
