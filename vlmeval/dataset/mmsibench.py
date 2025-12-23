import os
import re
import ast
import string
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import OrderedDict
from huggingface_hub import snapshot_download

from .image_mcq import ImageMCQDataset
from .video_base import VideoBaseDataset
from ..smp.misc import toliststr, get_cache_path, modelscope_flag_set
from ..smp.file import LMUDataRoot, dump, load


class MMSIBench(ImageMCQDataset):
    """
    MMSI-Bench.

    Reference:
      MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence
      https://arxiv.org/abs/2505.23764
    """

    TYPE = 'MCQ'

    # VLMEvalKit officially provides an MMSI *circular* TSV.
    # In this repo we only run the *non-circular* evaluation, which aligns with the
    # evaluation protocol described in the MMSI paper.
    # To avoid modifying upstream VLMEvalKit, we do NOT integrate the circular set here.
    # (Use the official pipeline if you need the circular split.)
    DATASET_URL = {
        'MMSIBench_wo_circular': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MMSIBench_wo_circular.tsv'  # noqa: E501
    }

    DATASET_MD5 = {
        'MMSIBench_wo_circular': '548c5f33f1a12948d5355d5f600749e4'
    }

    def _task_category(self):
        return [
            'Pos-Cam-Cam',
            'Pos-Obj-Obj',
            'Pos-Reg-Reg',
            'Pos-Cam-Obj',
            'Pos-Obj-Reg',
            'Pos-Cam-Reg',
            'Attr-Meas',
            'Attr-Appr',
            'Motion-Cam',
            'Motion-Obj',
            'MSR'
        ]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        # Prompt format aligned with MMSI code base
        options_prompt = 'Options: '
        for key, item in options.items():
            options_prompt += f'{key}: {item}, '
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None

        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt

        # MMSI Direct
        post_prompt = (
            "Answer with the option's letter from the given choices directly. "
            "Enclose the option's letter within ``."
        )

        prompt = f'{prompt}\n{post_prompt}'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import eval_mcq_score, build_mcq_score_fn

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='category',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'MMSIBench')
        )


class MMSIVideoBench(VideoBaseDataset):
    """
    MMSI-Video-Bench.

    Reference:
      MMSI-Video-Bench: A Holistic Benchmark for Video-Based Spatial Intelligence
      https://arxiv.org/abs/2512.10863
    """

    TYPE = 'MCQ'

    DATASET_URL = {
        'MMSIVideoBench': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MMSIVideoBench.tsv'  # noqa: E501
    }

    DATASET_MD5 = {
        'MMSIVideoBench': '814e84913f3faab4ef63c3dfb82a73a3'
    }

    _CATEGORY_TASK_ORDER = None
    LMUData_root = LMUDataRoot()

    def __init__(self, dataset, pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)
        self.repo_id = 'rbler/MMSI-Video-Bench'

    @classmethod
    def supported_datasets(cls):
        subsets = ['MMSIVideoBench']
        return subsets

    @classmethod
    def category_task_order(cls) -> OrderedDict:
        if cls._CATEGORY_TASK_ORDER is None:
            cls._CATEGORY_TASK_ORDER = OrderedDict(
                [
                    (
                        'Spatial Construction',
                        [
                            'Instance/Scene Attribute',
                            'Instance-Instance Spatial Relationship',
                            'Instance-Scene Spatial Relationship',
                            'Scene-Scene Spatial Relationship',
                            'Camera-Instance Spatial Relationship',
                            'Camera-Scene Spatial Relationship',
                        ],
                    ),
                    (
                        'Motion Understanding',
                        [
                            'Camera Motion',
                            'Instance Motion',
                            'Interactive Motion',
                        ],
                    ),
                    (
                        'Cross-Video',
                        [
                            'Memoery Update',
                            'Multi-View Integration',
                        ],
                    ),
                    ('Planning', ['Planning']),
                    ('Prediction', ['Prediction']),
                ]
            )
        return cls._CATEGORY_TASK_ORDER

    def download_mmsivideobench(self, repo_id='rbler/MMSI-Video-Bench'):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = '.mmsivideo_extracted'

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            def unzip_hf_zip(pth):
                import zipfile

                base_dir = pth
                zip_files = [
                    os.path.join(base_dir, f) for f in os.listdir(base_dir)
                    if f.endswith('.zip')
                ]
                zip_files.sort()

                for zip_file in tqdm(zip_files, desc='Unpacking Origin Data...'):
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue

                            rel = os.path.normpath(info.filename).lstrip('/\\')
                            dst = os.path.join(pth, rel)

                            absp = os.path.abspath(pth)
                            absd = os.path.abspath(dst)
                            if not absd.startswith(absp + os.sep):
                                raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                                out.write(src.read())

                sentinel_path = os.path.join(pth, SENTINEL_NAME)
                _write_sentinel(sentinel_path, text='done')
                print('MMSI-Video-Bench data extracted to current directory with original layout.')

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        return dataset_path

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        data = super().prepare_tsv(url, md5)

        dataset_path = self.download_mmsivideobench()
        self.dataset_path = dataset_path

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x):
                if isinstance(x, list):
                    return [fix_one(item) for item in x]

                if isinstance(x, str):
                    s = x.strip()
                    s = os.path.expanduser(os.path.expandvars(s))

                    if not dataset_path:
                        return os.path.normpath(s)
                    return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\/')))

                return x

            def to_abs(p):
                if isinstance(p, list):
                    return fix_one(p)

                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return fix_one(lst)
                    except Exception:
                        pass

                return fix_one(p)

            data['image_path'] = data['image_path'].map(to_abs)
            data['ref_images'] = data['ref_images'].map(to_abs)

        new_data_path = os.path.join(self.LMUData_root, 'MMSIVideoBench_abs_path.tsv')
        if not os.path.exists(new_data_path):
            dump(data, new_data_path)

        return dict(data_file=new_data_path, root=dataset_path)

    def save_video_frames(self, frames_list, video_fps=None):
        """
        Args:
            frames_list (List[List[str]]): Per-segment frame paths
            video_fps (float, optional): Original video fps (used only when self.fps > 0)

        Returns:
            frame_paths_per_seg (List[List[str]]): Sampled frame paths for each segment
        """
        seg_lengths = [len(seg) for seg in frames_list]
        total_frames = sum(seg_lengths)

        if total_frames == 0:
            empty = [[] for _ in frames_list]
            return empty

        # Sample in the global index space [0, total_frames - 1]
        indices_global = []

        if self.nframe > 0 and self.fps < 0:
            n = min(self.nframe, total_frames)
            if n == total_frames:
                indices_global = list(range(total_frames))
            else:
                indices_global = np.linspace(0, total_frames - 1, n, dtype=int).tolist()

        elif self.fps > 0 and video_fps is not None and video_fps > 0:
            # Sample based on fps
            total_duration = total_frames / video_fps
            required_frames = max(int(total_duration * self.fps), 1)
            step_size = video_fps / self.fps

            indices_global = [
                min(int(i * step_size), total_frames - 1)
                for i in range(required_frames)
            ]
        else:
            raise ValueError(
                f"Video Dataset requires fps > 0 or nframe > 0, "
                f"but got fps={self.fps}, nframe={self.nframe}"
            )

        # Deduplicate while preserving order
        seen = set()
        uniq_global = []
        for idx in indices_global:
            if idx not in seen:
                seen.add(idx)
                uniq_global.append(idx)
        indices_global = uniq_global

        # Map global indices back to each segment
        frame_paths_per_seg = [[] for _ in frames_list]

        # Compute starting global index offset for each segment
        offsets = []
        acc = 0
        for L in seg_lengths:
            offsets.append(acc)
            acc += L

        for g_idx in indices_global:
            # Find which segment this global index belongs to
            for seg_id, start in enumerate(offsets):
                end = start + seg_lengths[seg_id]
                if g_idx < end:
                    local_idx = g_idx - start
                    frame_paths_per_seg[seg_id].append(frames_list[seg_id][local_idx])
                    break

        return frame_paths_per_seg

    def parse_image_path(self, val):
        """
        For MMSI-Video `image_path`:
        - val can be:
            * a Python list: [[...], [...]]
            * a string of a list: '[["a.jpg"], ["b.jpg"]]'
        - Always return: List[List[str]]
        """
        if isinstance(val, str):
            try:
                obj = ast.literal_eval(val.strip())
            except Exception as e:
                raise ValueError(f"Failed to parse image_path from string: {val[:200]}") from e
        else:
            obj = val

        if not isinstance(obj, list):
            raise TypeError(f"Image_path must be list or str of list, got {type(obj)}")

        # ensure 2D: if it's 1D, wrap it
        if len(obj) > 0 and not isinstance(obj[0], list):
            obj = [obj]

        # normalize to List[List[str]]
        return [[str(x) for x in sub] for sub in obj]

    def parse_special_text(self, text):
        pattern = r'(<video>|<image>)'
        parts = re.split(pattern, text)

        result = []
        video_count = -1
        image_count = -1
        for part in parts:
            if not part:
                continue
            elif part == '<video>':
                video_count += 1
                result.append(f'<video>_{video_count}')
            elif part == '<image>':
                image_count += 1
                result.append(f'<image>_{image_count}')
            else:
                result.append(part)
        return result

    def build_prompt(self, line, video_llm=None):
        """
        Build the textual prompt and visual inputs for MMSI-Video.

        In the VLMEvalKit pipeline, *video-LLM models* are usually
        given raw video file paths and are allowed to decode and temporally
        sample frames by themselves (e.g. using their own fps / max_frames
        policy).

        For MMSI-Video, however, the benchmark already provides per-clip
        frame lists, and the underlying videos may start at non-zero
        timestamps. If we forced each video-LLM backend to resample directly
        from the mp4 files, every model wrapper would need to reimplement
        the same timing and sampling logic, which is complex and error-prone.

        To keep the pipeline simple and consistent across models, we instead
        treat the provided `image_path` frame lists as the canonical
        timeline, and only apply our nframe/fps subsampling on these frames,
        rather than resampling from the original videos in `video_list`.
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        # 1. Build prompt
        system_prompt = line['system_prompt']
        task_prompt = line['task_prompt']
        user_prompt = line['user_prompt']
        format_prompt = line['format_prompt']

        prompt = system_prompt + '\n' + task_prompt + user_prompt + format_prompt

        # 2. Video frames processing
        image_path = self.parse_image_path(line['image_path'])
        video_fps = line['video_fps']

        input_frames_per_seg = self.save_video_frames(image_path, video_fps)

        # 3. Fetch ref images
        ref_images = toliststr(line['ref_images'])

        # 4. Prepare input messages
        assert prompt.count('<video>') == len(input_frames_per_seg), \
            f"num <video> != num segments: {prompt.count('<video>')} vs {len(input_frames_per_seg)}"
        assert prompt.count('<image>') == len(ref_images), \
            f"num <image> != num ref_images: {prompt.count('<image>')} vs {len(ref_images)}"

        message = []

        split_list = self.parse_special_text(prompt)
        for item in split_list:
            if '<video>' in item:
                seg_idx = int(item.split('_')[-1])
                for frame_path in input_frames_per_seg[seg_idx]:
                    message.append(dict(type='image', value=frame_path))

            elif '<image>' in item:
                img_idx = int(item.split('_')[-1])
                img_path = ref_images[img_idx]
                message.append(dict(type='image', value=img_path))

            else:
                text_piece = item
                if text_piece:
                    message.append(dict(type='text', value=text_piece))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import eval_mcq_score, build_mcq_score_fn

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        category_task_order = self.category_task_order()
        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col=['task_type', 'sub_task_type'],
            order={
                'task_type': list(category_task_order.keys()),
                'sub_task_type': sum(category_task_order.values(), []),
            },
            dataset_name=getattr(self, 'dataset_name', 'MMSIVideoBench')
        )
