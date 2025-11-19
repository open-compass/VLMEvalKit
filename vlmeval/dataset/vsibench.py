# flake8: noqa
import ast
import os.path as osp
import decord
import json
import math
import numpy as np

from ..smp import *
from ..smp.file import load
from .video_base import VideoBaseDataset

from huggingface_hub import snapshot_download
from collections import OrderedDict


class VsiBench(VideoBaseDataset):

    MD5 = ''
    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    # EASI system prompt format from Holistic Evaluation of Multimodal LLMs on Spatial Intelligence. (https://arxiv.org/pdf/2508.13142)
    EASI_MCQ_SYS_PROMPT = (
        "You are a spatial-reasoning assistant. Always ground your answer in the visual evidence; "
        "do not hallucinate unseen objects. If uncertain, pick the most plausible option—never refuse or reply "
        "“insufficient information.” Think step by step and provide the answer. "
        "You should first provide a reasoning process, then provide a single option (an English letter) "
        "as the final answer. The reasoning process and the answer are enclosed within <think></think> "
        "and <answer></answer> tags, respectively, i.e., <think>reasoning process</think>, "
        "<answer>answer</answer>."
    )
    EASI_VQA_SYS_PROMPT = (
        "You are a spatial-reasoning assistant. Always ground your answer in the visual evidence; "
        "do not hallucinate unseen objects. If uncertain, pick the most plausible option—never refuse or reply "
        "“insufficient information. Think step by step and provide the answer. "
        "You should first provide a reasoning process, then provide one float number as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>. "
    )

    ORIGIN_PRE_PROMPT = "These are frames of a video."
    ORIGIN_MCQ_POST_PROMPT = "Answer with the option's letter from the given choices directly."
    ORIGIN_VQA_POST_PROMPT = "Answer briefly and directly in one float number."

    LMUData_root = LMUDataRoot()
    DATASET_URL = {}

    DATASET_URL = {
        "VSI-Bench": "https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench.tsv",  # noqa: E501
        "VSI-Bench-Debiased": "https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench-Debiased.tsv",  # noqa: E501
    }
    DATASET_MD5 = {key: None for key in DATASET_URL}

    def __init__(self, dataset, pack=False, nframe=0, fps=-1, sample_strategy='uniform_tail'):
        self.sample_strategy = sample_strategy
        self.base_name, self.is_debiased, self.variant = self.parse_dataset_name(dataset)
        print(f"VsiBench variant={self.variant}, debiased={self.is_debiased}")

        valid_strategies = {'uniform_tail', 'uniform', 'chunk_center'}
        if sample_strategy not in valid_strategies:
            raise ValueError(f"[{dataset}] Unsupported sample_strategy '{sample_strategy}'")

        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @staticmethod
    def parse_dataset_name(name: str,
                           default_variant: str = "origin",
                           default_base: str = "VSI-Bench"):
        if not isinstance(name, str):
            return default_variant, default_base, False

        lower = name.lower()

        is_debiased = "debiased" in lower
        base_name = "VSI-Bench-Debiased" if is_debiased else "VSI-Bench"

        if lower.endswith("_standard"):
            variant = "standard"
        elif lower.endswith("_origin"):
            variant = "origin"
        else:
            variant = default_variant

        return base_name, is_debiased, variant

    @classmethod
    def supported_datasets(cls):
        bases = ["VSI-Bench", "VSI-Bench-Debiased"]
        variants = ["origin", "standard"]
        return [f"{b}_{v}" for b in bases for v in variants]

    def get_task_type(self, question_type):
        MCQ_items = [
            'obj_appearance_order',
            'object_rel_direction_easy',
            'object_rel_direction_hard',
            'object_rel_direction_medium',
            'object_rel_distance',
            'route_planning',
        ]

        NA_items = [
            'object_abs_distance',
            'object_size_estimation',
            'object_counting',
            'room_size_estimation',
        ]

        if question_type in MCQ_items:
            return 'MCQ'
        elif question_type in NA_items:
            return 'NA'
        else:
            raise ValueError(f"Unkwon question type: {question_type}")

    def download_vsibench(self, repo_id='nyu-visionx/VSI-Bench'):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = ".vsibench_extracted"

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text="ok"):
                tmp = sentinel_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
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
                _write_sentinel(sentinel_path, text="done")
                print('VsiBench data extracted to current directory with original layout.')

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        return dataset_path

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[self.base_name]
        md5 = self.DATASET_MD5[self.base_name]

        _ = super().prepare_tsv(url, md5)

        dataset_path = self.download_vsibench()
        self.dataset_path = dataset_path

        variant_data_file = os.path.join(self.LMUData_root, f"{dataset_name}.tsv")

        return dict(data_file=variant_data_file, root=dataset_path)

    def save_video_frames(self, video_path, video_llm=False):
        vid_path = osp.join(self.data_root, 'video', video_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        indices = []

        if self.nframe > 0 and self.fps < 0:
            if self.sample_strategy == 'uniform_tail':
                indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
                if (video_nframes - 1) != indices[-1]:
                    indices.append(video_nframes - 1)

            elif self.sample_strategy == 'uniform':
                indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()

            elif self.sample_strategy == 'chunk_center':
                seg_size = float(video_nframes) / self.nframe
                indices = [int((seg_size / 2) + np.round(seg_size * i)) for i in range(self.nframe)]
                indices = np.clip(indices, 0, video_nframes - 1).tolist()

            else:
                raise ValueError(f"Unsupported sample strategy: {self.sample_strategy}")

            frame_paths = self.frame_paths(video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            if self.sample_strategy == 'uniform_tail' and (video_nframes - 1) != indices[-1]:
                indices.append(video_nframes - 1)

            frame_paths = self.frame_paths_fps(video_path, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question']
        question_type = str(line['question_type'])
        task_type = self.get_task_type(question_type)

        if task_type == "MCQ":
            options = ast.literal_eval(line['candidates'])
            formatted_options = '\n'.join(options)

        # use vsi origin prompt
        if self.variant == 'origin':
            if task_type == 'MCQ':
                prompt = "\n".join([self.ORIGIN_PRE_PROMPT, question, formatted_options, self.ORIGIN_MCQ_POST_PROMPT])
            else:
                prompt = "\n".join([self.ORIGIN_PRE_PROMPT, question, self.ORIGIN_VQA_POST_PROMPT])
        else:
            if task_type == 'MCQ':
                prompt = "\n".join([self.EASI_MCQ_SYS_PROMPT, question, formatted_options])
            else:
                prompt = "\n".join([self.EASI_VQA_SYS_PROMPT, question])

        message = []

        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'])))
        else:
            frames, _, _ = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import compute_mcq_score, compute_na_score

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        data['task_type'] = data['question_type'].apply(self.get_task_type)
        mcq_data = data[data['task_type'] == 'MCQ'].copy()
        na_data  = data[data['task_type'] == 'NA' ].copy()

        if len(mcq_data):
            mcq_scored = compute_mcq_score(mcq_data)
        else:
            mcq_scored = mcq_data

        if len(na_data):
            na_scored  = compute_na_score(na_data)
        else:
            na_scored = na_data

        summary = self._aggregate(mcq_scored, na_scored)

        try:
            to_dump = {
                'mcq_scored': mcq_scored,
                'na_scored': na_scored,
                'summary': summary
            }
            import pickle
            with open(result_file, 'wb') as f:
                pickle.dump(to_dump, f)
            print(f"[save] result saved to {result_file}")
        except Exception as e:
            warnings.warn(f"[save] failed to save result to {result_file}: {e}")

        base_no_suffix = eval_file[:-(len(suffix) + 1)]
        xlsx_path = f"{base_no_suffix}_extract_matching.xlsx"
        acc_tsv_path = f"{base_no_suffix}_acc.tsv"

        try:
            import pandas as pd

            frames = []
            if len(mcq_scored):
                df_mcq = mcq_scored.copy()
                df_mcq['task_type'] = 'MCQ'
                frames.append(df_mcq)
            if len(na_scored):
                df_na = na_scored.copy()
                df_na['task_type'] = 'NA'
                frames.append(df_na)

            if frames:
                merged = pd.concat(frames, axis=0, ignore_index=True)
            else:
                base_mcq_cols = list(mcq_data.columns) if len(mcq_data) else []
                base_na_cols  = list(na_data.columns)  if len(na_data)  else []
                all_cols = list(dict.fromkeys(base_mcq_cols + base_na_cols + [
                    'pred_extracted', 'hit', 'MRA:.5:.95:.05', 'task_type'
                ]))
                merged = pd.DataFrame(columns=all_cols)

            prefer_front = [
                'index', 'question_type', 'task_type',
                'prediction', 'pred_extracted', 'answer',
                'hit', 'MRA:.5:.95:.05'
            ]
            ordered = [c for c in prefer_front if c in merged.columns] + \
                      [c for c in merged.columns if c not in prefer_front]
            merged = merged[ordered]

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                merged.to_excel(writer, sheet_name="ALL", index=False)

            print(f"[save] extract & matching (merged) saved to {xlsx_path}")
        except Exception as e:
            warnings.warn(f"[save] failed to save merged extract xlsx to {xlsx_path}: {e}")

        try:
            acc_df = pd.DataFrame(
                [(k, v) for k, v in summary.items() if k not in ('tabulated_keys', 'tabulated_results')],
                columns=["metric", "value"]
            )
            acc_df.to_csv(acc_tsv_path, sep="\t", index=False)
            print(f"[save] accuracy table saved to {acc_tsv_path}")
        except Exception as e:
            warnings.warn(f"[save] failed to save acc tsv to {acc_tsv_path}: {e}")

        print(f"[{self.dataset_name}] summary: {summary}")
        return summary

    def _aggregate(self, mcq_df: pd.DataFrame, na_df: pd.DataFrame) -> dict:
        output = {}

        if len(mcq_df):
            for qtype, sub in mcq_df.groupby('question_type'):
                output[f"{qtype}_accuracy"] = float(sub['hit'].mean())

        if len(na_df):
            for qtype, sub in na_df.groupby('question_type'):
                output[f"{qtype}_MRA:.5:.95:.05"] = float(sub['MRA:.5:.95:.05'].mean())

        rel_keys = [
            'object_rel_direction_easy_accuracy',
            'object_rel_direction_medium_accuracy',
            'object_rel_direction_hard_accuracy',
        ]
        if all(k in output for k in rel_keys):
            output['object_rel_direction_accuracy'] = (
                output.pop(rel_keys[0]) + output.pop(rel_keys[1]) + output.pop(rel_keys[2])
            ) / 3.0

        output['overall'] = float(sum(output.values()) / len(output)) if output else 0.0

        res = OrderedDict()
        res['overall'] = output['overall'] * 100.0

        ordered_qtypes = [
            "object_counting",
            "object_abs_distance",
            "object_size_estimation",
            "room_size_estimation",
            "object_rel_distance",
            "object_rel_direction",
            "route_planning",
            "obj_appearance_order",
        ]
        metrics_order = ["accuracy", "MRA:.5:.95:.05"]

        for qtype in ordered_qtypes:
            for metric in metrics_order:
                key = f"{qtype}_{metric}"
                if key in output:
                    res[key] = output[key] * 100.0

        tab_keys = ", ".join(list(res.keys()))
        tab_vals = ", ".join([f"{v:.3f}" for v in res.values()])
        print(f"Tabulated results: {tab_keys}")
        print(f"Tabulated results: {tab_vals}")

        res["tabulated_keys"] = tab_keys
        res["tabulated_results"] = tab_vals
        return res
