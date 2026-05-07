from vlmeval.dataset.video_base import VideoBaseDataset
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from ..smp.file import load
from PIL import Image
import numpy as np
import portalocker
import zipfile
import ast
import os

NQ_QUESTION_TYPES = [
    "object_counting_single",
    "object_counting_multiple",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation_single",
    "room_size_estimation_multiple"
]


MCQ_QUESTION_TYPES = [
    "object_rel_direction_forward_easy",
    "object_rel_direction_backward_easy",
    "object_rel_direction_forward_hard",
    "object_rel_direction_backward_hard",
    "object_rel_distance_closest",
    "object_rel_distance_farthest",
    "route_planning"
]

PROMPT_PREFIX = "These are frames of a video."
REPO_ID = "3dlg-hcvc/ReVSI"
VIDEO_ROOT_DIR = "revsi_videos"
EXTRACT_SENTINEL = ".extracted"


def _safe_extract_zip(zip_path, target_dir):
    target_dir_abs = os.path.abspath(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            rel_path = os.path.normpath(info.filename).lstrip("/\\")
            dst_path = os.path.abspath(os.path.join(target_dir, rel_path))
            if not dst_path.startswith(target_dir_abs + os.sep):
                raise RuntimeError(f"Unsafe path in zip: {info.filename}")
            if info.is_dir():
                os.makedirs(dst_path, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with zf.open(info, "r") as src, open(dst_path, "wb") as dst:
                dst.write(src.read())


def _write_sentinel(path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("done")
    os.replace(tmp_path, path)


def _serialize_options(options):
    if isinstance(options, np.ndarray):
        options = options.tolist()
    if isinstance(options, (list, tuple)):
        return repr([str(option) for option in options])
    return options


def _parse_options(options):
    if isinstance(options, np.ndarray):
        return [str(option) for option in options.tolist()]
    if isinstance(options, (list, tuple)):
        return [str(option) for option in options]
    if isinstance(options, str):
        return [str(option) for option in ast.literal_eval(options)]
    return [str(options)]


def _mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = (abs(pred - target) / target) <= (1 - conf_intervs)
    return accuracy.mean()


def _pop_mean(output, metric_name, metric_keys):
    values = [output.pop(key) for key in metric_keys if key in output]
    if values:
        output[metric_name] = np.mean(values)


class ReVSI(VideoBaseDataset):

    TYPE = 'Video-VQA'

    def __init__(self, dataset='ReVSI', pack=False, nframe=None, **kwargs):
        if nframe in [None, "all", "all_frame"]:
            self.frame_subset = "all_frame"
            nframe = 128
        else:
            nframe = int(nframe)
            self.frame_subset = f"{nframe}_frame"
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=-1)

    @classmethod
    def supported_datasets(cls):
        return ['ReVSI']

    def prepare_dataset(self, dataset):
        subset = self.frame_subset
        dataset_table = load_dataset(REPO_ID, subset, split="test")
        dataset_table = dataset_table.add_column('video', [f"{x['scene_id']}.mp4" for x in dataset_table])
        df = dataset_table.to_pandas()
        if 'options' in df:
            df['options'] = df['options'].apply(_serialize_options)
        video_zip_path = hf_hub_download(repo_id=REPO_ID, filename="video.zip", repo_type="dataset")
        dataset_path = os.path.dirname(video_zip_path)
        video_root = os.path.join(dataset_path, VIDEO_ROOT_DIR)
        os.makedirs(video_root, exist_ok=True)
        sentinel_path = os.path.join(video_root, EXTRACT_SENTINEL)
        expected_video_path = os.path.join(video_root, subset, df["video"].iloc[0])

        def videos_ready():
            return os.path.exists(sentinel_path) and os.path.isfile(expected_video_path)

        if not videos_ready():
            lock_path = os.path.join(video_root, ".extract.lock")
            with portalocker.Lock(lock_path, "w", timeout=300):
                if not videos_ready():
                    _safe_extract_zip(video_zip_path, video_root)
                    _write_sentinel(sentinel_path)

        tsv_file_path = os.path.join(dataset_path, f"{subset}.tsv")
        df.to_csv(tsv_file_path, sep="\t", index=False)
        return dict(root=video_root, data_file=tsv_file_path)

    def video_path(self, video):
        return os.path.join(self.data_root, self.frame_subset, video)

    def save_video_frames(self, video):
        import decord

        vid_path = self.video_path(video)
        frame_key = os.path.join(self.frame_subset, os.path.splitext(video)[0])
        vid = decord.VideoReader(vid_path)
        video_fps = vid.get_avg_fps()
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(frame_key)
            lock_name = f"{os.path.splitext(video)[0]}.{self.nframe}frame.lock"
        elif self.fps > 0:
            total_duration = len(vid) / video_fps
            required_frames = max(int(total_duration * self.fps), 1)
            step_size = video_fps / self.fps
            indices = [min(int(i * step_size), len(vid) - 1) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(frame_key, len(indices))
            lock_name = f"{os.path.splitext(video)[0]}.{self.fps}fps.lock"
        else:
            raise ValueError('ReVSI requires either nframe > 0 or fps > 0 to extract frames')

        if np.all([os.path.exists(p) for p in frame_paths]):
            return frame_paths

        lock_dir = os.path.join(self.frame_root, self.frame_subset)
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, lock_name)
        with portalocker.Lock(lock_path, "w", timeout=30):
            if np.all([os.path.exists(p) for p in frame_paths]):
                return frame_paths
            images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
            for image, path in zip(images, frame_paths):
                if not os.path.exists(path):
                    image.save(path)
        return frame_paths

    def build_prompt(self, idx, video_llm):
        line = self.data.iloc[idx]
        question_type = line["question_type"]
        question = line["question"]
        if question_type in NQ_QUESTION_TYPES:
            post_prompt = "Answer the question using a single integer or decimal number."
            full_prompt = "\n".join([PROMPT_PREFIX, question, post_prompt]).strip()
        elif question_type in MCQ_QUESTION_TYPES:
            options = _parse_options(line["options"])
            options_str = "Options:\n" + "\n".join(options)
            post_prompt = "Answer with the option's letter from the given choices directly."
            full_prompt = "\n".join([PROMPT_PREFIX, question, options_str, post_prompt]).strip()
        message = []
        if video_llm:
            message.append(dict(type='video', value=self.video_path(line["video"])))
        else:
            for frame_path in self.save_video_frames(line["video"]):
                message.append(dict(type='image', value=frame_path))
        message.append(dict(type='text', value=full_prompt))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        df = load(eval_file)
        for i, row in df.iterrows():
            pred_answer = str(row["prediction"]).strip().split(" ")[0].rstrip(".").strip()
            gt_answer = str(row["ground_truth"])
            if row["question_type"] in MCQ_QUESTION_TYPES:
                accuracy = 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
            elif row["question_type"] in NQ_QUESTION_TYPES:
                try:
                    accuracy = _mean_relative_accuracy(
                        float(pred_answer), float(gt_answer), 0.5, 0.95, 0.05
                    )
                except (TypeError, ValueError, ZeroDivisionError):
                    accuracy = 0.0
            df.at[i, "accuracy"] = accuracy

        output = {}
        for question_type, per_question_type in df.groupby("question_type"):
            output[f"{question_type}_accuracy"] = per_question_type["accuracy"].mean()

        _pop_mean(output, "object_rel_direction_accuracy", [
            "object_rel_direction_forward_easy_accuracy",
            "object_rel_direction_backward_easy_accuracy",
            "object_rel_direction_forward_hard_accuracy",
            "object_rel_direction_backward_hard_accuracy",
        ])
        _pop_mean(output, "object_rel_distance_accuracy", [
            "object_rel_distance_closest_accuracy",
            "object_rel_distance_farthest_accuracy",
        ])
        _pop_mean(output, "object_counting_accuracy", [
            "object_counting_single_accuracy",
            "object_counting_multiple_accuracy",
        ])
        _pop_mean(output, "room_size_estimation_accuracy", [
            "room_size_estimation_single_accuracy",
            "room_size_estimation_multiple_accuracy",
        ])
        output["overall_accuracy"] = sum(output.values()) / len(output) if output else 0.0
        return output
