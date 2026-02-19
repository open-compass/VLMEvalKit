import huggingface_hub
from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich
import ast
import pandas as pd

FAIL_MSG = 'Failed to obtain answer via API.'


class VSIBench(VideoBaseDataset):
    MD5 = '021f7dd80ab879b4f3329290e804ed15'
    TYPE = 'VIDEO-MCQ-and-NA'

    MCA_QUESTION_TYPES = [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    ]
    NA_QUESTION_TYPES = [
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    ]

    def __init__(self, dataset='VSIBench', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['VSIBench']

    def prepare_dataset(self, dataset_name='VSIBench', repo_id='nyu-visionx/VSI-Bench'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'] + '.mp4')):
                    return False
            return True
        cache_path = get_cache_path(repo_id)

        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def unzip_videos(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'arkitscenes')):
                    zip_file = osp.join(pth, 'arkitscenes.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

                if not osp.exists(osp.join(pth, 'scannet')):
                    zip_file = osp.join(pth, 'scannet.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

                if not osp.exists(osp.join(pth, 'scannetpp')):
                    zip_file = osp.join(pth, 'scannetpp.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return
                self.data_list = []
                with open(osp.join(pth, 'test.jsonl'), 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        self.data_list.append({
                            'id': data['id'],
                            'prefix': data['dataset'],
                            'video': data['scene_name'],
                            'type': data['question_type'],
                            'question': data['question'],
                            'answer': data['ground_truth'],
                            'candidates': data['options'],
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_videos(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def save_video_frames(self, line):
        vid_path = osp.join(self.data_root, line['prefix'], line['video'] + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(f"{line['prefix']}/{line['video']}")
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(f"{line['prefix']}/{line['video']}", len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        message = []
        img_frame_paths = self.save_video_frames(line)
        for im in img_frame_paths:
            message.append(dict(type='image', value=im))

        pre_prompt = "These are frames of a video."  # pre_prompt = ""

        assert line['type'] in self.NA_QUESTION_TYPES or line['type'] in self.MCA_QUESTION_TYPES

        if line['type'] in self.NA_QUESTION_TYPES:
            post_prompt = "Please answer the question using a single word or phrase."
            prompt = pre_prompt + "\n" + line['question'] + "\n" + post_prompt
            message.append(dict(type='text', value=prompt))
        else:
            options = "Options:\n" + "\n".join(ast.literal_eval(line["candidates"]))
            post_prompt = "Answer with the option's letter from the given choices directly."
            prompt = "\n".join([pre_prompt, line['question'], options, post_prompt])
            message.append(dict(type='text', value=prompt))
        return message

    @classmethod
    def critic_multichoice(self, pred, ans):  # Evaluation of MCA questions
        if pred.lower() == ans.lower() or pred.lower().startswith(ans.lower() + "."):
            return 1
        else:
            return 0

    @classmethod
    def mra(self, pred, ans):  # Evaluation of NA questions
        try:
            ans_num = float(ans)
            pred_num = float(pred)
            acc = 0
            for i in range(20):
                theta = 0.5 + i * 0.05
                if abs(pred_num - ans_num) / ans_num < 1 - theta:
                    acc += 1
            return acc / 10
        except Exception as e:
            print("Error:", e)
            return 0

    @classmethod
    def eva_one_row(self, row):
        assert row['type'] in self.NA_QUESTION_TYPES or row['type'] in self.MCA_QUESTION_TYPES
        if row['type'] in self.MCA_QUESTION_TYPES:
            return self.critic_multichoice(row['prediction'], row['answer'])
        else:
            return self.mra(row['prediction'], row['answer'])

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        result_file = get_intermediate_file_path(eval_file, '_result', 'xlsx')
        df = pd.read_excel(eval_file)

        df['score'] = df.apply(self.eva_one_row, axis=1)

        overall_stats = [{
            "Type": "overall",
            "Count": len(df),
            "Score_sum": df['score'].sum(),
            "Avg_score": df['score'].sum() / len(df)}
        ]
        type_stats = df.groupby('type').agg(
            count=('score', 'size'),     # Row number
            score_sum=('score', 'sum')   # Score sum
        ).reset_index()

        for id, row in type_stats.iterrows():
            overall_stats.append({
                "Type": row['type'],
                "Count": row['count'],
                "Score_sum": row['score_sum'],
                "Avg_score": row['score_sum'] / row['count']
            })
        overall_stats = pd.DataFrame(overall_stats)
        dump(overall_stats, score_file)
        dump(df, result_file)
        return overall_stats
