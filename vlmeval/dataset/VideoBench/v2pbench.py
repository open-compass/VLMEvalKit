from huggingface_hub import snapshot_download
from vlmeval.smp import *
from .video_base import VideoBaseDataset
import os
import zipfile
import shutil
import json
import pandas as pd
import re


# for mimo-vl
def remove_think_blocks(text: str) -> str:
    """
    Remove all occurrences of <think>...</think> or <think>...</think>
    (including the tags) from the input text.
    Handles multiline content and multiple blocks.
    """
    # 修复 W605: 使用原始字符串 r'' 来处理反斜杠转义
    pattern = r'<think>.*?(?:<\/think>|<\\/think>)'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]


def xlsx2json(xlsx_file, json_file):
    df = pd.read_excel(xlsx_file)
    df.to_json(json_file, orient='records')


def calu_acc_main(file_path, txt_file=None):
    # Load data
    with open(file_path, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    durations = [0, 240, 1800, 7200]
    dim_mapping = {
        1: "OA", 2: "HA", 3: "OD", 4: "FM", 5: "CR", 6: "PU", 7: "CI",
        9: "FT", 10: "RT", 12: "AS", 13: "SR", 14: "GC"
    }

    dim_nums = 16
    dim_list_sum = [0] * dim_nums
    dim_list_cor = [0] * dim_nums

    short_cor, short_sum = 0, 0
    medium_cor, medium_sum = 0, 0
    long_cor, long_sum = 0, 0

    f = open(txt_file, "w", encoding="utf-8") if txt_file else None

    def log(msg):
        """Print to both the console and the file"""
        print(msg)
        if f:
            f.write(msg + "\n")

    for line in data:
        dim = line["dimension"]
        dim_list_sum[dim - 1] += 1

        if line["duration"] < durations[1]:
            short_sum += 1
        elif line["duration"] < durations[2]:
            medium_sum += 1
        else:
            long_sum += 1

        if line["score"] == 1:
            dim_list_cor[dim - 1] += 1
            if line["duration"] < durations[1]:
                short_cor += 1
            elif line["duration"] < durations[2]:
                medium_cor += 1
            else:
                long_cor += 1

    for index, (dim_cor, dim_sum) in enumerate(zip(dim_list_cor, dim_list_sum)):
        if index + 1 not in [8, 11, 15, 16]:
            if dim_sum != 0:
                log(f"{dim_mapping[index + 1]}: {dim_cor / dim_sum:.3f}")
            else:
                log(f"Dimension is zero: {dim_mapping[index + 1]}")

    log("-" * 58)
    if short_sum != 0:
        log(f"Short\nCorrect: {short_cor}, Total: {short_sum}, Accuracy: {short_cor / short_sum:.3f}")
    if medium_sum != 0:
        log(f"Medium\nCorrect: {medium_cor}, Total: {medium_sum}, Accuracy: {medium_cor / medium_sum:.3f}")
    if long_sum != 0:
        log(f"Long\nCorrect: {long_cor}, Total: {long_sum}, Accuracy: {long_cor / long_sum:.3f}")

    log("-" * 58)
    cor_data = sum(dim_list_cor)
    all_data = sum(dim_list_sum)
    log(f"Total Correct: {cor_data}")
    log(f"Total Success: {all_data}")
    log(f"Accuracy:      {cor_data / all_data:.3f}")

    if f:
        f.close()


prompt_multi_choice = "\nGive only your option letter that accurately addresses the question, no other words."


# unzip video files
def unzip_video(pth):
    videos_dir = os.path.join(pth, "videos/")

    if os.path.exists(os.path.join(videos_dir, "ActivityNet")) and \
       os.path.exists(os.path.join(videos_dir, "EgoSchema")) and \
       os.path.exists(os.path.join(videos_dir, "LVBench")) and \
       os.path.exists(os.path.join(videos_dir, "MLVU")) and \
       os.path.exists(os.path.join(videos_dir, "MSRVTT-QA")) and \
       os.path.exists(os.path.join(videos_dir, "MSVD-QA")) and \
       os.path.exists(os.path.join(videos_dir, "MVBench")) and \
       os.path.exists(os.path.join(videos_dir, "NExTVideo")) and \
       os.path.exists(os.path.join(videos_dir, "Perception_Test")) and \
       os.path.exists(os.path.join(videos_dir, "TVBench")) and \
       os.path.exists(os.path.join(videos_dir, "VCGBench-Diverse")) and \
       os.path.exists(os.path.join(videos_dir, "Video-MME_xk")) and \
       os.path.exists(os.path.join(videos_dir, "Video-MME_yk")):
        print("All videos have been extracted. Skipping extraction step.")
    else:
        for fname in os.listdir(videos_dir):
            if fname.endswith(".zip"):
                zip_path = os.path.join(videos_dir, fname)
                base_name = os.path.splitext(fname)[0]
                target_dir = os.path.join(videos_dir, base_name)

                print(f"unzip: {zip_path} -> {videos_dir}")

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(videos_dir)

                nested_dir = os.path.join(target_dir, base_name)
                if os.path.isdir(nested_dir):
                    print(f"Extra directory found: {nested_dir}, moving files...")
                    for item in os.listdir(nested_dir):
                        src_path = os.path.join(nested_dir, item)
                        dst_path = os.path.join(target_dir, item)

                        if os.path.isdir(src_path):
                            shutil.move(src_path, dst_path)
                        else:
                            shutil.move(src_path, target_dir)

                    shutil.rmtree(nested_dir)
                    print(f"Removed extra directory: {nested_dir}")

                print(f"Extraction completed: {zip_path}\n")

    if os.path.exists(os.path.join(pth, "frames")):
        print("Frames already exist, skipping extraction step.")
    else:
        # vp_frames.zip
        zip_path = os.path.join(pth, "vp_frames.zip")
        target_dir = pth

        print(f"unzip: {zip_path} -> {target_dir}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)


class V2PBench(VideoBaseDataset):

    SYS = 'You are an AI assistant responsible for answering questions about videos.'

    FRAMES_TMPL_NOPACK = """
You will be provided with {} separate frames uniformly sampled from a video, \
the frames are provided in chronological order of the video.
Please analyze these images and provide the answer to the question about the video content.
"""

    TYPE = 'Video-VQA'

    def __init__(self, dataset='V2P-Bench', pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['V2P-Bench']

    def prepare_dataset(self, dataset_name='V2P-Bench', repo_id='gaotiexinqu/V2P-Bench'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
                unzip_video(dataset_path)
        self.video_path = dataset_path
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def build_prompt_nopack(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        relative_video_path = os.path.join("videos", line['video_path'])
        video_path = os.path.join(self.video_path, relative_video_path)
        vp_frame_path = os.path.join(self.video_path, "frames", line["frame_path"])

        question = line['question'] + prompt_multi_choice
        print(f"video_llm: {video_llm}")
        if video_llm:
            message = []
            message.append(dict(type='text', value=question))
            message.append(dict(type='video', value=video_path))
            message.append(dict(type='image', value=vp_frame_path))
            return message
        else:
            frames = self.save_video_frames(line['video_path'])
            sys_prompt = self.FRAMES_TMPL_NOPACK.format(len(frames))
            message = [dict(type='text', value=sys_prompt)]
            for im in frames:
                message.append(dict(type='image', value=im))

            message.append(dict(type='image', value=vp_frame_path))
            message.append(dict(type='text', value=question))
        return message

    def build_prompt(self, line, video_llm):
        return self.build_prompt_nopack(line, video_llm)

    @staticmethod
    def remove_side_quote(s, syms=[',', '"', "'"]):
        if np.all([x in syms for x in s]):
            return ''
        while s[0] in syms:
            s = s[1:]
        while s[-1] in syms:
            s = s[:-1]
        return s

    @staticmethod
    def robust_json_load(s):
        try:
            jsons = list(extract_json_objects(s))
            assert len(jsons) == 1
            return jsons[0]
        except:
            if '{' in s and s.find('{') == s.rfind('{'):
                sub_str = s[s.find('{') + 1:].strip()
                lines = sub_str.split('\n')
                res = {}
                for l in lines:
                    l = l.strip()
                    if ': ' in l:
                        key = l.split(': ')[0].strip()
                        val = l.split(': ')[1].strip()
                        key = V2PBench.remove_side_quote(key)
                        val = V2PBench.remove_side_quote(val)
                        if len(key) and len(val):
                            res[key] = val
                return res
            return None

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                data.loc[idx, 'score'] = int(extract_characters_regex(remove_think_blocks(pred)) == ans)

            dump(data, score_file)

        score_json = eval_file.replace('.xlsx', '_score.json')
        txt_file = eval_file.replace('.xlsx', '_score.txt')
        xlsx2json(score_file, score_json)
        calu_acc_main(score_json, txt_file)
