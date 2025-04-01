from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich
import random
import json
import ast
from glob import glob
from tqdm import tqdm

FAIL_MSG = 'Failed to obtain answer via API.'


detailed_caption_prompts = [
    "The images are given containing equally spaced video frames. Please imagine the video based on the sequence of frames, and provide a faithfully detailed description of this video in more than three sentences.",
    "You are given a sequence of equally spaced video frames. Based on these frames, imagine the full video and provide a detailed description of what is happening in more than three sentences.",
    "The following set contains equally spaced video frames. Imagine the video from which these frames were taken and describe it in detail in at least three sentences.",
    "Below are equally spaced frames from a video. Use these frames to visualize the entire video and provide a detailed description in more than three sentences.",
    "A sequence of equally spaced video frames is presented. Please imagine the full video and write a faithfully detailed description of the events in more than three sentences.",
    "The images provided include equally spaced frames from a video. Based on these frames, imagine the video and describe it comprehensively in at least three sentences.",
    "You are given equally spaced frames from a video. Use these frames to envision the entire video and provide a detailed description of the events in more than three sentences.",
    "The sequence includes equally spaced frames from a video. Imagine the full video based on these frames and provide a detailed description in more than three sentences.",
    "The provided images contain equally spaced frames from a video. Visualize the video from these frames and describe it in detail in more than three sentences.",
    "Here are equally spaced frames from a video. Based on these frames, imagine the video and provide a detailed, faithful description of it in more than three sentences.",
    "The set of images includes equally spaced video frames. Please imagine the video these frames come from and describe it comprehensively in at least three sentences.",
    "Describe the video based on these frames in a few sentences.",
    "What is happening in the video shown in these frames?",
    "Explain the video using these frames.",
    "Imagine the video from these frames and describe it in detail in a few sentences.",
    "Based on these frames, provide a narrative of the video in more than three sentences.",
    "Describe the events in the video shown by these frames in at least three sentences.",
    "Visualize the video from these frames and explain what is happening in more than three sentences.",
    "Describe the sequence of events in the video depicted by these frames in a detailed manner.",
    "Given these equally spaced frames, imagine the entire video and provide a detailed description of the events, including the setting, characters, and actions, in more than three sentences.",
    "Visualize the video based on these frames and write a comprehensive description of what happens, describing the beginning, middle, and end in at least three sentences.",
    "Using these frames as a reference, imagine the full video and provide a thorough description of the plot, including key details and actions, in more than three sentences.",
    "Based on the sequence of these frames, describe the entire video in detail, mentioning important aspects such as the context, movements, and transitions in more than three sentences.",
    "Imagine the video that corresponds to these frames and provide an elaborate description, covering the storyline, visual elements, and any notable features in at least three sentences."
]


class VideoMMLU_CAP(VideoBaseDataset):

    MD5 = ''

    TYPE = 'Video-VQA'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='Video_MMLU_CAP', pack=False, nframe=0, fps=-1, subset='all', limit=1.0):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

        if subset == 'all':
            pass
        elif subset == 'Math':
            self.data = self.data[self.data['discipline'] == 'Math']
        elif subset == 'physics':
            self.data = self.data[self.data['discipline'] == 'physics']
        elif subset == 'chemistry':
            self.data = self.data[self.data['discipline'] == 'chemistry']
        else:
            raise ValueError(f'Invalid subset: {subset}')
        
        if limit <= 1.0 and limit > 0:
            sample_num = int(limit * len(self.data))
            self.data = self.data.iloc[:sample_num]
        elif limit > 1.0 and limit < len(self.data):
            self.data = self.data.iloc[:limit]
        else:
            raise ValueError(f'Invalid limit: {limit}')
    

    @classmethod
    def supported_datasets(cls):
        return ['VideoMMLU_CAP']

    def prepare_dataset(self, dataset_name='Video_MMLU_CAP', repo_id='Enxin/Video-MMLU'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video']:
                if not osp.exists(osp.join(pth, 'youtube_videos', video_pth)):
                    return False
            return True

        def untar_video_data(archive_file, cache_dir):
            import tarfile
            with tarfile.open(archive_file, "r") as tar_ref:
                tar_ref.extractall(cache_dir)
                print(f"Extracted all files from {archive_file} to {cache_dir}")

        def unzip_video_data(archive_file, cache_dir):
            import zipfile
            with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
                print(f"Extracted all files from {archive_file} to {cache_dir}")

        def concat_archive_parts(parts, output_file):
            with open(output_file, "wb") as out_file:
                from tqdm import tqdm
                for part in tqdm(sorted(parts)):
                    with open(part, "rb") as part_file:
                        out_file.write(part_file.read())
            print(f"Concatenated parts {parts} into {output_file}")

        if os.path.exists(repo_id):
            dataset_path = repo_id
        else:
            cache_path = get_cache_path(repo_id)
            if cache_path is not None and check_integrity(cache_path):
                dataset_path = cache_path
            else:
                cache_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
                if not glob(osp.join(cache_path, "youtube_videos")):
                    # 查找所有的压缩文件
                    tar_files = glob(osp.join(cache_path, "**/*.tar*"), recursive=True)
                    zip_files = glob(osp.join(cache_path, "**/*.zip*"), recursive=True)
                    
                    parts_dict = {}
                    # 分组处理tar文件
                    for f in tar_files:
                        base_name = f.split(".tar")[0]
                        if base_name not in parts_dict:
                            parts_dict[base_name] = {'type': 'tar', 'parts': []}
                        parts_dict[base_name]['parts'].append(f)
                    
                    # 分组处理zip文件
                    for f in zip_files:
                        base_name = f.split(".zip")[0]
                        if base_name not in parts_dict:
                            parts_dict[base_name] = {'type': 'zip', 'parts': []}
                        parts_dict[base_name]['parts'].append(f)

                    # 处理所有压缩文件
                    for base_name, info in parts_dict.items():
                        print(f"Processing archive: {base_name}")
                        archive_type = info['type']
                        parts = info['parts']
                        
                        # 设置输出文件名
                        output_file = base_name + (".tar" if archive_type == 'tar' else ".zip")
                        
                        # 如果有多个部分，先合并
                        if len(parts) > 1 and not osp.exists(output_file):
                            print('Start concatenating archive parts')
                            concat_archive_parts(parts, output_file)
                        elif len(parts) == 1:
                            output_file = parts[0]

                        # 解压文件
                        if not osp.exists(osp.join(cache_path, osp.basename(base_name))):
                            if archive_type == 'tar':
                                untar_video_data(output_file, cache_path)
                            else:
                                unzip_video_data(output_file, cache_path)

                dataset_path = cache_path

        self.video_path = osp.join(dataset_path, 'youtube_videos/')
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=osp.join(dataset_path, 'video'))

    def build_prompt_pack(self, line):
        if isinstance(line, int):
            assert line < len(self)
            video = self.videos[line]
        elif isinstance(line, pd.Series):
            video = line['video']
        elif isinstance(line, str):
            video = line

        frames = self.save_video_frames(video)
        message = []
        for im in frames:
            message.append(dict(type='image', value=im))

        prompt = random.choice(detailed_caption_prompts)
        message.append(dict(type='text', value=prompt, role='user'))
        return message

    def build_prompt_nopack(self, line, video_llm):
        """Build prompt for a single line without packing"""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        
        prompt = random.choice(detailed_caption_prompts)

        if video_llm:
            video_path = os.path.join(self.video_path, line['video'])
            return [
                dict(type='video', value=video_path),  
                dict(type='text', value=prompt)
            ]
        else:
            frames = self.save_video_frames(os.path.splitext(line['video'])[0])
            message = []
            for im in frames:
                message.append(dict(type='image', value=im))
            message.append(dict(type='text', value=prompt))
            return message

    def build_prompt(self, line, video_llm):
        if self.pack and not video_llm:
            return self.build_prompt_pack(line)
        else:
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
                        key = VideoMMLU_CAP.remove_side_quote(l.split(': ')[0].strip())
                        val = VideoMMLU_CAP.remove_side_quote(l.split(': ')[1].strip())
                        if len(key) and len(val):
                            res[key] = val
                return res
            return None

    def load_pack_answers(self, data_raw):
        vstats = defaultdict(lambda: 0)
        data = defaultdict(lambda: {})

        for k in data_raw:
            ans = data_raw[k].strip()
            if FAIL_MSG in ans:
                vstats['GEN_FAIL'] += 1
                continue
            res = self.robust_json_load(ans)
            if res is not None:
                data[k] = res
                vstats['PARSE_OK'] += 1
            else:
                vstats['PARSE_FAIL'] += 1

        # return data
        meta = cp.deepcopy(self.data)
        lt = len(meta)
        prediction = []
        for i in range(lt):
            line = meta.iloc[i]
            vid = line['video']
            idx = str(line['index'])
            prediction.append(data[vid][idx] if idx in data[vid] else None)
        meta['prediction'] = prediction
        vstats['VALIDQ'] = len([x for x in prediction if x is not None])
        vstats['INVALIDQ'] = len([x for x in prediction if x is None])
        return meta, vstats

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.video_mmlu import get_dimension_rating, prepare_response_prompt, prepare_score_prompt, SYSTEM_CAL_SCORE_PROMPT_CAP, SYSTEM_GENER_PRED_PROMPT

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        judge = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 4)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)
        
        response_file = eval_file.replace('.xlsx', f'_{judge}_response.pkl')
        tmp_file = eval_file.replace('.xlsx', f'_{judge}_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', f'_{judge}_rating.json')
        score_file = eval_file.replace('.xlsx', f'_{judge}_score.xlsx')

        judge_kwargs['temperature'] = 0.0
        model = build_judge(**judge_kwargs)

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            
            expanded_data = []
            for idx, row in data.iterrows():
                try:
                    questions = ast.literal_eval(row['question']) if isinstance(row['question'], str) else row['question']
                    for q_dict in questions:
                        new_row = row.copy()
                        new_row['question'] = q_dict['question']
                        new_row['answer'] = q_dict['answer']   
                        expanded_data.append(new_row)
                except Exception as e:
                    print(f"Error parsing questions for row {idx}")
                    print(f"Error message: {str(e)}")
                    continue
            
            expanded_df = pd.DataFrame(expanded_data).reset_index(drop=True)
            
            data_un = expanded_df[~expanded_df['index'].isin(res)]
            data_un = data_un[~pd.isna(data_un['prediction'])]
            lt = len(data_un)
            
            response_prompts = [prepare_response_prompt(data_un.iloc[i]) for i in range(lt)]
            indices = [data_un.iloc[i]['index'] for i in range(lt)]
            
            model.system_prompt = SYSTEM_GENER_PRED_PROMPT
            if len(response_prompts):
                print(f"Processing {len(response_prompts)} valid prompts out of {lt} total items")
                _ = track_progress_rich(
                    model.generate,
                    response_prompts,
                    keys=indices,
                    save=response_file,
                    nproc=nproc,
                    chunksize=nproc
                )

            pred_map = load(response_file)
            data_un['pred_response'] = [pred_map[idx] for idx in data_un['index']]
            score_prompts = [prepare_score_prompt(data_un.iloc[i]) for i in range(lt)]
            model.system_prompt = SYSTEM_CAL_SCORE_PROMPT_CAP
            if len(score_prompts):
                _ = track_progress_rich(
                    model.generate,
                    score_prompts,
                    keys=indices,
                    save=tmp_file,
                    nproc=nproc,
                    chunksize=nproc
                )

            score_map = load(tmp_file)
            data['score'] = [score_map[idx] for idx in data['index']]

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating


class VideoMMLU_QA(VideoBaseDataset):

    MD5 = ''

    TYPE = 'Video-VQA'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='Video_MMLU_QA', pack=False, nframe=0, fps=-1, subset='all', limit=1.0):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

        if subset == 'all':
            pass
        elif subset == 'Math':
            self.data = self.data[self.data['discipline'] == 'Math']
        elif subset == 'physics':
            self.data = self.data[self.data['discipline'] == 'physics']
        elif subset == 'chemistry':
            self.data = self.data[self.data['discipline'] == 'chemistry']
        else:
            raise ValueError(f'Invalid subset: {subset}')
        
        if limit <= 1.0 and limit > 0:
            sample_num = int(limit * len(self.data))
            self.data = self.data.iloc[:sample_num]
        elif limit > 1.0 and limit < len(self.data):
            self.data = self.data.iloc[:limit]
        else:
            raise ValueError(f'Invalid limit: {limit}')
    

    @classmethod
    def supported_datasets(cls):
        return ['VideoMMLU_QA']

    def prepare_dataset(self, dataset_name='Video_MMLU_QA', repo_id='Enxin/Video-MMLU'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video']:
                if not osp.exists(osp.join(pth, 'youtube_videos', video_pth)):
                    return False
            return True

        if os.path.exists(repo_id):
            dataset_path = repo_id
        else:
            cache_path = get_cache_path(repo_id)
            if cache_path is not None and check_integrity(cache_path):
                dataset_path = cache_path
            else:
                cache_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
                if not glob(osp.join(cache_path, "youtube_videos")):
                    tar_files = glob(osp.join(cache_path, "**/*.tar*"), recursive=True)

                    def untar_video_data(tar_file, cache_dir):
                        import tarfile
                        with tarfile.open(tar_file, "r") as tar_ref:
                            tar_ref.extractall(cache_dir)
                            print(f"Extracted all files from {tar_file} to {cache_dir}")

                    def concat_tar_parts(tar_parts, output_tar):
                        with open(output_tar, "wb") as out_tar:
                            from tqdm import tqdm
                            for part in tqdm(sorted(tar_parts)):
                                with open(part, "rb") as part_file:
                                    out_tar.write(part_file.read())
                        print(f"Concatenated parts {tar_parts} into {output_tar}")

                    tar_parts_dict = {}

                    # Group tar parts together
                    for tar_file in tar_files:
                        base_name = tar_file.split(".tar")[0]
                        if base_name not in tar_parts_dict:
                            tar_parts_dict[base_name] = []
                        tar_parts_dict[base_name].append(tar_file)

                    # Concatenate and untar split parts
                    for base_name, parts in tar_parts_dict.items():
                        print(f"Extracting following tar files: {parts}")
                        output_tar = base_name + ".tar"
                        if not osp.exists(output_tar):
                            print('Start concatenating tar files')

                            concat_tar_parts(parts, output_tar)
                            print('Finish concatenating tar files')

                        if not osp.exists(osp.join(cache_path, osp.basename(base_name))):
                            untar_video_data(output_tar, cache_path)
                dataset_path = cache_path
        self.video_path = osp.join(dataset_path, 'youtube_videos/')
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=osp.join(dataset_path, 'video'))

    def build_prompt_pack(self, line):
        if isinstance(line, int):
            assert line < len(self)
            video = self.videos[line]
        elif isinstance(line, pd.Series):
            video = line['video']
        elif isinstance(line, str):
            video = line

        frames = self.save_video_frames(video)
        message = []
        for im in frames:
            message.append(dict(type='image', value=im))

        prompt = line['question']+ '\nAnswer briefly and directly in one sentence.'
        message.append(dict(type='text', value=prompt, role='user'))
        return message

    def build_prompt_nopack(self, line, video_llm):
        """Build prompt for a single line without packing"""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        
        prompt = line['question'] + '\nAnswer briefly and directly in one sentence.'

        if video_llm:
            video_path = os.path.join(self.video_path, line['video'])
            return [
                dict(type='video', value=video_path),  
                dict(type='text', value=prompt)
            ]
        else:
            frames = self.save_video_frames(os.path.splitext(line['video'])[0])
            message = []
            for im in frames:
                message.append(dict(type='image', value=im))
            message.append(dict(type='text', value=prompt))
            return message

    def build_prompt(self, line, video_llm):
        if self.pack and not video_llm:
            return self.build_prompt_pack(line)
        else:
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
                        key = VideoMMLU_QA.remove_side_quote(l.split(': ')[0].strip())
                        val = VideoMMLU_QA.remove_side_quote(l.split(': ')[1].strip())
                        if len(key) and len(val):
                            res[key] = val
                return res
            return None

    def load_pack_answers(self, data_raw):
        vstats = defaultdict(lambda: 0)
        data = defaultdict(lambda: {})

        for k in data_raw:
            ans = data_raw[k].strip()
            if FAIL_MSG in ans:
                vstats['GEN_FAIL'] += 1
                continue
            res = self.robust_json_load(ans)
            if res is not None:
                data[k] = res
                vstats['PARSE_OK'] += 1
            else:
                vstats['PARSE_FAIL'] += 1

        # return data
        meta = cp.deepcopy(self.data)
        lt = len(meta)
        prediction = []
        for i in range(lt):
            line = meta.iloc[i]
            vid = line['video']
            idx = str(line['index'])
            prediction.append(data[vid][idx] if idx in data[vid] else None)
        meta['prediction'] = prediction
        vstats['VALIDQ'] = len([x for x in prediction if x is not None])
        vstats['INVALIDQ'] = len([x for x in prediction if x is None])
        return meta, vstats

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.video_mmlu import get_dimension_rating, prepare_score_prompt, SYSTEM_CAL_SCORE_PROMPT_QA

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        judge = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 4)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)
        
        tmp_file = eval_file.replace('.xlsx', f'_{judge}_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', f'_{judge}_rating.json')
        score_file = eval_file.replace('.xlsx', f'_{judge}_score.xlsx')

        judge_kwargs['temperature'] = 0.0
        model = build_judge(**judge_kwargs)

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if model.fail_msg not in v}

            data = load(eval_file)
            data_un = data[~data['index'].isin(res)]
            data_un = data_un[~pd.isna(data_un['prediction'])]
            lt = len(data_un)
            prompts = [prepare_score_prompt(data_un.iloc[i]) for i in range(lt)]
            indices = [data_un.iloc[i]['index'] for i in range(lt)]
            model.system_prompt = SYSTEM_CAL_SCORE_PROMPT_QA
            if len(prompts):
                _ = track_progress_rich(
                    model.generate,
                    prompts,
                    keys=indices,
                    save=tmp_file,
                    nproc=nproc,
                    chunksize=nproc
                )
            score_map = load(tmp_file)
            data['score'] = [score_map[idx] if idx in score_map else -1 for idx in data['index']]
            rejected = [x for x in score_map.values() if FAIL_MSG in x]
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(score_map)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as 0 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating


