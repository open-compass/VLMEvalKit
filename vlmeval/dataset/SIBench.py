from .image_base import ImageBaseDataset
from .image_mcq import ImageMCQDataset
from .video_base import VideoBaseDataset
from ..smp import *
import os
import decord
import re
import warnings
from .utils import build_judge, DEBUG_MESSAGE



class SIBench(ImageMCQDataset, ImageBaseDataset, VideoBaseDataset):
    # -------------------------------------------------
    #
    # Before running this script, you must download the
    # required data from the following Hugging Face repo:
    #
    # ==> https://huggingface.co/datasets/Two-hot/SIBench
    #
    # Please download it and place it in the 'data/' directory.
    #
    # -------------------------------------------------
    MODALITY = 'MixedInput'
    TYPE = 'MixedOutput'

    NEED_EXTRA_PROMPT_SOURCE = ['vstibench', 'MMSI-Bench', '3DSRBench', 'OmniSpatial', 'Spatial-MM', 'SpatialMQA',
                         'VSI-Bench', 'STI-Bench', 'SpatialEval', 'SITE-Bench', 'SPHERE-VLM', 'SRBench', 'BLINK'
                         ]
    # do not need = SpatialBench, SPAR-Bench, Super-CLEVR-3D, Omni3D-Bench
    SETTING = ['relative_distance', 'Reach_Prediction', 'Object_Shape', 'Height', 'Existence', 'Spatial_Compatibility',
               'Coordinate_Conversion', 'Counting', 'Route_Planning', 'Trajectory_Description', 'Geometric_Reasoning',
               'Spatial_Imagination', 'Object_Size_Estimation', 'Spatial_Grid', 'Situational_QA', 'Velocity_Acceleration',
               'Maze_Navigation', 'Temporal-Appearance_Order', 'Camera_Pose', 'Occlusion', 'multi-view_reasoning',
               'Object_Localization',"Spatial_Relation", "SIBench", "SIBench-mini"
               ]

# Counting Camera_Pose Coordinate_Conversion multi-view_reasoning Object_Shape Object_Size_Estimation Occlusion relative_distance Situational_QA Spatial_Grid Spatial_Relation Trajectory_Description
# Reach_Prediction Height Existence Spatial_Compatibility Route_Planning Geometric_Reasoning Velocity_Acceleration Spatial_Imagination Temporal-Appearance_Order Object_Localization
    VIDEO_MODALITY_INCLUDED_SETTING = ['']

    FRAMES_TMPL_SYS = """
You will receive {} distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and answer the question based on your observations.
"""
    FRAMES_TMPL_SYS_4VIDEO_LLM = """
You will receive several distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and answer the question based on your observations.
"""
    
    def __init__(self, dataset='MMBench', skip_noimg=True, nframe=30, fps=-1):
        super(SIBench, self).__init__(dataset, skip_noimg)

        self.frame_tmpl = 'frame-{}-of-{}.jpg'
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'

        self.nframe = nframe
        self.fps = fps
        if self.fps > 0 and self.nframe > 0:
            raise ValueError('fps and nframe should not be set at the same time')
        if self.fps <= 0 and self.nframe <= 0:
            raise ValueError('fps and nframe should be set at least one valid value')

    @classmethod
    def supported_datasets(cls):
        return cls.SETTING
    
    def add_extra_prompt(self, prompt, answer_type, data_source):
        if data_source in self.NEED_EXTRA_PROMPT_SOURCE:
            if answer_type == 'MCQ':
                prompt += "\nSelect from the given options, answer with letters only."
            elif answer_type == 'YN':
                prompt += "\nAnswer with 'Yes' or 'No' only."
            elif answer_type.startswith('Number'):
                prompt += "\nAnswer using a single number and nothing else."
            else:
                raise NotImplementedError(f"Answer type '{answer_type}' is not supported. Supported types are: 'MCQ', 'YN', 'Number'.")
        elif data_source is None:
            raise KeyError("Required key 'data_source' is missing.")
        return prompt

    def frame_paths(self, video, data_base):
        # need self.frame_root & self.frame_tmpl & self.nframe
        frame_root = osp.join(data_base, video.split('/')[0], 'frames')
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def save_video_frames(self, line, data_base):
        # need self.nframe & self.fps
        video = line['video_path']
        vid_path = os.path.normpath(os.path.join(data_base, line['video_path']))
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video, data_base)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def save_video_into_images(self, line, data_base):
        frame_paths = self.save_video_frames(line, data_base)
        return frame_paths
    
    def build_prompt_for_video(self, line, video_llm, data_base):
        # need video_llm 
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = os.path.normpath(os.path.join(data_base, line['video_path']))
        prompt = line['question']
        answer_type = line.get('type')
        data_source = line.get('data_source')
        prompt = self.add_extra_prompt(prompt, answer_type, data_source)

        if video_llm: # video_llm
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS_4VIDEO_LLM)]
            message.append(dict(type='text', value=prompt))
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line, data_base)
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS.format(len(img_frame_paths)))]
            message.append(dict(type='text', value=prompt))
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        return message

    def build_prompt_for_image(self, line, data_base):
        msgs = []
        if line.get('image_path'):
            tgt_path = toliststr(''.join(line['image_path'].split()).split(','))
            for _ in range(len(tgt_path)):
                tgt_path[_] = os.path.join(data_base, tgt_path[_])
        else:
            raise KeyError("Required key 'image_path' is missing.")

        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        
        question = line['question']
        prompt = question
        answer_type = line.get('type')
        data_source = line.get('data_source')
        prompt = self.add_extra_prompt(prompt, answer_type, data_source)
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def build_prompt(self, line, video_llm=None, data_base='.'):
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        if line.get('input_type') in ['image', 'multi-view']:
            return self.build_prompt_for_image(line=line, data_base=data_base)
        elif line.get('input_type') == 'video':
            video_data_base = data_base.replace('/data', '/data_sampled_video')
            return self.build_prompt_for_video(line=line, video_llm=video_llm, data_base=video_data_base)
        else:
            raise NotImplementedError(f"Unrecognized input type: {line.get('input_type')}. Just support 'image', 'multi-view' and 'video'.")

    def extract_numbers_from_string(self, text, reverse_order):
        number_strings = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
        result = []
        for num_str in number_strings:
            cleaned_str = num_str.replace(',', '')
            try:
                result.append(float(cleaned_str))
            except ValueError:
                continue
                
        if reverse_order:
            result.reverse()
                
        return result
    
    def compute_mra(self, y_true, y_pred):
        C = np.arange(0.5, 1.0, 0.05)
        mra_sum = 0
        for theta in C:
            relative_error = np.abs(y_pred - y_true) / y_true
            if relative_error < (1 - theta):
                mra_sum += 1
        mra = mra_sum / len(C)
        return mra

    def yn_Extraction(self, pred):
        pred = pred.strip().lower()
        pred = re.sub(r'[^\w\s]', '', pred)

        if pred == "yes":
            return "yes"
        elif pred == "no":
            return "no"
        else:
            return pred

    def check_string_format(self, s):
        # 1: ("A.", "B:", etc.)
        if re.match(r'^[A-F][\.\:]', s):
            return True
        # 2: ("(A)", " (A)", etc.)
        if '(' in s[:3]:
            return True
        # 3: ("A", "Apple", "A Answer", etc.)
        if s[0] in 'ABCDEF':
            return True

        return False

    def mcq_check(self, predict):
        if isinstance(predict, float):
            predict = 'z'
        if '(' in predict[:3]:
            predict = predict[1]
        predict = predict.split('.')[0].split(':')[0]

        return predict


    def build_prompt_mcq(self, reasoning_text):
        prompt_template = """You are a multiple-choice answer extractor.
            Your sole task is to identify the final answer from a piece of reasoning text and return *only* the corresponding option letter.
            Your response must strictly follow the format: return only the option letter, enclosed in English double quotes. Do not include any other text, explanation, or prefixes.
            ---
            **Example 1:**
            **Input:** "Based on the analysis, options A and B are clearly wrong. Option C mentions... This is correct. Therefore, the final answer is C."
            **Output:** "C"
            **Example 2:**
            **Input:** "Let's go through them one by one. A... B... C... D... After a comprehensive comparison, option A's description is the most complete and accurate. So, the answer is A."
            **Output:** "A"
            **Example 3:**
            **Input:** "The analysis shows that B is the correct choice because..."
            **Output:** "B"
            ---
            Now, strictly following the format above, extract the answer from the following text:

            """
        return prompt_template + reasoning_text

    def llm_process(self, pred, model):
        prompt = self.build_prompt_mcq(pred)
        logger = get_logger('Evaluation')
        retry = 3

        while retry:
            ans = model.generate(prompt).strip(" '\"")
            if 'Failed to obtain answer via API' in ans:
                logger.warning('GPT API failed to answer. ')
            else:
                if ans:
                    return ans #dict(opt=ans, log=ans)
                else:
                    logger.warning(
                        f'Failed to in infer: prediction is {ans}'
                    )
            retry -= 1

            if retry == 0:
                return 'z' #dict(opt='z', log='Failed to predict')

    def extract_mcq(self, pred, model):
        need_llm = not self.check_string_format(pred)
        if need_llm:
            pred = self.llm_process(pred, model)

        return self.mcq_check(pred)


    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import extract_characters_regex, report_acc
        from .utils.yorn import YOrN_Extraction
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        FAIL_MSG = 'Failed to obtain answer via API.'
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        # tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')
        score_file_csv = eval_file.replace('.xlsx', '_score.csv')

        model = build_judge(**judge_kwargs)
        if not model.working():
            warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
            warnings.warn(DEBUG_MESSAGE)
            model = None

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            cnt_rejected = 0
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]
                output_type = data.loc[data['index'] == idx, 'type'].values[0]

                if output_type == 'MCQ':
                    extract_pred = self.extract_mcq(pred, model) # extract_characters_regex(pred)
                    if extract_pred == '':
                        cnt_rejected += 1
                        data.loc[data['index'] == idx, 'hit'] = 0
                    else:
                        data.loc[data['index'] == idx, 'hit'] = int(extract_pred == ans)
                elif output_type == 'YN':
                    extract_pred_yn = self.yn_Extraction(pred[:3]) # YOrN_Extraction(pred)
                    ans_yn = self.yn_Extraction(ans[:3])
                    if ans_yn == 'yes' or ans_yn == 'no':
                        ans = ans_yn
                        pred = extract_pred_yn
                    if pred == 'Unknown':
                        cnt_rejected += 1
                        data.loc[data['index'] == idx, 'hit'] = 0
                    else:
                        data.loc[data['index'] == idx, 'hit'] = int(pred.strip().lower() == ans.strip().lower())
                elif output_type.startswith('Number'):
                    try:
                        extract_pred = eval(str(pred.strip()))
                    except Exception:
                        extract_pred = -1.0 #pred.strip()  # self.extract_numbers_from_string(pred, True)

                    ans = eval(str(ans))
                    if output_type == 'Number': 
                        data.loc[data['index'] == idx, 'hit'] = self.compute_mra(ans, extract_pred) #data.loc[data['index'] == idx, 'hit'] = 0 #
                    elif output_type == 'Number_Int':
                        data.loc[data['index'] == idx, 'hit'] = int(extract_pred == ans)
                    else:
                        NotImplementedError(f'Unsupported output type {output_type}.')

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {cnt_rejected} questions. '
                f'Those questions will be counted as 0 score in ALL rating.'
            )

            dump(data, score_file)
        data = load(score_file)
        acc = report_acc(data)
        dump(acc, score_file_csv)
        return acc
