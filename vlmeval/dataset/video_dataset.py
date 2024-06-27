from huggingface_hub import snapshot_download
from abc import abstractmethod
from json import JSONDecoder
from .config import dataset_md5_dict
from ..smp import *


FAIL_MSG = 'Failed to obtain answer via API.'


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


def prepare_MMBench_Video(dataset_name='MMBench-Video', repo_id='nebulae09/MMBench-Video'):

    def check_integrity(pth):
        data_file = osp.join(pth, f'{dataset_name}.tsv')
        if md5(data_file) != dataset_md5_dict['MMBench-Video']:
            return False
        data = load(data_file)
        for video_pth in data['video_path']:
            if not osp.exists(osp.join(pth, video_pth)):
                return False
        return True

    cache_path = get_cache_path(repo_id)
    if cache_path is not None and check_integrity(cache_path):
        dataset_path = cache_path
    else:
        dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
        unwrap_hf_pkl(dataset_path)
    data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

    return dict(data_file=data_file, root=osp.join(dataset_path, 'video'))


# Return a dictionary containing the data_file and root (video root) of the dataset
def prepare_dataset(dataset_name):
    if dataset_name == 'MMBench-Video':
        return prepare_MMBench_Video(dataset_name='MMBench-Video', repo_id='nebulae09/MMBench-Video')
    return None


class TSVDatasetVideo:

    TYPE = 'VIDEO'

    def __init__(self,
                 dataset='MMBench-Video',
                 pack=False):

        assert dataset in ['MMBench-Video']
        try:
            import decord
        except:
            warnings.warn('Please install decord via `pip install decord`.')
        ret = prepare_dataset(dataset)
        assert ret is not None
        lmu_root = LMUDataRoot()
        self.frame_root = osp.join(lmu_root, 'images', dataset)
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = 'frame-{}-of-{}.jpg'

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)

        assert 'question' in self.data and 'video' in self.data
        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos
        self.pack = pack

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

    def frame_paths(self, video, num_frames=8):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, video, num_frames=8):
        frame_paths = self.frame_paths(video, num_frames)
        flag = np.all([osp.exists(p) for p in frame_paths])
        if flag:
            return frame_paths
        vid_path = osp.join(self.data_root, video + '.mp4')
        vid = decord.VideoReader(vid_path)
        step_size = len(vid) / (num_frames + 1)
        indices = [int(i * step_size) for i in range(1, num_frames + 1)]
        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]
        for im, pth in zip(images, frame_paths):
            im.save(pth)
        return frame_paths

    @abstractmethod
    def build_prompt(self, idx, num_frames=8):
        pass


class MMBenchVideo(TSVDatasetVideo):

    SYS = 'You are an AI assistant responsible for answering questions about videos.'
    FRAMES_TMPL_PACK = """
You will be provided with {} separate frames uniformly sampled from a video, \
the frames are provided in chronological order of the video.
Please analyze these images and provide the answer / answers to the \
following question / questions about the video content.
If multiple questions are provided (with indices I1, I2, I3, ...), \
you should organize your answers in the following json format:
{{
    'I1': 'Answer to Question I1',
    'I2': 'Answer to Question I2',
    ...
}}
Otherwise, please directly reply with your response to the only question.
Even if the information in these separate frames is not enough to give an answer,
PLEASE GIVE A RESPONSE TO EACH OF THE QUESTIONS IN THE FORMAT DESCRIBED ABOVE.
"""

    FRAMES_TMPL_NOPACK = """
You will be provided with {} separate frames uniformly sampled from a video, \
the frames are provided in chronological order of the video.
Please analyze these images and provide the answer to the question about the video content.
Please directly reply with your response to the only question.
Even if the information in these separate frames is not enough to give an answer, \
PLEASE GIVE A RESPONSE TO EACH OF THE QUESTIONS IN THE FORMAT DESCRIBED ABOVE.
"""

    def __init__(self, dataset='MMBench-Video', pack=False):
        super().__init__(dataset=dataset, pack=pack)

    def build_prompt_pack(self, line, num_frames):
        if isinstance(line, int):
            assert line < len(self)
            video = self.videos[line]
        elif isinstance(line, pd.Series):
            video = line['video']
        elif isinstance(line, str):
            video = line

        frames = self.save_video_frames(video, num_frames)
        sub = self.data[self.data['video'] == video]
        sys_prompt = self.SYS + self.FRAMES_TMPL_PACK.format(num_frames)
        message = [dict(type='text', value=sys_prompt)]
        for im in frames:
            message.append(dict(type='image', value=im))
        nq = len(sub)
        prompt = 'Questions: \n{}\nAnswers: \n'
        qs = {int(sub.iloc[i]['index']): sub.iloc[i]['question'] for i in range(nq)}
        prompt = prompt.format(json.dumps(qs))
        message.append(dict(type='text', value=prompt))
        return message

    def build_prompt_nopack(self, line, num_frames):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames = self.save_video_frames(line['video'], num_frames)
        sys_prompt = self.SYS + self.FRAMES_TMPL_NOPACK.format(num_frames)
        message = [dict(type='text', value=sys_prompt)]
        for im in frames:
            message.append(dict(type='image', value=im))
        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        return message

    def build_prompt(self, line, num_frames):
        return self.build_prompt_pack(line, num_frames) if self.pack else self.build_prompt_nopack(line, num_frames)

    @staticmethod
    def extract_json_objects(text, decoder=JSONDecoder()):
        pos = 0
        while True:
            match = text.find('{', pos)
            if match == -1:
                break
            try:
                result, index = decoder.raw_decode(text[match:])
                yield result
                pos = match + index
            except ValueError:
                pos = match + 1

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
            jsons = list(MMBenchVideo.extract_json_objects(s))
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
                        key = MMBenchVideo.remove_side_quote(key)
                        val = MMBenchVideo.remove_side_quote(val)
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
