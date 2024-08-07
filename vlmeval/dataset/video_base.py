from abc import abstractmethod
from ..smp import *


def video_root_map(dataset):
    return dataset


class VideoBaseDataset:
    MODALITY = 'VIDEO'

    def __init__(self, dataset='MMBench-Video', pack=False, skip_novideo=True):
        try:
            import decord
        except:
            warnings.warn('Please install decord via `pip install decord`.')

        # the init for previous two video dataset
        if dataset in ['MMBench-Video', 'Video-MME']:
            self.dataset_name = dataset
            ret = self.prepare_dataset(dataset)
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

        # dataset init without prepare_dataset, just like image_base
        else:
            lmu_root = LMUDataRoot()
            # You can override this variable to save image files to a different directory
            self.dataset_name = dataset
            self.frame_root = osp.join(lmu_root, 'images', dataset)
            self.frame_tmpl = 'frame-{}-of-{}.jpg'
            data, data_root = self.load_data(dataset)
            self.data_root = data_root
            self.meta_only = True
            self.skip_novideo = skip_novideo
            if skip_novideo and 'video' in data:
                data = data[~pd.isna(data['video'])]

            data['index'] = [str(x) for x in data['index']]
            data['index'] = [str(x) for x in data['index']]

            if 'video' in data:
                self.meta_only = False

            if 'video_path' in data:
                paths = [toliststr(x) for x in data['video_path']]
                data['video_path'] = [x[0] if len(x) == 1 else x for x in paths]

            if np.all([istype(x, int) for x in data['index']]):
                data['index'] = [int(x) for x in data['index']]

            self.data = data
            self.post_build(dataset)

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

    def load_data(self, dataset):
        url = self.DATASET_URL[dataset]
        file_md5 = self.DATASET_MD5[dataset]
        return self.prepare_tsv(url, file_md5)

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
        return load(data_path), data_root

    def post_build(self, dataset):
        pass

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
            if not osp.exists(pth):
                im.save(pth)
        return frame_paths

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return ['MMBench-Video', 'Video-MME'] + list(cls.DATASET_URL)

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @abstractmethod
    def build_prompt(self, idx, num_frames=8):
        pass

    @abstractmethod
    def prepare_dataset(self, dataset):
        # The prepare_dataset function should return a dictionary containing:
        # `root` (directory that containing video files)
        # `data_file` (the TSV dataset file)
        pass
