import pandas as pd
from abc import abstractmethod
from PIL import Image
from vlmeval.smp import *


class GenBaseTextDataset:

    TYPE = 'T2I'
    MODALITY = 'TEXT'
    DATASET_URL = {}
    DATASET_MD5 = {}
    DEFAULT_JUDGE = 'chatgpt-0125'

    SUPPORT_GEN = True
    NUM_GENERATIONS = 1

    PRED_FORMAT = "{model_name}_{dataset_name}.tsv"
    RATING_FORMAT = "{model_name}_{dataset_name}_{judge_name}_rating.json"
    JUDGE_FORMAT = "{model_name}_{dataset_name}_{judge_name}.tsv"
    FAIL_MSG = 'Failed to obtain answer via API.'

    def __init__(self, dataset, **kwargs):
        self.dataset_name = dataset
        data = self.load_data(dataset)
        data['index'] = [str(x) for x in data['index']]
        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def prepare_tsv(self, url, file_md5=None):
        if osp.isfile(url) and osp.exists(url):
            return load(url)
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        print('data_path: ', data_path)
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
        return load(data_path)

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        assert isinstance(line, pd.Series) or isinstance(line, dict)
        mmqa_display(line)

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        url = self.DATASET_URL[dataset]
        file_md5 = self.DATASET_MD5[dataset]
        return self.prepare_tsv(url, file_md5)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        assert 'question' in line, line
        return [dict(type='text', value=str(line['question']))]

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @classmethod
    def is_response_err(cls, x):
        if x is None or pd.isna(x) or x == '':
            return True
        for pattern in [cls.FAIL_MSG, "思考过程过长", "Error code: "]:
            if pattern in str(x):
                return True
        return False

    @classmethod
    def report_response_err_rate(cls, model_name, dataset_name, root):
        pred_file = cls.PRED_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
        pred_file = osp.join(root, pred_file)
        assert osp.exists(pred_file), f'Pred file {pred_file} does not exist.'
        data = load(pred_file)
        err_rate = sum([cls.is_response_err(x) for x in data['prediction']]) / (len(data))
        err_rate = round(err_rate, 4)
        return err_rate

    @classmethod
    def report_judge_err_rate(cls, model_name, dataset_name, root, **kwargs):
        logger = get_logger('ImageBaseDataset')
        judge_file = None
        if cls.JUDGE_FORMAT is not None:
            if 'judge_name' not in cls.JUDGE_FORMAT:
                judge_file = cls.JUDGE_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
            else:
                judge_name = kwargs['judge_name'] if 'judge_name' in kwargs else cls.DEFAULT_JUDGE
                judge_file = cls.JUDGE_FORMAT.format(
                    model_name=model_name, dataset_name=dataset_name, judge_name=judge_name)
        if judge_file is None:
            return None
        judge_file = osp.join(root, judge_file)
        if not osp.exists(judge_file):
            logger.warning(f'Judge file {judge_file} does not exist. ')
            return None
        data = load(judge_file)
        if 'log' in data:
            err_rate = sum([cls.is_response_err(x) for x in data['log']]) / (len(data))
            err_rate = round(err_rate, 4)
            return err_rate
        else:
            logger.warning(f'Judge file {judge_file} does not contain `log` field. ')
            return None

    @classmethod
    def parse_df_rating(self, df, factor=100):
        if len(df) > 1:
            if 'split' in df and 'test' in set(df['split']):
                df = df[df['split'] == 'test']
            elif 'split' in df and 'validation' in set(df['split']):
                df = df[df['split'] == 'validation']

        assert len(df) == 1
        dic = {k: df.iloc[0][k] for k in df.columns}
        if 'split' in dic:
            dic.pop('split')
        for k in dic:
            if k.lower() == 'overall':
                dic[k.lower()] = dic.pop(k)
                break
        if 'overall' in dic and dic['overall'] > 1:
            return dic
        dic = {k: v * factor if not istype(v, str) else v for k, v in dic.items()}
        return dic

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        RATING_FORMAT = cls.RATING_FORMAT

        if 'judge_name' in RATING_FORMAT:
            judge_name = kwargs.get('judge_name', cls.DEFAULT_JUDGE)
            rating_file = RATING_FORMAT.format(model_name=model_name, dataset_name=dataset_name, judge_name=judge_name)
        else:
            rating_file = RATING_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
        rating_file = osp.join(root, rating_file)
        rating = load(rating_file)
        overall = None
        if isinstance(rating, dict):
            for k in rating.keys():
                if k.lower() == 'overall':
                    overall = rating[k]
                    rating[k.lower()] = rating.pop(k)
                    break
        elif isinstance(rating, pd.DataFrame):
            rating = cls.parse_df_rating(rating)
            overall = rating.get('overall', None)
        assert overall is not None, f'Overall rating not found in {rating_file}'
        res = {}
        res['overall'] = overall
        if verbose:
            res['rating'] = rating
        return res

    @classmethod
    def report(cls, model_name, dataset_name, root=None, verbose=False, **kwargs):
        import vlmeval
        if root is None:
            default_root = osp.join(vlmeval.__path__[0], '../outputs/')
            root = os.getenv('MMEVAL_ROOT', default_root)
        root = osp.join(root, model_name)
        # Default Name, if the name is not used, should override
        res = dict(model_name=model_name, dataset_name=dataset_name)
        res['response_err_rate'] = cls.report_response_err_rate(model_name, dataset_name, root)
        res['judge_err_rate'] = cls.report_judge_err_rate(model_name, dataset_name, root, **kwargs)
        res.update(cls.report_score(model_name, dataset_name, root, verbose=verbose, **kwargs))
        return res

    @staticmethod
    def is_path_image(s):
        def is_local_image(s):
            if osp.isfile(s) or osp.islink(s):
                try:
                    im = Image.open(s)
                    return isinstance(im, Image.Image)
                except:
                    return False
            else:
                return False

        def is_image_url(s):
            return s.startswith('http') and validators.url(s)

        return is_local_image(s) or is_image_url(s)

    @staticmethod
    def extract_single_image_from_response(response):
        if isinstance(response, str) and response.startswith('[') and response.endswith(']'):
            try:
                response = eval(response)
            except:
                raise ValueError(f'Failed to parse response {response} as json. ')
        assert isinstance(response, list), response

        def extract_last_image(instance):
            if isinstance(instance, str):
                return instance if GenBaseTextDataset.is_path_image(instance) else None
            elif isinstance(instance, list):
                images = [x if GenBaseTextDataset.is_path_image(x) else None for x in instance]
                for s in reversed(images):
                    if s is not None:
                        return s
                return None
        images = [extract_last_image(instance) for instance in response]
        for item in images:
            if item is not None:
                return item
        return None


class GenBaseImageDataset(GenBaseTextDataset):

    TYPE = 'TI2I'
    MODALITY = 'IMAGE'

    def __init__(self, dataset, **kwargs):
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, 'images', dataset)
        data = self.load_data(dataset)
        data['index'] = data['index'].astype(str)
        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'image' in data:
            images = [toliststr(x) for x in data['image']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        self.data = data
        self.post_build(dataset)

    def dump_image_atomic(self, im_b64, base_name):
        tgt_path = osp.join(self.img_root, base_name)
        dname = osp.dirname(tgt_path)
        if not osp.exists(dname):
            os.makedirs(dname, exist_ok=True)
        if not read_ok(tgt_path):
            decode_base64_to_image_file(im_b64, tgt_path)
        return tgt_path

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                if 'image_path' in line:
                    image_path = line['image_path']
                else:
                    index = line['index']
                    image_path = [f'{index}_{i}.png' for i in range(len(line['image']))]
                for img, im_name in zip(line['image'], image_path):
                    tgt_path.append(self.dump_image_atomic(img, im_name))

            elif isinstance(line['image'], str) and 'image_path' in line:
                assert isinstance(line['image_path'], str)
                tgt_path = [self.dump_image_atomic(line['image'], line['image_path'])]
            else:
                im_name = f"{line['index']}.png"
                tgt_path = [self.dump_image_atomic(line['image'], im_name)]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])
            read_ok_flag = [read_ok(x) for x in tgt_path]
            # Might be the Relative Path
            if not all(read_ok_flag):
                tgt_path_abs = [osp.join(self.img_root, x) for x in tgt_path]
                read_ok_flag = [read_ok(x) for x in tgt_path_abs]
                assert read_ok_flag, f"Field `image` is missing and we could not find {tgt_path} both as absolute or relative paths. "  # noqa
                tgt_path = tgt_path_abs

        return tgt_path

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs
