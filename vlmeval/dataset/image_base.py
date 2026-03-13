import pandas as pd
from abc import abstractmethod
from vlmeval.smp import *


def img_root_map(dataset):
    if 'MM_NIAH' in dataset:
        return 'MMNIAH'
    if 'CRPE' in dataset:
        return 'CRPE'
    if 'OCRVQA' in dataset:
        return 'OCRVQA'
    if 'COCO_VAL' == dataset:
        return 'COCO'
    if "QSpatial" in dataset:
        return "QSpatial"

    mmbench_root_map = {
        'MMBench_DEV_EN': 'MMBench', 'MMBench_TEST_EN': 'MMBench',
        'MMBench_DEV_CN': 'MMBench', 'MMBench_TEST_CN': 'MMBench',
        'MMBench_DEV_KO': 'MMBench',
        'MMBench': 'MMBench', 'MMBench_CN': 'MMBench',
        'MMBench_DEV_EN_V11': 'MMBench_V11', 'MMBench_TEST_EN_V11': 'MMBench_V11',
        'MMBench_DEV_CN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_V11',
        'MMBench_V11': 'MMBench', 'MMBench_CN_V11': 'MMBench',
    }
    if dataset in mmbench_root_map:
        return mmbench_root_map[dataset]
    return dataset


class ImageBaseDataset:

    MODALITY = 'IMAGE'
    DATASET_URL = {}
    DATASET_MD5 = {}
    DEFAULT_JUDGE = 'chatgpt-0125'

    PRED_FORMAT = "{model_name}_{dataset_name}.tsv"
    RATING_FORMAT = "{model_name}_{dataset_name}_acc.csv"
    JUDGE_FORMAT = "{model_name}_{dataset_name}_openai_result.tsv"
    FAIL_MSG = 'Failed to obtain answer via API.'

    def __init__(self, dataset='MMBench', skip_noimg=True):
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, 'images', img_root_map(dataset))

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]

        data['index'] = [str(x) for x in data['index']]

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= 64 and image_map[k] in image_map:
                    idx = image_map[k]
                    # assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        IS_GEN_BENCH = getattr(self, 'SUPPORT_GEN', False)

        if not IS_GEN_BENCH:
            if np.all([istype(x, int) for x in data['index']]):
                data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    @classmethod
    def pred_file_basename(cls, model_name, dataset_name):
        return cls.PRED_FORMAT.format(model_name=model_name, dataset_name=dataset_name)

    @classmethod
    def judge_file_basename(cls, model_name, dataset_name, **kwargs):
        info = {'model_name': model_name, 'dataset_name': dataset_name}
        if 'judge_name' in cls.JUDGE_FORMAT:
            info['judge_name'] = kwargs.get('judge_name', cls.DEFAULT_JUDGE)
        return cls.JUDGE_FORMAT.format(**info)

    @classmethod
    def rating_file_basename(cls, model_name, dataset_name, **kwargs):
        info = {'model_name': model_name, 'dataset_name': dataset_name}
        if 'judge_name' in cls.RATING_FORMAT:
            info['judge_name'] = kwargs.get('judge_name', cls.DEFAULT_JUDGE)
        return cls.RATING_FORMAT.format(**info)

    @staticmethod
    def extract_model_dataset(file_name):
        from vlmeval.dataset import SUPPORTED_DATASETS
        fname = osp.splitext(file_name)[0].split('/')[-1]
        parts = fname.split('_')
        for i in range(len(parts)):
            if '_'.join(parts[i:]) in SUPPORTED_DATASETS:
                return '_'.join(parts[:i]), '_'.join(parts[i:])
        return None, None

    @classmethod
    def get_judge_file_path(cls, eval_file, **kwargs):
        model_name, dataset_name = cls.extract_model_dataset(eval_file)
        dname = osp.dirname(eval_file)
        return osp.join(dname, cls.judge_file_basename(model_name, dataset_name, **kwargs))

    @classmethod
    def get_rating_file_path(cls, eval_file, **kwargs):
        model_name, dataset_name = cls.extract_model_dataset(eval_file)
        dname = osp.dirname(eval_file)
        return osp.join(dname, cls.rating_file_basename(model_name, dataset_name, **kwargs))

    def prepare_tsv(self, url, file_md5=None, img_zip=None, img_zip_md5=None):
        def is_local_path(path: str) -> bool:
            if osp.exists(path) and osp.isfile(path):
                return True
            return False

        if is_local_path(url):
            return load(url)

        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name_legacy = url.split('/')[-1]
        file_name = f"{self.dataset_name}.tsv"
        data_path_legacy = osp.join(data_root, file_name_legacy)
        data_path = osp.join(data_root, file_name)

        self.data_path = data_path
        if osp.exists(data_path):
            if file_md5 is None or md5(data_path) == file_md5:
                pass
            else:
                warnings.warn(f'The tsv file is in {data_root}, but the md5 does not match, will re-download')
                download_file(url, data_path)
                update_flag = True
        else:
            if osp.exists(data_path_legacy) and (file_md5 is None or md5(data_path_legacy) == file_md5):
                warnings.warn(
                    'Due to a modification in #1055, the local target file name has changed. '
                    f'We detected the tsv file with legacy name {data_path_legacy} exists and will do the rename. '
                )
                import shutil
                shutil.move(data_path_legacy, data_path)
            else:
                download_file(url, data_path)
                update_flag = True

        # === Unzip image ===
        # When images are pre-packaged in a zip file, we need to download and unzip it to LMUDataRoot/images
        # At that time, the tsv include 'image_path', which is the relative path to LMUDataRoot/images
        if img_zip is not None:
            data = load(data_path)
            assert 'image_path' in data and 'image' not in data
            assert img_zip.startswith('http'), f'img_zip must be a url, but got {img_zip}'
            img_root = osp.join(LMUDataRoot(), 'images')
            tmp_file = None
            ready_flag = True
            for pth in data['image_path']:
                pth = toliststr(pth)
                if not all([osp.exists(osp.join(img_root, p)) for p in pth]):
                    ready_flag = False
                    break
            if not ready_flag:
                tmp_file = osp.join('/tmp/', osp.basename(img_zip))
                download_file(img_zip, tmp_file, md5sum=img_zip_md5)
                import zipfile
                with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
                    zip_ref.extractall(img_root)
            new_pth = []
            for pth in data['image_path']:
                pth = toliststr(pth)
                abs_pth = [osp.join(img_root, p) for p in pth]
                assert all([osp.exists(p) for p in abs_pth]), f'image_path {pth} not found in {img_root}'
                new_pth.append(abs_pth[0] if len(abs_pth) == 1 else abs_pth)
            data['image_path'] = new_pth
            if tmp_file is not None and osp.exists(tmp_file):
                os.remove(tmp_file)
            return data

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                localize_tsv(data_path, local_path)
            data_path = local_path
        return load(data_path)

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
        url = self.DATASET_URL.get(dataset, None)
        if url is None or url == '':
            url = dataset + '.tsv'
        file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        return self.prepare_tsv(url, file_md5)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

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

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @classmethod
    def is_response_err(cls, x):
        if x is None or pd.isna(x) or x == '':
            return True
        for pattern in [cls.FAIL_MSG, "思考过程过长", "Error code: ", '"error":{"message":']:
            if pattern in str(x):
                return True
        return False

    @classmethod
    def report_response_err_rate(cls, model_name, dataset_name, root):
        pred_file = cls.pred_file_basename(model_name, dataset_name)
        pred_file = osp.join(root, pred_file)
        assert osp.exists(pred_file), f'Pred file {pred_file} does not exist.'
        data = load(pred_file)
        err_rate = sum([cls.is_response_err(x) for x in data['prediction']]) / (len(data))
        err_rate = round(err_rate, 4)
        return {'response_err_rate': err_rate}

    @classmethod
    def report_judge_err_rate(cls, model_name, dataset_name, root, **kwargs):
        logger = get_logger('ImageBaseDataset')
        ret = {'judge_err_rate': None}
        if cls.JUDGE_FORMAT is not None:
            judge_file = cls.judge_file_basename(model_name, dataset_name, **kwargs)
        else:
            return ret
        judge_file = osp.join(root, judge_file)
        if not osp.exists(judge_file):
            logger.warning(f'Judge file {judge_file} does not exist. ')
            return ret
        data = load(judge_file)
        if 'log' in data:
            err_rate = sum([cls.is_response_err(x) for x in data['log']]) / (len(data))
            err_rate = round(err_rate, 4)
            ret['judge_err_rate'] = err_rate
        else:
            logger.warning(f'Judge file {judge_file} does not contain `log` field. ')
        err_rate = sum([cls.is_response_err(x) for x in data['prediction']]) / (len(data))
        err_rate = round(err_rate, 4)
        ret['response_err_rate (judge)'] = err_rate
        return ret

    @classmethod
    def parse_df_rating(self, df, factor=100):
        if len(df) > 1:
            if 'split' in df:
                for sp_name in ['test', 'validation', 'circular_none']:
                    if sp_name in set(df['split']):
                        df = df[df['split'] == sp_name]
                        break
                assert len(df) == 1
            else:
                warnings.warn('Multiple lines in df, will use the first row by default. ')
                df = df.iloc[:1]

        assert len(df) == 1, df
        dic = dict(df.iloc[0])
        dic = {k: v for k, v in dic.items() if not pd.isna(v) and k != 'split'}
        dic = {k: float(v) if isinstance(v, float) else v for k, v in dic.items()}
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
        rating_file = cls.rating_file_basename(model_name, dataset_name, **kwargs)
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
        res.update(cls.report_response_err_rate(model_name, dataset_name, root))
        res.update(cls.report_judge_err_rate(model_name, dataset_name, root, **kwargs))
        res.update(cls.report_score(model_name, dataset_name, root, verbose=verbose, **kwargs))
        return res

    @abstractmethod
    def inference(self, model, sample):
        pass

    def with_inferencer(self):
        return getattr(self.inference, '__isabstractmethod__', True)
