import pandas as pd
import hashlib
from ..smp import *
from .dataset_config import dataset_URLs, dataset_md5_dict, DATASET_TYPE
from .custom_prompt import CustomPrompt


def check_md5(data_path, dataset):
    if dataset not in dataset_md5_dict:
        warnings.warn(f'We do not have an md5 record for dataset {dataset}, skip the md5 check. ')
        return True
    assert osp.exists(data_path)
    with open(data_path, 'rb') as f:
        hash = hashlib.new('md5')
        for chunk in iter(lambda: f.read(2**20), b''):
            hash.update(chunk)
    if str(hash.hexdigest()) == dataset_md5_dict[dataset]:
        return True
    else:
        warnings.warn('this data file is incomplete, so it needs to be downloaded again.')
        return False


def split_MMMU(msgs):
    text, images = None, []
    for s in msgs:
        if s['type'] == 'image':
            images.append(s['value'])
        elif s['type'] == 'text':
            assert text is None
            text = s['value']
    text_segs = text.split('<image ')
    segs = [dict(type='text', value=text_segs[0])]
    for i, seg in enumerate(text_segs):
        if i == 0:
            continue
        assert istype(seg[0], int) and seg[1] == '>'
        image_idx = int(seg[0]) - 1
        segs.append(dict(type='image', value=images[image_idx]))
        segs.append(dict(type='text', value=seg[2:]))
    return segs


def prep_tsv(dataset):
    data_root = LMUDataRoot()
    assert osp.exists(data_root)
    update_flag = False

    if dataset in dataset_URLs:
        url = dataset_URLs[dataset]
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)

        if osp.exists(data_path) and check_md5(data_path, dataset):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
            update_flag = True
    else:
        data_path = osp.join(data_root, dataset + '.tsv')
        assert osp.exists(data_path)

    if file_size(data_path, 'GB') > 1:
        local_path = data_path.replace('.tsv', '_local.tsv')
        if not osp.exists(local_path) or update_flag or os.environ.get('FORCE_LOCAL', None):
            from ..tools import LOCALIZE
            LOCALIZE(data_path, local_path)
        return local_path
    else:
        return data_path


class TSVDataset(CustomPrompt):

    def __init__(self, dataset='MMBench', skip_noimg=True):

        self.data_root = LMUDataRoot()
        self.dataset = dataset
        self.dataset_type = DATASET_TYPE(dataset)
        self.data_path = prep_tsv(dataset)
        data = load(self.data_path)

        self.skip_noimg = skip_noimg
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]

        # Prompt for Captioning
        if listinstr(['COCO'], dataset):
            data['question'] = [(
                'Please describe this image in general. Directly provide the description, '
                'do not include prefix like "This image depicts". '
            )] * len(data)

        data['index'] = [str(x) for x in data['index']]

        self.meta_only = True
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]

            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data

    def __len__(self):
        return len(self.data)

    def build_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line, dataset)

        prompt = line['question']
        if DATASET_TYPE(dataset) == 'multi-choice':
            question = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                prompt += 'Please select the correct answer from the options above. \n'
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['ocrvqa', 'textvqa', 'chartqa', 'docvqa'], dataset.lower()):
                prompt += '\nAnswer the question using a single word or phrase.\n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        mmqa_display(line)
