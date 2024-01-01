import pandas as pd
import hashlib
from ..smp import *
from .dataset_config import dataset_URLs, dataset_md5_dict, img_root_map, DATASET_TYPE
from .custom_prompt import CustomPrompt

def isliststr(s):
    return (s[0] == '[') and (s[-1] == ']')

def check_md5(data_path, dataset):
    try:
        with open(data_path, 'rb') as f:
            hash = hashlib.new('md5')
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
        if str(hash.hexdigest()) == dataset_md5_dict[dataset]:
            return True
        else:
            warnings.warn('this data file is incomplete, so it needs to be downloaded again.')
            return False
    except:
        return False
    
def split_MMMU(struct):
    assert 'image' in struct and 'text' in struct
    text, images = struct['text'], struct['image']
    text_segs = text.split('<image ')
    segs = [text_segs[0]]
    for i, seg in enumerate(text_segs):
        if i == 0:
            continue
        assert istype(seg[0], int) and seg[1] == '>'
        image_idx = int(seg[0]) - 1
        segs.append(images[image_idx])
        segs.append(seg[2:])
    return segs

class TSVDataset(CustomPrompt):
    
    def __init__(self, dataset='MMBench', img_root=None, skip_noimg=True):

        self.data_root = LMUDataRoot()
        assert osp.exists(self.data_root)

        self.dataset = dataset
        self.dataset_type = DATASET_TYPE(dataset)

        url = dataset_URLs[dataset]
        file_name = url.split('/')[-1]
        data_path = osp.join(self.data_root, file_name)

        if osp.exists(data_path) and md5(data_path) == dataset_md5_dict[dataset]:
            pass
        else:
            warnings.warn("The dataset tsv is not downloaded")
            download_file(url, data_path)

        data = load(data_path)
        self.skip_noimg = skip_noimg
        if skip_noimg:
            data = data[~pd.isna(data['image'])]

        # Prompt for Captioning
        if listinstr(['COCO'], dataset):
            data['question'] = ['Please describe this image in general. Directly provide the description, do not include prefix like "This image depicts". '] * len(data)

        data['index'] = [str(x) for x in data['index']]
        data['image'] = [str(x) for x in data['image']]
        
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]
    
        data['image'] = [
            eval(image_map[k]) if isliststr(image_map[k]) else image_map[k] 
            for k in data['index']
        ]
        if 'image_path' in data:
            data['image_path'] = [
                eval(pths) if isliststr(pths) else pths for pths in data['image_path']
            ]
        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]
            
        self.data = data

        img_root = img_root if img_root is not None else osp.join('images', img_root_map[dataset])
        os.makedirs(img_root, exist_ok=True)
        self.img_root = img_root

    def __len__(self):
        return len(self.data)

    def build_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

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
        
        return dict(image=tgt_path, text=prompt)
    
    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        mmqa_display(line)
    
