import pandas as pd
from vlmeval.smp import *

LAST_MODIFIED = 231126000000

dataset_URLs = {
    'MMBench_DEV_EN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv", 
    'MMBench_TEST_EN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv", 
    'MMBench_DEV_CN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv", 
    'MMBench_TEST_CN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv", 
    "MMBench": "https://opencompass.openxlab.space/utils/VLMEval/MMBench.tsv",  # Link Invalid, Internal Only
    "MMBench_CN": "https://opencompass.openxlab.space/utils/VLMEval/MMBench_CN.tsv",    # Link Invalid, Internal Only
    'CCBench': "https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv", 
    'MME': "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv", 
    'SEEDBench_IMG': "https://opencompass.openxlab.space/utils/VLMEval/SEEDBench_IMG.tsv", 
    "CORE_MM": "https://opencompass.openxlab.space/utils/VLMEval/CORE_MM.tsv"
}

img_root_map = {
    'MMBench_DEV_EN': "MMBench", 
    'MMBench_TEST_EN': "MMBench", 
    'MMBench_DEV_CN': "MMBench", 
    'MMBench_TEST_CN': "MMBench", 
    "MMBench": "MMBench",  # Link Invalid, Internal Only
    "MMBench_CN": "MMBench",    # Link Invalid, Internal Only
    'CCBench': "CCBench", 
    'MME': "MME", 
    "CORE_MM": "CORE_MM", 
    'SEEDBench_IMG': "SEEDBench_IMG", 
}

assert set(dataset_URLs) == set(img_root_map)

def DATASET_TYPE(dataset):
    if 'mmbench' in dataset.lower() or 'seedbench' in dataset.lower() or 'ccbench' in dataset.lower():
        return 'multi-choice'
    elif 'MME' in dataset:
        return 'Y/N'
    return 'QA'

def isliststr(s):
    return (s[0] == '[') and (s[-1] == ']')

class TSVDataset:
    
    def __init__(self, dataset='MMBench', img_root=None):

        self.data_root = LMUDataRoot()
        assert osp.exists(self.data_root)

        self.dataset = dataset
        self.dataset_type = DATASET_TYPE(dataset)
        
        url = dataset_URLs[dataset]
        file_name = url.split('/')[-1]
        data_path = osp.join(self.data_root, file_name)

        if osp.exists(data_path) and int(last_modified(data_path)) > LAST_MODIFIED:
            pass
        else:
            warnings.warn("The dataset tsv is not downloaded")
            download_file(url, data_path)

        data = load(data_path)

        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if k >= 1000000 and listinstr(['MMBench', 'CCBench'], self.dataset):
                image_map[k] = image_map[k % 1000000]
            elif k % 2 == 1 and self.dataset in ['MME']:
                image_map[k] = image_map[k - 1]
    
        data['image'] = [
            eval(image_map[k]) if isliststr(image_map[k]) else image_map[k] 
            for k in data['index']
        ]
        if 'image_path' in data:
            data['image_path'] = [
                eval(pths) if isliststr(pths) else pths for pths in data['image_path']
            ]

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

        if isinstance(line['image'], list):
            tgt_path = []
            for img, im_name in zip(line['image'], line['image_path']):
                path = osp.join(self.img_root, im_name)
                if not osp.exists(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not osp.exists(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
       
        prompt = line['question']

        if listinstr(['MMBench', 'CCBench', 'SEEDBench'], dataset):
            question = line['question']
            option_candidate = ['A', 'B', 'C', 'D', 'E']
            options = {
                cand: line[cand]
                for cand in option_candidate
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
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        return dict(image=tgt_path, text=prompt)
    
    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        mmqa_display(line)
    