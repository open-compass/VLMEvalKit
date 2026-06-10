import json
import os
import shutil

from tqdm import tqdm

from src.core.preprocess import normalized_table
from src.core.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("recogition_table_dataset")
class RecognitionTableDataset():
    def __init__(self, cfg_task):
        gt_file = cfg_task['dataset']['ground_truth']['data_path']
        pred_file = cfg_task['dataset']['prediction']['data_path']
        self.pred_table_format = cfg_task['dataset']['prediction'].get('table_format', 'html')

        references, predictions = self.load_data(gt_file), self.load_data(pred_file)
        self.samples = self.normalize_data(references, predictions)

    def normalize_data(self, references, predictions):
        if self.pred_table_format == 'latex2html':
            os.makedirs('./temp', exist_ok=True)

        samples = []
        ref_keys = list(references.keys())

        for img in tqdm(ref_keys, total=len(ref_keys), ncols=140, ascii=True, desc='Normalizing data'):
            if self.pred_table_format == 'html':
                r = references[img]['html']
                p = predictions[img]['html']
            elif self.pred_table_format == 'latex':
                r = references[img]['latex']
                p = predictions[img]['latex']
            else:
                raise ValueError(f'Invalid table format: {self.pred_table_format}')

            img_id = references[img]["page_image_name"]
            p = normalized_table(p, self.pred_table_format)
            r = normalized_table(r, self.pred_table_format)
            samples.append({
                'gt': p,
                'pred': r,
                'img_id': img_id,
                'gt_attribute': [references[img]['attribute']],
            })

        if self.pred_table_format == 'latex2html':
            shutil.rmtree('./temp')
        return samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def load_data(self, data_path):
        result_dict = {}
        with open(data_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            result_dict[sample["image_path"]] = sample

        return result_dict
