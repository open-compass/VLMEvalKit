import json
import os
import re
import shutil

from tqdm import tqdm

from ..registry.registry import DATASET_REGISTRY

try:
    from ..utils.ocr_utils import get_text_for_block
except Exception:
    def get_text_for_block(*args, **kwargs):
        return ''
from ..utils.data_preprocess import (clean_string, normalized_formula, normalized_table,
                                     textblock2unicode)


@DATASET_REGISTRY.register("recogition_text_dataset")
class RecognitionTextDataset():
    # Evaluate at text block granularity, without considering one-to-one bbox matching
    def __init__(self, cfg_task):
        gt_file = cfg_task['dataset']['ground_truth']['data_path']
        pred_folder = cfg_task['dataset']['prediction']['data_path']
        self.samples = self.load_data(gt_file, pred_folder)

    def load_data(self, gt_file, pred_folder):
        samples = []
        with open(gt_file, 'r') as f:
            gts = json.load(f)

        for gt in gts:
            img_name = os.path.basename(gt['image_path'])
            gt_text = gt['text']
            pred_file = os.path.join(pred_folder, img_name[:-4] + '.json')
            if not os.path.exists(pred_file):
                print(f'Cannot find pred for {img_name}')
                continue
            else:
                with open(pred_file, 'r') as f:
                    pred_spans = json.load(f)
                pred_text = get_text_for_block(gt, pred_spans)
            samples.append({
                "gt": gt_text,
                'pred': pred_text,
                'img_id': img_name
            })
        return samples


@DATASET_REGISTRY.register("omnidocbench_single_module_dataset")
class OmiDocBenchSingleModuleDataset():
    # Evaluate at text block granularity, without considering one-to-one bbox matching
    def __init__(self, cfg_task):
        gt_key = cfg_task['dataset']['ground_truth']['data_key']
        pred_file = cfg_task['dataset']['ground_truth']['data_path']
        pred_key = cfg_task['dataset']['prediction']['data_key']

        self.category_filter = cfg_task['dataset']['ground_truth'].get('category_filter', [])
        self.category_type = cfg_task['dataset'].get('category_type')
        self.samples = self.load_data(pred_file, pred_key, gt_key)

    def load_data(self, pred_file, pred_key, gt_key):
        samples = []
        with open(pred_file, 'r') as f:
            preds = json.load(f)
        count = 0
        for pred in preds:
            img_name = os.path.basename(pred['page_info']['image_path'])
            for i, ann in enumerate(pred['layout_dets']):
                if not ann.get(gt_key):
                    continue
                if self.category_filter:
                    if ann['category_type'] not in self.category_filter:
                        continue
                if not ann.get(pred_key):
                    # print(f'Cannot find pred for {img_name}. ann is {ann}')
                    # pdb.set_trace()
                    count += 1
                    continue
                else:
                    gt_text = ann[gt_key]
                    norm_gt = gt_text
                    pred_text = ann[pred_key]
                    norm_pred = pred_text
                    if self.category_type:
                        if self.category_type == 'text':
                            norm_gt = clean_string(textblock2unicode(ann[gt_key]))
                            norm_pred = clean_string(textblock2unicode(ann[pred_key]))
                        elif self.category_type == 'formula':
                            norm_gt = normalized_formula(ann[gt_key])
                            norm_pred = normalized_formula(ann[pred_key])
                        elif self.category_type == 'table':
                            norm_gt = normalized_table(ann[gt_key], gt_key)
                            norm_pred = normalized_table(ann[pred_key], gt_key)
                        else:
                            raise ValueError(f'Invalid category type: {self.category_type}')

                samples.append({
                    "gt": gt_text,
                    "norm_gt": norm_gt,
                    "gt_attribute": [ann['attribute']],
                    'pred': pred_text,
                    "norm_pred": norm_pred,
                    'img_id': img_name
                })
        print(f'Cannot find pred for {count} samples.')

        return samples


@DATASET_REGISTRY.register("recogition_formula_dataset")
class RecognitionFormulaDataset():
    def __init__(self, cfg_task):
        gt_file = cfg_task['dataset']['ground_truth']['data_path']
        pred_file = cfg_task['dataset']['prediction']['data_path']

        self.samples = self.load_data(gt_file, pred_file)

    def load_data(self, gt_file, pred_file):
        """
        Load a list of image paths and their corresponding formulas.
        The function skips empty lines and lines without corresponding images.

        Args:
            image_path (str): The path to the directory containing the image files.
            math_file (str): The path to the text file containing the formulas.

        Returns:
            list, list: A list of image paths and a list of corresponding formula
        """

        with open(gt_file, 'r') as f:
            math_gts = [line.strip() for line in f.readlines()]

        with open(pred_file, 'r') as f:
            math_preds = [line.strip() for line in f.readlines()]

        if len(math_preds) != len(math_gts):
            raise ValueError("The number of prediction does not match the number of ground truth.")

        norm_gts = [self.normalize_text(gt) for gt in math_gts]   # Formula normalization
        norm_preds = [self.normalize_text(pred) for pred in math_preds]

        samples = []
        img_id = 0
        for gt, pred in zip(norm_gts, norm_preds):
            samples.append({
                'gt': gt,
                'pred': pred,
                'img_id': img_id
            })
            img_id += 1

        return samples

    def normalize_text(self, text):
        """Remove unnecessary whitespace from LaTeX code."""
        text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
        letter = '[a-zA-Z]'
        noletter = '[\\W_^\\d]'
        names = [x[0].replace(' ', '') for x in re.findall(text_reg, text)]
        text = re.sub(text_reg, lambda match: str(names.pop(0)), text)
        news = text
        while True:
            text = news
            news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', text)
            news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
            news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
            if news == text:
                break
        return text

    def __getitem__(self, idx):
        return self.samples[idx]


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

        for img in tqdm(ref_keys, total=len(ref_keys), ncols=140,
                        ascii=True, desc='Normalizing data'):
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
            # print('p:', p)
            # print('r:', r)
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
