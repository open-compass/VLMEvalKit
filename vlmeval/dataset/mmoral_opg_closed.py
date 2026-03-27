import os
import os.path as osp
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from ..smp import decode_base64_to_image_file, dump, load, read_ok
from .image_base import ImageBaseDataset


class MMOralBase(ImageBaseDataset):
    """Shared image-dumping logic for MMOral-OPG benchmarks."""

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root, f"{line['index']}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z


class MMOral_OPG_CLOSED(MMOralBase):
    """Closed-ended MMOral-OPG benchmark (4-option MCQ)."""

    TYPE = 'MCQ'

    DATASET_URL = {
        'MMOral_OPG_CLOSED': 'https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench/resolve/main/MMOral-OPG-Bench-Closed-Ended.tsv'  # noqa: E501
    }

    DATASET_MD5 = {
        'MMOral_OPG_CLOSED': 'b13cff13ffce25225d5de0efed8e53fa'
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = line['question']

        options_prompt = 'Options:\n'
        for i in [['A', '1'], ['B', '2'], ['C', '3'], ['D', '4']]:
            option_value = str(line[f'option{i[1]}'])
            options_prompt += f"{i[0]}. {option_value}\n"

        prompt = (
            f'Question: {question}\n'
            + options_prompt
            + 'Please answer the above multiple-choice question by selecting the single correct option (A, B, C, or D). '  # noqa: E501
            + 'If the provided information is insufficient to determine a clear answer, please choose the most likely '  # noqa: E501
            + 'correct option based on the available data and your judgment.'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Direct accuracy evaluation on single-choice predictions."""
        from .utils.mmoral_opg import get_single_choice_prediction

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        detail_result_file = eval_file.replace(f'.{suffix}', '_detailed_acc.csv')

        if not osp.exists(result_file) or not osp.exists(detail_result_file):
            data = load(eval_file)
            assert 'answer' in data and 'prediction' in data
            data['prediction'] = [str(x) for x in data['prediction']]
            data['answer'] = [str(x) for x in data['answer']]

            tot = defaultdict(lambda: 0)
            score = defaultdict(lambda: 0)

            main_category_list = ['Teeth', 'Patho', 'HisT', 'Jaw', 'SumRec']
            categories = set()
            subcategories = set()

            for _, line in data.iterrows():
                category = line.get('category', 'unknown')
                categories.add(category)
                subcategory = category.replace(',', '_')
                subcategories.add(subcategory)

                for main_cat in main_category_list:
                    if main_cat in category:
                        tot[main_cat] += 1

                tot[category] += 1
                tot[subcategory] += 1
                tot['Overall'] += 1

            for _, line in tqdm(data.iterrows()):
                category = line.get('category', 'unknown')
                subcategory = category.replace(',', '_')

                index2ans = {
                    'A': line['option1'],
                    'B': line['option2'],
                    'C': line['option3'],
                    'D': line['option4'],
                }

                fact_option = get_single_choice_prediction(
                    line['prediction'], ['A', 'B', 'C', 'D'], index2ans
                )

                if fact_option == line['answer']:
                    for main_cat in main_category_list:
                        if main_cat in category:
                            score[main_cat] += 1

                    score[category] += 1
                    score[subcategory] += 1
                    score['Overall'] += 1

            main_result = defaultdict(list)
            main_category_list.append('Overall')
            for cat in main_category_list:
                main_result['Category'].append(cat)
                main_result['tot'].append(tot[cat])
                main_result['acc'].append(
                    score[cat] / tot[cat] * 100 if tot[cat] > 0 else 0
                )

            detailed_categories = list(categories) + ['Overall']
            detailed_result = defaultdict(list)
            for cat in detailed_categories:
                detailed_result['Category'].append(cat)
                detailed_result['tot'].append(tot[cat])
                detailed_result['acc'].append(
                    score[cat] / tot[cat] * 100 if tot[cat] > 0 else 0
                )

            main_df = pd.DataFrame(main_result)
            detailed_df = pd.DataFrame(detailed_result)

            main_df = main_df.sort_values('Category')
            detailed_df = detailed_df.sort_values('Category')

            dump(main_df, result_file)
            dump(detailed_df, detail_result_file)

        result = pd.read_csv(result_file)
        return result
