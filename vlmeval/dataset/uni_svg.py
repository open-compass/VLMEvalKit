from .image_base import ImageBaseDataset
from ..smp import *


class UniSVG(ImageBaseDataset):

    TYPE = "VQA"

    DATASET_URL = {
        "UniSVG": "https://opencompass.openxlab.space/utils/VLMEval/UniSVG.tsv"
    }

    DATASET_MD5 = {
        "UniSVG": "c972880c8b6abe00276ff764b47b3513"
    }

    RATING_FORMAT = '{model_name}_{dataset_name}_score.json'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        task_type = line['category']
        question = line['question']

        msgs = []

        image_tasks = {
            "ISVGEN",
            "ISVGUN_usage",
            "ISVGUN_category",
            "ISVGUN_description",
            "ISVGUN_color"
        }

        if task_type in image_tasks:
            if self.meta_only:
                tgt_path = toliststr(line['image_path'])
            else:
                tgt_path = self.dump_image(line)

            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]

        msgs.append(dict(type='text', value=question))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.uni_svg import evaluate_uni_svg
        return evaluate_uni_svg(eval_file, dataset_name=self.dataset_name)

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        RATING_FORMAT = cls.RATING_FORMAT
        rating_file = RATING_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
        rating_file = osp.join(root, rating_file)
        rating = load(rating_file)
        overall = rating['final_score'] * 100
        res = {}
        res['overall'] = overall
        if verbose:
            res['rating'] = rating
        return res
