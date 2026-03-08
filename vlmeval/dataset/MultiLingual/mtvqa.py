from ..image_base import ImageBaseDataset
from ...smp import *
from ..utils import LLM_VERIFIER, build_judge


JUDGE_PROMPT = """\
You are an AI assistant to help me to judge if a model prediction is correct given the question and the groundtruth answer. \
The question may be in various different languages rather than English. \
You will be provided with a question, a groundtruth answer, and a model prediction. \
Your task is to judge if the model prediction is correct. \
If the model prediction is an exact match of the groundtruth answer, output Yes. \
Otherwise, output No.

Example 1:
Question: ما هو الرقم التسلسلي للزجاجة؟
Answer: 2 007474 277103
Prediction: **2007474277103**
Your output: Yes

Example 2:
Question: Dove si terrà l"evento?
Answer: Bergamo
Prediction: **Bergamo** (Fiera Bergamo)
Your output: Yes

Your Task:
Question: {question}\nAnswer: {answer}\nPrediction: {prediction}\nYour output: \
"""  # noqa: E501


def YorN_verifier(x):
    x = x.lower()
    if 'yes' in x and 'no' not in x:
        return True
    elif 'no' in x and 'yes' not in x:
        return False
    else:
        return None


def MTVQA_auxeval(model, line):
    if line['answer'] in line['prediction']:
        return True
    verifier = LLM_VERIFIER(model, JUDGE_PROMPT, YorN_verifier)
    return verifier.verify(line)


class MTVQADataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MTVQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/MTVQA_TEST.tsv'
    }
    DATASET_MD5 = {'MTVQA_TEST': 'd87c17dbab934b7cd89c0a3c1c5657f4'}
    # deepseek-v3-0324
    DEFAULT_JUDGE = 'deepseek'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_{judge_name}_score.json'

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data and 'category' in data
        data['prediction'] = [str(x).strip().lower().replace('.', '') for x in data['prediction']]
        data['answer'] = [str(x).strip().lower().replace('.', '') for x in data['answer']]

        model_name = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f'_{model_name}', 'tsv')
        score_file = get_intermediate_file_path(eval_file, f'_{model_name}_score', 'json')
        nproc = judge_kwargs.pop('nproc', 16)

        if osp.exists(storage):
            data = load(storage)
        else:
            lines = [row for _, row in data.iterrows()]
            model = build_judge(**judge_kwargs)
            tups = [dict(model=model, line=line) for line in lines]
            hit = track_progress_rich(
                MTVQA_auxeval,
                tups,
                nproc=nproc,
                desc="MT-VQA evaluation"
            )
            data['hit'] = hit
            dump(data, storage)

        cates = set(data['category'])
        rating = {}
        for cate in cates:
            rating[cate] = np.mean([hit == 1 for hit, c in zip(data['hit'], data['category']) if c == cate])  # noqa: E501
        rating['Overall'] = np.mean([hit == 1 for hit in data['hit']])
        rating = {k: v * 100 for k, v in rating.items()}

        dump(rating, score_file)
        return rating

    # MT-VQA adopts a custom prompt
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                item[
                    'value'] += '\nAnswer the question using a word or phrase in the language of the question.'
        return msgs

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        rating_file = cls.RATING_FORMAT.format(
            model_name=model_name, dataset_name=dataset_name, judge_name=cls.DEFAULT_JUDGE)
        rating_file = osp.join(root, rating_file)
        rating = load(rating_file)
        res = {'overall': rating['Overall']}
        if verbose:
            res['rating'] = rating
        return res
