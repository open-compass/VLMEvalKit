from ..smp import *
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'


def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_score_ICE():
    example_1 = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent.
Only perform trivial simplifications
Examples:
    Expression 1: $2x+3$
    Expression 2: $3+2x$
Yes
    Expression 1: 70°
    Expression 2: 120
No
    Expression 1: 3/2
    Expression 2: 1.5
Yes
    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$
No
    Expression 1: d
    Expression 2: b
No
    Expression 1: 1.881 yr
    Expression 2: 1.92
No
    Expression 1: 44.1 mm
    Expression 2: 0.032 m
No
    Expression 1: 1,4,3,2
    Expression 2: 1,2,3,4
No
    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$
Yes
    Expression 1: 3245/5
    Expression 2: 649
No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)
    Expression 1: 2/(-3)
    Expression 2: -2/3
Yes
(trivial simplifications are allowed)
    Expression 1: 72 degrees
    Expression 2: 72
Yes
(give benefit of the doubt to units)
    Expression 1: 64
    Expression 2: 64 square feet
Yes
    Expression 1: R28
    Expression 2: 28
Yes
(ignore currency symbols if the numeric value is the same)
---
YOUR TASK
Respond with only "Yes" or "No" (without quotes). Do not include a rationale.
    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

    return example_1


def build_MMReason_gpt4_extract_prompt(line):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    prediction = str(line['prediction'])
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt


def build_MMReason_gpt4_score_prompt(line):
    full_prompt = get_gpt4_score_ICE()
    full_prompt = full_prompt % {"expression1": line['extract'], "expression2": line['answer']}

    return full_prompt


def post_check_score(line, prefetch=False):
    ans = str(line['answer']).strip()
    response = str(line['extract']).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False


def MMReason_auxeval_extract(model, line):
    prompt = build_MMReason_gpt4_extract_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_extract=log, extract=res)
    log += 'All 5 retries failed.\n'
    return dict(log_extract=log, extract='')


def MMReason_auxeval_score(model, line):
    prompt = build_MMReason_gpt4_score_prompt(line)
    log = ''
    retry = 5
    if post_check_score(line, prefetch=True):
        res = post_check_score(line, prefetch=True)
        return dict(log_score='Prefetch succeed', score=True)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res or res.strip() not in ['Yes', 'No']:
            log += f'Try {i}: output is {prediction}, res is {res}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_score=log, score=res == 'Yes' or res.strip() == 'Yes')
    log += 'All 5 retries failed.\n'
    return dict(log_score=log, score=False)


def get_acc_with_condition(res_pd, key, value):
    """
    Calculate the accuracy of predictions with a specific condition
    """
    total_pd = res_pd[res_pd[key] == value]
    correct_pd = total_pd[total_pd['score']]
    acc = '{:.4f}'.format(len(correct_pd) / len(total_pd)) if len(total_pd) > 0 else '0.0000'
    return len(correct_pd), len(total_pd), acc


def MMReason_acc(result_file):
    df = load(result_file)
    total = len(df)
    correct = sum(1 for _, row in df.iterrows() if row['score'])
    accuracy = round(correct / total, 2)
    scores = {'average': {'discipline': 'Overall', 'accuracy': accuracy, 'correct': correct, 'total': total}}

    discipline_total = dict()
    discipline_correct = dict()
    for _, row in df.iterrows():
        if row['discipline'] not in discipline_total:
            discipline_total[row['discipline']] = 1
        elif row['discipline'] in discipline_total:
            discipline_total[row['discipline']] += 1

        if row['discipline'] not in discipline_correct:
            discipline_correct[row['discipline']] = 0

        if row['score']:
            discipline_correct[row['discipline']] += 1

    for key in discipline_total.keys():
        scores[key] = {
            'discipline': key, 'accuracy': discipline_correct[key] / discipline_total[key],
            'correct': discipline_correct[key], 'total': discipline_total[key]
        }

    return pd.DataFrame.from_dict(scores, orient='index')


class MMReason(ImageBaseDataset):
    TYPE = 'VQA'
    mini_path = 'https://opencompass.openxlab.space/utils/VLMEval/MMReason_testmini.tsv'
    DATASET_URL = {
        'MMReason_testmini': mini_path,
    }
    DATASET_MD5 = {'MMReason_testmini': '630205345349ac51c9999d4fbfd1d630'}
    DEFAULT_JUDGE = 'gpt-4.1'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}_score.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_{judge_name}_acc.csv'

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        model = judge_kwargs['model']
        storage_extract = get_intermediate_file_path(eval_file, f'_{model}_extract', 'tsv')
        tmp_file_extract = get_intermediate_file_path(eval_file, f'_{model}_extract', 'pkl')
        storage_score = get_intermediate_file_path(eval_file, f'_{model}_score', 'tsv')
        tmp_file_score = get_intermediate_file_path(eval_file, f'_{model}_score', 'pkl')
        nproc = judge_kwargs.pop('nproc', 16)
        # stage1: extract the answer
        if not osp.exists(storage_extract):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MMReason evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_extract):
                ans = load(tmp_file_extract)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMReason_auxeval_extract,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_extract,
                )
                ans = load(tmp_file_extract)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_extract'] == v['log_extract'] and ans[k]['extract'] == v['extract']

            data['extract'] = [ans[idx]['extract'] for idx in data['index']]
            data['log_extract'] = [ans[idx]['log_extract'] for idx in data['index']]
            dump(data, storage_extract)

        # stage2: score the answer
        if not osp.exists(storage_score):
            data = load(storage_extract)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), ('MMReason evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_score):
                ans = load(tmp_file_score)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMReason_auxeval_score,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_score,
                )
                ans = load(tmp_file_score)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_score'] == v['log_score'] and ans[k]['score'] == v['score']

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log_score'] = [ans[idx]['log_score'] for idx in data['index']]
            dump(data, storage_score)

        score = MMReason_acc(storage_score)
        score_pth = storage_score.replace('_score.tsv', '_acc.csv')
        dump(score, score_pth)
        return score

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        judge_name = kwargs['judge_name'] if 'judge_name' in kwargs else cls.DEFAULT_JUDGE
        rating_file = cls.RATING_FORMAT.format(model_name=model_name, dataset_name=dataset_name, judge_name=judge_name)
        rating_file = osp.join(root, rating_file)
        df = load(rating_file)
        rating = {k: float(v) * 100 for k, v in zip(df['discipline'], df['accuracy'])}
        overall = rating['Overall']
        res = {}
        res['overall'] = overall
        if verbose:
            res['rating'] = rating
        return res
