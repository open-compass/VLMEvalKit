from vlmeval.smp import *
from .image_base import ImageBaseDataset

from .utils import build_judge
import timeout_decorator
from latex2sympy2_extended import latex2sympy


def is_equal(asw: str, gt_asw: str) -> bool:
    if not isinstance(asw, str) != str or not isinstance(gt_asw, str):
        print('Warning: input is not string')
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    return False


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_lens_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}


def post_check(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    try:
        res = str(response)
        ans = str(ans)
    except ValueError:
        pass

    try:
        if is_equal(res, ans):
            return res if prefetch else True
        else:
            return False
    except Exception as err:
        logging.warning(f'{type(err)}: {err}')
        return False


def LENS_auxeval(model, line):
    try:
        prompt = build_lens_gpt4_prompt(line)
        log = ''
        retry = 5
        try:
            if post_check(line, prefetch=True):
                res = post_check(line, prefetch=True)
                return dict(log='Prefetch succeed', res=res)
        except Exception as e:
            logging.warning(f"Prefetch failed for index {line.get('index', 'unknown')}: {e}")

        for i in range(retry):
            try:
                prediction = line['prediction']
                res = model.generate(prompt, temperature=i * 0.5)

                if res is None:
                    raise ValueError("Model returned None")

                if FAIL_MSG in res:
                    log += f'Try {i}: output is {prediction}, failed to parse.\n'
                else:
                    log += 'Succeed'
                    return dict(log=log, res=res)

            except Exception as api_err:
                log += f'Try {i} Exception: {type(api_err)} - {api_err}\n'
                continue

        log += 'All 5 retries failed.\n'
        return dict(log=log, res='')

    except Exception as critical_err:
        logging.critical(f"Critical Error in LENS_auxeval: {critical_err}")
        return dict(log=f"Critical Error: {critical_err}", res='')


def LENS_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    from tqdm import tqdm
    for i in tqdm(range(lt)):
        item = data.iloc[i]
        cate = item['category']
        tot['Overall'] += 1
        tot[cate] += 1
        if item['log'] == 'Prefetch succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1
        is_correct = post_check(item, prefetch=False)
        if is_correct:
            hit['Overall'] += 1
            hit[cate] += 1

        # Print details
        # idx = item.get('index', 'N/A')
        # q = item.get('question', 'N/A')
        # gt = item.get('answer', 'N/A')
        # raw_pred = str(item.get('prediction', 'N/A')).replace('\n', ' ')
        # processed_res = item.get('res', 'N/A')
        # status = "Yes" if is_correct else "No"
        # msg = (
        #     f"\n--------------------------------------------------\n"
        #     f"Index: {idx}\n"
        #     f"Question: {q}\n"
        #     f"Correct answer: {gt}\n"
        #     f"Model original: {raw_pred[:100]}\n"
        #     f"Answer after processing: {processed_res}\n"
        #     f"Judgment result: {status}"
        # )
        # tqdm.write(msg)

    res = defaultdict(list)
    for k in tot.keys():
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)
    res = pd.DataFrame(res).sort_values('Subject', ignore_index=True)
    # res.columns = ['Subject', 'Total', 'Prefetch', 'Hit', 'Prefetch rate (%)', 'Accuracy rate (Acc %)']
    return res


class LENS(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'LENS-CN-QA':
        'https://huggingface.co/datasets/songlier/LENS/resolve/main/LENS-CN-QA.tsv',
        'LENS-CN-QA_MINI':
        'https://huggingface.co/datasets/songlier/LENS/resolve/main/LENS-CN-QA_MINI.tsv'
    }
    DATASET_MD5 = {
        'LENS-CN-QA': 'D382365A2C977543BEB890BAC240E731',
        'LENS-CN-QA_MINI':'4CEA1BDE46537DE2428C1D05A0B36094'
    }
    DEFAULT_JUDGE = 'gpt-4o-mini'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_{judge_name}_score.csv'

    def evaluate(self, eval_file, **judge_kwargs):
        model_name = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f'_{model_name}', 'tsv')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model_name}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'LENS evaluation requires a working OPENAI API\n'
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    LENS_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = LENS_acc(storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')
        dump(score, score_pth)
        return score
