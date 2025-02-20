from vlmeval import *
from .image_base import ImageBaseDataset
from .utils import build_judge
from .utils.multiple_choice import report_acc, eval_vanilla, eval_circular_group
from ..utils import track_progress_rich

EVAL_PROMPT = """
You are an AI assistant who will help me to extract the answer of a visual-language question from a model response and determine if the answer is correct.
You will be provided with: 1. the question (text only); 2. the model response; 3. the ground-truth.
By default, you should determine if the answer is correct based on if the meaning of the model response align with the ground-truth.
{requirement}

[Begin Question]\n{question}\n[End Question]
[Begin Response]\n{response}\n[End Response]
[Begin Ground-Truth]\n{ground_truth}\n[End Ground-Truth]

You should output your judgement in the following format ("X" should be "yes" or "no"):

CORRECT: [[X]]
REASON: [[Your reason]]
"""


def ShortQA_prompt(line):
    answer = line['answer']
    if answer[0] == '[' and answer[-1] == ']':
        answer = eval(answer)
    else:
        answer = [answer]
    multi_ans = False
    if len(answer) > 1:
        if 'multi_ans' in line and not pd.isna(line['multi_ans']):
            multi_ans = line['multi_ans']
        else:
            multi_ans = True
    requirement = ''
    if len(answer) > 1:
        if multi_ans:
            requirement = "The provided ground-truth is a list. The answer is correct if the model response contains and only contains all the ground-truths (no other answer included)."
        else:
            requirement = 'The provided ground-truth is a list. If the model response matches any of the ground-truths, the answer is correct. '
    return EVAL_PROMPT.format(
        question=line['question'],
        response=line['prediction'],
        ground_truth=answer,
        requirement=requirement)


def ShortQA_auxeval(model, line):
    def proc_str(s):
        chs = set(s)
        chs = [x for x in chs if x not in string.ascii_letters + ': ']
        for ch in chs:
            s = s.replace(ch, ' ')
        return s

    def extraction(resp):
        lines = resp.split('\n')
        correct, reason = None, None
        for l in lines:
            l = l.upper()
            l = proc_str(l).strip()
            if l.startswith('CORRECT:'):
                l = l[8:].strip()
                if l in ['YES', 'NO']:
                    correct = 1 if l == 'YES' else 0
                    break
        for word in ['REASON:', 'reason:', 'Reason:']:
            if word in resp:
                reason = resp.split(word)[1].strip()
                break
        return (correct, reason)
        
    prompt = ShortQA_prompt(line)
    retry = 3
    for i in range(retry):
        output = model.generate(prompt, temperature=0.5 * i)
        ans = extraction(output)
        # print(output, ans)
        if ans[0] in [0, 1]:
            return dict(hit=ans[0], log=ans[1])

    return dict(hit=0, log='Fail to Judge')


def Comprehensive_auxeval(model, data):
    def valid(record, key_name):
        return key_name in record and (not pd.isna(record[key_name])) and record[key_name] != ''
    
    if isinstance(data, pd.DataFrame) and len(data) > 1:
        # Should Adopt CircularEval
        assert valid(data.iloc[0], 'A')
        data['GT'] = data['answer']
        return eval_circular_group(model, data)
    else:
        item = data.iloc[0] if isinstance(data, pd.DataFrame) else data
        if valid(item, 'A') and len(item['answer']) == 1:
            item['GT'] = item['answer']
            return eval_vanilla(model, item)
        else:
            return ShortQA_auxeval(model, item)
        

class ImageShortQADataset(ImageBaseDataset):
    TYPE = 'Short'

    DATASET_URL = {
        'LiveMMBench_Infographic': '',
        'LiveMMBench_Perception': '',
        'LiveMMBench_Reasoning': '',
        'LiveMMBench_Reasoning_circular': '',
    }

    DATASET_MD5 = {}

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\nPlease directly provide a short answer to the question. '
        return msgs

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        storage = eval_file.replace('.xlsx', '_judge.xlsx')
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        
        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)
            
            model = judge_kwargs.get('model', 'gpt-4o-mini')
            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(model=model, **judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                model = None
                warnings.warn('OPENAI_API_KEY is not working properly, will use exact matching for evaluation')

            if model is not None:
                if 'g_index' not in data:
                    lines = [data.iloc[i] for i in range(len(data))]
                    indices = [x['index'] for x in lines if x['index'] not in ans_map]
                    lines = [x for x in lines if x['index'] not in ans_map]
                    tups = [(model, line) for line in lines]
                else:
                    main_data = data[[x == y for x, y in zip(data['index'], data['g_index'])]]
                    lines = [data[data['g_index'] == x] for x in main_data['index']]
                    indices = [x.iloc[0]['g_index'] for x in lines if x.iloc[0]['g_index'] not in ans_map] 
                    lines = [x for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    tups = [(model, x) for x in lines]
                    data = main_data

                if len(lines):
                    res = track_progress_rich(
                        Comprehensive_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['hit'] = [x['hit'] for x in judge_results]
            data['log'] = [x['log'] for x in judge_results]
            dump(data, storage)
        
        data = load(storage)
        acc = report_acc(data)
        
        score_file = eval_file.replace(f'.xlsx', '_acc.csv')
        dump(acc, score_file)
        return acc
