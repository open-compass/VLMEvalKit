from ..smp import *
from .image_base import ImageBaseDataset
from .image_mcq import MMMUDataset
from .utils.mathv import MATH_V_auxeval_MCQ, MATH_V_auxeval_Open
from .utils.multiple_choice import report_acc
from .utils.judge_util import build_judge


def extract_choices(line):
    assert not pd.isna(line['A'])
    choices = []
    for ch in 'ABCDEFGHIJKL':
        if not pd.isna(line[ch]):
            choices.append(line[ch])
        else:
            break
    return choices


def EMMA_auxeval(model, line):
    line['answer'], line['prediction'] = str(line['answer']), str(line['prediction'])
    if line['type'] == 'MCQ':
        choices = extract_choices(line)
        return MATH_V_auxeval_MCQ(model, line, choices)
    else:
        return MATH_V_auxeval_Open(model, line)


class EMMADataset(ImageBaseDataset):

    COT_INST = "Please solve the problem step by step. "
    DIRECT_INST = "Please ensure that your output only contains the final answer without any additional content (such as intermediate reasoning steps)."  # noqa: E501
    MCQ_FMT = "{context}\n\n{question}\n\n{options}\n\nAnswer with the option's letter from the given choices. "
    OPEN_FMT = "{context}\n\n{question}\n\nAnswer the question using a single word or phrase. "

    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    DEFAULT_JUDGE = 'gpt-4o-1120'
    TYPE = 'VQA'

    DATASET_URL = {
        'EMMA': 'https://opencompass.openxlab.space/utils/VLMEval/EMMA.tsv',
        'EMMA_COT': 'https://opencompass.openxlab.space/utils/VLMEval/EMMA.tsv',
        'EMMA_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/EMMA_MINI.tsv',
        'EMMA_MINI_COT': 'https://opencompass.openxlab.space/utils/VLMEval/EMMA_MINI.tsv'
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        context = line['context']
        question = line['question']
        example = ""
        _ = {}
        if line['type'] == 'MCQ':
            for ch in string.ascii_uppercase:
                if ch in line and not pd.isna(line[ch]):
                    example += f"{ch}: {line[ch]}\n"

            prompt_tmpl = EMMADataset.MCQ_FMT
            if not pd.isna(context) and context is not None:
                prompt = prompt_tmpl.format(context=context, question=question, options=example)
            else:
                prompt = prompt_tmpl.split('{context}\n\n')[1].format(question=question, options=example)
            prompt += EMMADataset.COT_INST if 'COT' in self.dataset_name else EMMADataset.DIRECT_INST
        else:
            prompt_tmpl = EMMADataset.OPEN_FMT
            if not pd.isna(context) and context is not None:
                prompt = prompt_tmpl.format(context=context, question=question)
            else:
                prompt = prompt_tmpl.split('{context}\n\n')[1].format(question=question)
            prompt += EMMADataset.COT_INST if 'COT' in self.dataset_name else EMMADataset.DIRECT_INST

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return MMMUDataset.split_MMMU(msgs)

    def evaluate(self, eval_file, **judge_kwargs):

        model = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 16)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=1024, **judge_kwargs)
            assert model.working(), 'EMMA evaluation requires a working OPENAI API\n'
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
                _ = track_progress_rich(
                    EMMA_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)

            keys = ['res', 'log', 'hit']
            for k in keys:
                data[k] = [ans[idx][k] for idx in data['index']]
            dump(data, storage)

        score = report_acc(load(storage))
        score_pth = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(score, score_pth)
        return score
