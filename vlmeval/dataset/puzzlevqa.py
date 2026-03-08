from .image_mcq import ImageMCQDataset
from .utils import build_judge
from vlmeval.smp import *


class PuzzleVQA(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_URL = {
        'PuzzleVQA': 'https://opencompass.openxlab.space/utils/VLMEval/PuzzleVQA.tsv'
    }
    DATASET_MD5 = {
        'PuzzleVQA': '6a4a40bd8967db728c06202e8265a49b',
    }
    DEFAULT_JUDGE = 'gpt-4o-mini'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = line['options']
        letters = string.ascii_uppercase

        options = '\n'.join([f"({letters[i]}) {item}" for i, item in enumerate(eval(options))])

        prompt = ''
        prompt += question + '\nOptions:\n' + options
        prompt += "\nSolve the multiple-choice question and then answer with the option letter from the given choices. "
        prompt += "The last line of your response should be of the following format:"
        prompt += "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
        prompt += "Think step by step before answering."

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import mcq_vanilla_eval, report_acc

        nproc = judge_kwargs.pop('nproc', 16)
        judge_name = judge_kwargs.get('model', 'EM')
        if judge_name == 'EM':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                model = None
                judge_name = 'EM'
        judge_file = self.get_judge_file_path(eval_file, judge_name=judge_name)
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge_name}', 'pkl')
        rating_file = self.get_rating_file_path(eval_file, judge_name=judge_name)

        if not osp.exists(judge_file):
            data = load(eval_file)
            data = mcq_vanilla_eval(model, data, nproc, result_file=tmp_file)
            dump(data, judge_file)
        else:
            data = load(judge_file)

        acc = report_acc(data)
        dump(acc, rating_file)
        return acc
