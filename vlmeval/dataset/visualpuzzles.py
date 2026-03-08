import re
from vlmeval.smp import *
from .image_mcq import ImageMCQDataset
from .utils import build_judge


class VisualPuzzles(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_URL = {
        'VisualPuzzles': 'https://opencompass.openxlab.space/utils/VLMEval/VisualPuzzles.tsv'
    }
    DATASET_MD5 = {
        'VisualPuzzles': '12bcc3f6dd7a11b33ffcd526b5601076',
    }
    DEFAULT_JUDGE = 'gpt-4o-mini'

    def format_options(self, opt_str):
        if not opt_str or opt_str == 'nan':
            return None

        # 提取所有被引号包住的内容
        items = re.findall(r"'(.*?)'", opt_str)

        # 生成 A. B. C. D.
        letters = string.ascii_uppercase
        formatted = [f"({letters[i]}) {item}" for i, item in enumerate(items)]

        return "\n".join(formatted)

    def options_to_list(self, opt_str):
        if not opt_str or opt_str == 'nan':
            return None

        # 提取所有被引号包住的内容
        items = re.findall(r"'(.*?)'", opt_str)
        return items

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        if not pd.isna(line['options']):
            options = "Options:\n" + self.format_options(line['options'])
        else:
            options = 'Options: Choose from (A) (B) (C) (D) in the image.'
        prompt = ''
        prompt += question + '\n' + options
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
        # breakpoint()
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
            # Preprocess Options
            data['options'] = [self.options_to_list(str(opt)) for opt in data['options']]
            data = mcq_vanilla_eval(model, data, nproc, result_file=tmp_file)
            dump(data, judge_file)
        else:
            data = load(judge_file)

        acc = report_acc(data, inds=['category', 'difficulty'])
        dump(acc, rating_file)
        return acc
