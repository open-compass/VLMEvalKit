from .text_base import TextBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from .utils.multiple_choice import report_acc, mcq_vanilla_eval, merge_vanilla_judge


class TextMCQDataset(TextBaseDataset):

    TYPE = 'MCQ'
    DEFAULT_JUDGE = 'chatgpt-0125'
    JUDGE_FORMAT = "{model_name}_{dataset_name}_{judge_name}.tsv"
    RATING_FORMAT = "{model_name}_{dataset_name}_{judge_name}_acc.csv"
    VANILLA_JUDGE_FORMAT = "{model_name}_{dataset_name}_{judge_name}_vanilla.tsv"
    VANILLA_RATING_FORMAT = "{model_name}_{dataset_name}_{judge_name}_vanilla_acc.csv"

    DATASET_URL = {}
    DATASET_MD5 = {}

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        msgs = []

        msgs.append(dict(type='text', value=prompt))

        return msgs

    @classmethod
    def evaluate_heuristic(cls, eval_file, **judge_kwargs):
        # assert dataset is not None
        nproc = judge_kwargs.pop('nproc', 16)

        # Some preprocessing
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = data['prediction'].astype(str)
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        circular = ('g_index' in data)

        model_name = judge_kwargs.get('model', 'exact_matching')

        if model_name == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        if model is None:
            model_name = 'EM'
        judge_file = cls.get_judge_file_path(eval_file, judge_name=model_name)
        rating_file = cls.get_rating_file_path(eval_file, judge_name=model_name)
        tmp_file = judge_file.replace('.tsv', '.pkl')

        acc_func = report_acc

        if circular:
            vanilla_judge_file = judge_file.replace(model_name, model_name + '_vanilla')
            vanilla_rating_file = rating_file.replace(model_name, model_name + '_vanilla')
            vanilla_tmp_file = vanilla_judge_file.replace('.tsv', '.pkl')
            if not osp.exists(vanilla_judge_file):
                vanilla_data = mcq_vanilla_eval(model, data, nproc, vanilla_tmp_file)
                dump(vanilla_data, vanilla_judge_file)
            else:
                vanilla_data = load(vanilla_judge_file)

            circ_df = merge_vanilla_judge(vanilla_data)
            dump(circ_df, judge_file)
            # May have different report acc functions for different datasets
            acc = acc_func(circ_df)
            dump(acc, rating_file)

            if os.environ.get('PRINT_VANILLA', None) == '1':
                vanilla_acc = acc_func(vanilla_data)
                circ0_data = vanilla_data[vanilla_data['index'] == vanilla_data['g_index']]
                circ0_acc = acc_func(circ0_data)
                acc_map = {'vanilla': vanilla_acc, 'circular': acc, 'vanilla_0': circ0_acc}
                # Merge & Print the Evaluation Results
                for k, v in acc_map.items():
                    if 'split' not in v:
                        v['split'] = [None] * len(v)
                    if len(v) == 1 and pd.isna(v['split'][0]):
                        v['split'] = [k]
                    else:
                        assert not pd.isna(v['split'][0])
                        v['split'] = [k + '_' + sp for sp in v['split']]
                score_all = [acc_map['vanilla_0'], acc_map['vanilla'], acc_map['circular']]
                score_all = pd.concat(score_all)
                dump(score_all, vanilla_rating_file)
        else:
            if not osp.exists(judge_file):
                data = mcq_vanilla_eval(model, data, nproc, tmp_file)
                dump(data, judge_file)
            else:
                data = load(judge_file)
            acc = acc_func(data)
            dump(acc, rating_file)
        return acc


class CustomTextMCQDataset(TextMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                localize_tsv(data_path, local_path)
            data_path = local_path
        return load(data_path)
