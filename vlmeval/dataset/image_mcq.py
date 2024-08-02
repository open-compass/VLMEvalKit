import warnings

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *


class ImageMCQDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv',
        'MMBench_TEST_EN': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv',
        'MMBench_DEV_CN': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv',
        'MMBench_TEST_CN': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv',
        'MMBench': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench.tsv',  # Internal Only
        'MMBench_CN': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_CN.tsv',    # Internal Only
        # MMBench v1.1
        'MMBench_DEV_EN_V11': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN_V11.tsv',
        'MMBench_TEST_EN_V11': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN_V11.tsv',
        'MMBench_DEV_CN_V11': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN_V11.tsv',
        'MMBench_TEST_CN_V11': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN_V11.tsv',
        'MMBench_V11': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_V11.tsv',  # Internal Only
        'MMBench_CN_V11': 'https://opencompass.openxlab.space/utils/VLMEval/MMBench_CN_V11.tsv',    # Internal Only
        # SEEDBench Series
        'SEEDBench_IMG': 'https://opencompass.openxlab.space/utils/VLMEval/SEEDBench_IMG.tsv',
        'SEEDBench2': 'https://huggingface.co/datasets/VLMEval/SEEDBench2/resolve/main/SEEDBench2.tsv',
        'SEEDBench2_Plus': 'https://opencompass.openxlab.space/utils/VLMEval/SEEDBench2_Plus.tsv',
        # ScienceQA Series
        'ScienceQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/ScienceQA_VAL.tsv',
        'ScienceQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/ScienceQA_TEST.tsv',
        # MMT-Bench
        'MMT-Bench_ALL_MI': 'https://opencompass.openxlab.space/utils/VLMEval/MMT-Bench_ALL_MI.tsv',
        'MMT-Bench_ALL': 'https://opencompass.openxlab.space/utils/VLMEval/MMT-Bench_ALL.tsv',
        'MMT-Bench_VAL_MI': 'https://opencompass.openxlab.space/utils/VLMEval/MMT-Bench_VAL_MI.tsv',
        'MMT-Bench_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/MMT-Bench_VAL.tsv',
        # AesBench
        'AesBench_VAL': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_VAL.tsv',
        'AesBench_TEST': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_TEST.tsv',
        # Q-Bench1
        'Q-Bench1_VAL': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_VAL.tsv',
        'Q-Bench1_TEST': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_TEST.tsv',
        # A-Bench
        'A-Bench_VAL': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_VAL.tsv',
        'A-Bench_TEST': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_TEST.tsv',
        # Other Benchmarks
        'CCBench': 'https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv',
        'AI2D_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv',
        'MMStar': 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv',
        'RealWorldQA': 'https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv',
        'MLLMGuard_DS': 'https://opencompass.openxlab.space/utils/VLMEval/MLLMGuard_DS.tsv',
        'BLINK': 'https://opencompass.openxlab.space/utils/VLMEval/BLINK.tsv',
    }

    DATASET_MD5 = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'b6caf1133a01c6bb705cf753bb527ed8',
        'MMBench_TEST_EN': '6939fadb0ce626fefc0bdc9c64efc528',
        'MMBench_DEV_CN': '08b8fc3324a5ed74155350f57be69fbd',
        'MMBench_TEST_CN': '7e1239baf0ee4c8b513e19705a0f317e',
        'MMBench': '4115aea3383f3dd0083be6a633e0f820',  # Internal Only
        'MMBench_CN': '2e053ffc90ea598b1feae13c36dc13ee',    # Internal Only
        # MMBench v1.1
        'MMBench_DEV_EN_V11': '30c05be8f2f347a50be25aa067248184',
        'MMBench_TEST_EN_V11': '26f0f15381a21720255091d3e0316ce6',
        'MMBench_DEV_CN_V11': '593f9b5f6bea453d870a798b34ae4f37',
        'MMBench_TEST_CN_V11': '74bbe4556dac745613c7cbe5ad787050',
        'MMBench_V11': 'b9276414f57af1308dcc4d0cd9b42e7c',  # Internal Only
        'MMBench_CN_V11': '95f6980dd1b4de38e3cbffe0305a3f25',    # Internal Only
        # SEEDBench
        'SEEDBench_IMG': '68017231464752261a2526d6ca3a10c0',
        'SEEDBench2': '4ec15cf864c4f16274112284f531813e',
        'SEEDBench2_Plus': 'e32d3216dc4f452b0fe497a52015d1fd',
        # ScienceQA
        'ScienceQA_VAL': '96320d05e142e585e7204e72affd29f3',
        'ScienceQA_TEST': 'e42e9e00f9c59a80d8a5db35bc32b71f',
        # MMT-Bench
        'MMT-Bench_ALL_MI': '5272157097e19cdd7cb41e412ab3b7c7',
        'MMT-Bench_ALL': 'b273a2f4c596fe4f2605de0494cd632f',
        'MMT-Bench_VAL_MI': 'c7d7b998eb5cd9aa36c7d4f721472462',
        'MMT-Bench_VAL': '8dd4b730f53dbf9c3aed90ca31c928e0',
        # AesBench
        'AesBench_VAL': '3edb0c319e9187aa0b97fe7a11700a8c',
        'AesBench_TEST': '58b1f7ba2cc32e1d68896d6ee716bbf8',
        # Q-Bench1
        'Q-Bench1_VAL': '837bdb6cd2da571713543462815187b7',
        'Q-Bench1_TEST': '15e759bfd58c9d5f30b23a317d347153',
        # A-Bench
        'A-Bench_VAL': '218563ec50d34bb336c814143a5bb9c1',
        'A-Bench_TEST': '567013fb033a20cf23f51d8e865bd16c',
        # Other Benchmarks
        'CCBench': 'f5dde47f24dc5a6fb6e595b409b466ac',
        'AI2D_TEST': '0f593e0d1c7df9a3d69bf1f947e71975',
        'MMStar': 'e1ecd2140806c1b1bbf54b43372efb9e',
        'RealWorldQA': '92321028d2bc29040284b6674721e48f',
        'MLLMGuard_DS': '975fc0dd7119386e198c37d71e274b3f',
        'BLINK': '3b6649b6a662184ea046908e5506260e',
    }

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

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
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, report_acc_MMT, mcq_circular_eval, mcq_vanilla_eval
        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 4)

        circular = False
        if listinstr(['mmbench', 'ccbench'], dataset.lower()):
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            dump(data, eval_file)
            circular = True

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        else:
            acc = report_acc(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        if dataset == 'AesBench_VAL':
            warnings.info('Note that AesBench VAL is just a toy version of AesBench TEST. For full results, \
                           please evaluate on AesBench TEST. The AesBench TEST dataset is more than 20 times \
                           larger than the VAL dataset and the leaderboard results are based on AesBench TEST.')
        return acc


class MMMUDataset(ImageMCQDataset):

    DATASET_URL = {
        'MMMU_DEV_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv',
        'MMMU_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv',
    }

    DATASET_MD5 = {
        'MMMU_DEV_VAL': '521afc0f3bf341e6654327792781644d',
        'MMMU_TEST': 'c19875d11a2d348d07e5eb4bdf33166d',
    }

    @staticmethod
    def split_MMMU(msgs):
        text, images = None, []
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None
                text = s['value']
        text_segs = text.split('<image ')
        if len(text_segs) == 1:
            return msgs

        segs = [dict(type='text', value=text_segs[0])]
        for i, seg in enumerate(text_segs):
            if i == 0:
                continue
            assert istype(seg[0], int) and seg[1] == '>'
            image_idx = int(seg[0]) - 1
            segs.append(dict(type='image', value=images[image_idx]))
            segs.append(dict(type='text', value=seg[2:]))
        return segs

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        msgs = self.split_MMMU(msgs)
        return msgs


class CustomMCQDataset(ImageMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
