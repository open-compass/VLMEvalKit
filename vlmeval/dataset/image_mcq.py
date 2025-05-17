import warnings

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import pandas as pd

MMMB_URLS = {
    'MMMB_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ar.tsv',
    'MMMB_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_cn.tsv',
    'MMMB_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_en.tsv',
    'MMMB_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_pt.tsv',
    'MMMB_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ru.tsv',
    'MMMB_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_tr.tsv',
}

MTL_MMBench_URLS = {
    'MMBench_dev_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ar.tsv',
    'MMBench_dev_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_cn.tsv',
    'MMBench_dev_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_en.tsv',
    'MMBench_dev_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_pt.tsv',
    'MMBench_dev_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_tr.tsv',
    'MMBench_dev_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ru.tsv',
}

MMMB_MD5 = {
    'MMMB_ar': 'f3a18b6385f1d9701840aa42de27aead', 'MMMB_cn': '13ed82fa89730037292fcaa27f08f430',
    'MMMB_en': '1cd781a71ec5a2983c090b84105d6a01', 'MMMB_pt': '548ea2b3bb2da991790386f0015d30d1',
    'MMMB_ru': 'ce1cc8a0533425ab0d86b326ebfc2984', 'MMMB_tr': '0733739d43090327975294292bc5cd67'
}

MTL_MMBench_MD5 = {
    'MMBench_dev_ar': '4271b4a0d0200e1a86380a878e0d64a4', 'MMBench_dev_cn': '2ed5135326fed02c8e51ea50dda8222f',
    'MMBench_dev_en': 'd9ab776fc018b3d45785e9a5c23431c2', 'MMBench_dev_pt': '4ddfbcd27ef12444b908c03831cd0295',
    'MMBench_dev_tr': '4fab39d501389d3d6cc90264bb708f11', 'MMBench_dev_ru': '5ba1171ff2e68f80637bf78349e402a5'
}


class ImageMCQDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv',
        'MMBench_TEST_EN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN.tsv',
        'MMBench_DEV_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN.tsv',
        'MMBench_TEST_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN.tsv',
        'MMBench': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench.tsv',  # Internal
        'MMBench_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_CN.tsv',  # Internal
        # MMBench v1.1
        'MMBench_DEV_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv',
        'MMBench_TEST_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN_V11.tsv',
        'MMBench_DEV_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN_V11.tsv',
        'MMBench_TEST_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN_V11.tsv',
        'MMBench_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_V11.tsv',  # Internal
        'MMBench_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_CN_V11.tsv',  # Internal
        # SEEDBench Series
        'SEEDBench_IMG': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv',
        'SEEDBench2': 'https://huggingface.co/datasets/VLMEval/SEEDBench2/resolve/main/SEEDBench2.tsv',
        'SEEDBench2_Plus': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench2_Plus.tsv',
        # ScienceQA Series
        'ScienceQA_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_VAL.tsv',
        'ScienceQA_TEST': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_TEST.tsv',
        # MMT-Bench
        'MMT-Bench_ALL_MI': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_ALL_MI.tsv',
        'MMT-Bench_ALL': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_ALL.tsv',
        'MMT-Bench_VAL_MI': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL_MI.tsv',
        'MMT-Bench_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL.tsv',
        # AesBench
        'AesBench_VAL': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_VAL.tsv',
        'AesBench_TEST': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_TEST.tsv',
        # Q-Bench1
        'Q-Bench1_VAL': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_VAL.tsv',
        'Q-Bench1_TEST': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_TEST.tsv',
        # A-Bench
        'A-Bench_VAL': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_VAL.tsv',
        'A-Bench_TEST': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_TEST.tsv',
        # R-Bench
        'R-Bench-Dis': 'https://huggingface.co/datasets/lcysyzxdxc/R-Bench/blob/main/R-bench-dis.tsv',
        'R-Bench-Ref': 'https://huggingface.co/datasets/lcysyzxdxc/R-Bench/blob/main/R-bench-ref.tsv',
        # Other Benchmarks
        'CCBench': 'https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv',
        'AI2D_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv',
        'AI2D_TEST_NO_MASK': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST_NO_MASK.tsv',
        'MMStar': 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv',
        'RealWorldQA': 'https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv',
        'MLLMGuard_DS': 'https://opencompass.openxlab.space/utils/VLMEval/MLLMGuard_DS.tsv',
        'BLINK': 'https://opencompass.openxlab.space/utils/VLMEval/BLINK.tsv',
        'TaskMeAnything_v1_imageqa_random': (
            'https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random/'
            'resolve/main/TaskMeAnything-v1-imageqa-random.tsv'
        ),
        'A-OKVQA': 'https://huggingface.co/datasets/Allen8/A-OKVQA/resolve/main/a-okvqa.tsv',
        'WorldMedQA-V': 'https://opencompass.openxlab.space/utils/VLMEval/WorldMedQA-V.tsv',
        'VisOnlyQA-VLMEvalKit': (
            'https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real/'
            'resolve/main/visonlyqa_vlmevalkit.tsv'
        ),
        '3DSRBench': (
            'https://huggingface.co/datasets/ccvl/3DSRBench/'
            'resolve/main/3dsrbench_v1_vlmevalkit_circular.tsv'
        ),
        'MMCR': 'http://opencompass.openxlab.space/utils/VLMEval/MMCR.tsv',
        'MMSci_DEV_MCQ': 'https://opencompass.openxlab.space/utils/VLMEval/MMSci_DEV_MCQ.tsv',
        "MMVP": "http://opencompass.openxlab.space/utils/VLMEval/MMVP.tsv",
        # For Internal Use Only
        'MMBench_V11_MINI': 'https://opencompass.openxlab.space/utils/TEST/MMBench_V11_MINI.tsv',
        'MMStar_MINI': 'https://opencompass.openxlab.space/utils/TEST/MMStar_MINI.tsv',
        'AI2D_MINI': 'https://opencompass.openxlab.space/utils/TEST/AI2D_MINI.tsv',
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
        'SEEDBench2_Plus': '7cb2323950d71f049df70e5162062af3',
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
        # R-Bench
        'R-Bench-Dis': 'd6e961dbfc43350688af2560226830b4',
        'R-Bench-Ref': '270c1cb555acb523f3fdb178ed57021d',
        # Other Benchmarks
        'CCBench': 'f5dde47f24dc5a6fb6e595b409b466ac',
        'AI2D_TEST': '0f593e0d1c7df9a3d69bf1f947e71975',
        'AI2D_TEST_NO_MASK': 'fd8f463634d4fe9fbd23b876e8eea5be',
        'MMStar': 'e1ecd2140806c1b1bbf54b43372efb9e',
        'RealWorldQA': '4de008f55dc4fd008ca9e15321dc44b7',
        'MLLMGuard_DS': '975fc0dd7119386e198c37d71e274b3f',
        'BLINK': '3b6649b6a662184ea046908e5506260e',
        'TaskMeAnything_v1_imageqa_random': '023fef69e2ca21827afb77c5ec3bc889',
        'WorldMedQA-V': '441e63875e30c87f5750528b57b41285',
        "VisOnlyQA-VLMEvalKit": 'cf460a31d2acb8d3a7cecd0e69298bfa',
        '3DSRBench': '13a99f33164dc1b9faf0e8b8b01fd6f2',
        'MMCR': '9052635f2c3835bdb87755ef73564f5e',
        'MMSci_DEV_MCQ': '71c82f81920a84526803574f719099a7',
        "MMVP": "8cb732b141a0cba5b42159df2839e557",
    }

    DATASET_URL.update(MMMB_URLS)
    DATASET_URL.update(MTL_MMBench_URLS)
    DATASET_MD5.update(MMMB_MD5)
    DATASET_MD5.update(MTL_MMBench_MD5)

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
        from .utils.multiple_choice import (
            report_acc, report_acc_MMT, report_acc_MMSci, mcq_circular_eval, mcq_vanilla_eval
        )
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
        if listinstr(['mmbench', 'ccbench', 'circular', 'mmcr'], dataset.lower()):
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
        elif 'MMSci' in dataset:
            acc = report_acc_MMSci(data)
        else:
            acc = report_acc(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        if dataset == 'AesBench_VAL':
            warnings.warn('Note that AesBench VAL is just a toy version of AesBench TEST. For full results, \
                           please evaluate on AesBench TEST. The AesBench TEST dataset is more than 20 times \
                           larger than the VAL dataset and the leaderboard results are based on AesBench TEST.')
        if dataset == 'VisOnlyQA-VLMEvalKit':
            warnings.warn('Note that the results on VisOnlyQA-VLMEvalKit are different from the results on \
                           the original VisOnlyQA. VisOnlyQA-VLMEvalKit does not include the \
                           chemistry__shape_multi split and uses a different evaluation prompt. Please \
                           explicitly specify the version of the dataset when you report results.')

        return acc


class MedXpertQA_MM_test(ImageMCQDataset):

    DATASET_URL = {
        'MedXpertQA_MM_test': 'https://opencompass.openxlab.space/utils/VLMEval/MedXpertQA_MM_test.tsv',
    }

    DATASET_MD5 = {
        'MedXpertQA_MM_test': '73c12d28ebdfca97c5fd3c3be3fe357b',
    }


class MMMUDataset(ImageMCQDataset):

    DATASET_URL = {
        'MMMU_DEV_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv',
        'MMMU_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv',
    }

    DATASET_MD5 = {
        'MMMU_DEV_VAL': '585e8ad75e73f75dcad265dfd0417d64',
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


class MMMUProDataset(MMMUDataset):

    TYPE = 'MCQ_MMMU_Pro'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'MMMU_Pro_V' in self.dataset_name:
            self.data['question'] = ['placeholder'] * len(self.data)

    DATASET_URL = {
        'MMMU_Pro_10c': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv',
        'MMMU_Pro_10c_COT': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv',
        'MMMU_Pro_V': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv',
        'MMMU_Pro_V_COT': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv',
    }

    DATASET_MD5 = {
        'MMMU_Pro_10c': '22cee868fe6b680d14b99bfff6db8172',
        'MMMU_Pro_10c_COT': '22cee868fe6b680d14b99bfff6db8172',
        'MMMU_Pro_V': 'd01441a87b3dbe721b5a04652ae38009',
        'MMMU_Pro_V_COT': 'd01441a87b3dbe721b5a04652ae38009',
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        if 'MMMU_Pro_V' in self.dataset_name:
            question = 'Answer the following multiple-choice question in the image. '
            if 'COT' in self.dataset_name:
                question += (
                    "The last line of your response should be of the following format: 'Answer: $LETTER' "
                    "(without quotes) where LETTER is one of the options. Think step by step before answering. "
                )
            else:
                question += "Answer directly with the option letter from the given choices. "
            if isinstance(tgt_path, list):
                assert len(tgt_path) == 1
                tgt_path = tgt_path[0]
            return [dict(type='image', value=tgt_path), dict(type='text', value=question)]
        else:
            question = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            prompt = ''
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                if 'COT' in self.dataset_name:
                    prompt += (
                        "Answer the following multiple-choice question. The last line of your response should be of "
                        "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options. "
                        "Think step by step before answering. "
                    )
                else:
                    prompt += "Answer directly with the option letter from the given choices. "

            msgs = []
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]
            msgs.append(dict(type='text', value=prompt))
            msgs = self.split_MMMU(msgs)
            return msgs

    def cot_postproc(self, response):
        lines = response.strip().split('\n')
        lines = [x.strip() for x in lines]
        cands = [x for x in lines if x.startswith('Answer:')]
        if len(cands) == 1:
            counter = defaultdict(lambda: 0)
            for ch in cands[0]:
                if ch in string.ascii_uppercase:
                    counter[ch] += 1
            if len(counter) == 1:
                return list(counter.keys())[0]
            else:
                return cands[0][7:]
        return response

    def evaluate(self, eval_file, **judge_kwargs):
        if 'COT' in self.dataset_name:
            data = load(eval_file)
            data['prediction'] = [self.cot_postproc(x) for x in data['prediction']]
            tgt = eval_file.replace('.xlsx', '_cotpost.xlsx')
            dump(data, tgt)
            res = super().evaluate(tgt, **judge_kwargs)
            acc_org = eval_file.replace('.xlsx', '_acc.csv')
            acc_now = eval_file.replace('.xlsx', '_cotpost_acc.csv')
            shutil.copy(acc_now, acc_org)
            return res
        else:
            return super().evaluate(eval_file, **judge_kwargs)


class MUIRDataset(ImageMCQDataset):

    DATASET_URL = {
        'MUIRBench': 'http://opencompass.openxxlab.com/utils/VLMEval/MUIRBench.tsv'
    }

    DATASET_MD5 = {
        'MUIRBench': '2e5e6fd7699761b08a7cb3ab8c0c2ec8'
    }

    @staticmethod
    def split_MUIR(msgs):
        text, images = None, []

        # Separate images and text from msgs
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None  # Ensure only one text entry is expected
                text = s['value']

        # Split text by <image> tags
        text_segs = text.split('<image>')

        # Initialize the segments list
        segs = []

        # Iterate through the text segments and images
        for i, seg in enumerate(text_segs):
            # Append the image if this is not the first segment and there are still images left
            if i > 0 and i - 1 < len(images):
                segs.append(dict(type='image', value=images[i - 1]))
            # Append the text segment (if it's non-empty)
            if len(seg) > 0:
                segs.append(dict(type='text', value=seg))

        return segs

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
        # options_prompt = ''
        options_prompt = '\n'.join([f'{key}. {item}' for key, item in options.items()])
        # for key, item in options.items():
        #     options_prompt += f'{key}. {item}\n'

        prompt = ''

        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "\nAnswer with the option's letter from the given choices directly."

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        msgs = self.split_MUIR(msgs)
        return msgs


class GMAIMMBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'GMAI-MMBench_VAL': 'https://huggingface.co/datasets/VLMEval/GMAI-MMBench/resolve/main/GMAI-MMBench_VAL.tsv',
        'GMAI_mm_bench_TEST_part_1': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_1.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_2': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_2.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_3': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_3.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_4': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_4.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_5': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_5.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_6': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_6.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_7': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_7.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_8': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_8.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_9': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_9.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_10': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_10.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_11': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_11.tsv',  # noqa: E501
    }

    DATASET_MD5 = {
        'GMAI-MMBench_VAL': '254bd581627866f1c499d3d6b4422324',
        'GMAI_mm_bench_TEST_part_1': '900d735231230a63f4ed45665c078ef4',
        'GMAI_mm_bench_TEST_part_2': '1b27ab621386945d7e4a765ad2d22b0e',
        'GMAI_mm_bench_TEST_part_3': '44bdc2b6267dd505d529b8cad06f0fb2',
        'GMAI_mm_bench_TEST_part_4': '5a04a04fcac9f1466709f242fdb80acb',
        'GMAI_mm_bench_TEST_part_5': 'c70baf8909eda9af0ddeab275c721336',
        'GMAI_mm_bench_TEST_part_6': '825abc39596b644dead9350d0cfa3b96',
        'GMAI_mm_bench_TEST_part_7': 'defb8aed2fb77365a76b6b9abd6a2701',
        'GMAI_mm_bench_TEST_part_8': 'ff490d60b85f2bb0abb67a435b298c65',
        'GMAI_mm_bench_TEST_part_9': 'ff67c86f40da93b09139ac1d1ba5dc6b',
        'GMAI_mm_bench_TEST_part_10': '3dae94627b9ac0fe00180d4780fbf6dc',
        'GMAI_mm_bench_TEST_part_11': 'd08dc813f0eb6bbab63cae2a9d113c4b',
    }

    @classmethod
    def supported_datasets(cls):
        return ['GMAI-MMBench_VAL', 'GMAI-MMBench_TEST']

    def load_data(self, dataset):
        if dataset == 'GMAI-MMBench_VAL':
            data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
            if file_size(data_path, 'GB') > 1:
                local_path = data_path.replace('.tsv', '_local.tsv')
                if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                    from ..tools import LOCALIZE
                    LOCALIZE(data_path, local_path)
                data_path = local_path
            return load(data_path)
        elif dataset == 'GMAI-MMBench_TEST':
            dfs = []
            for part_num in range(1, 12):
                part_name = f'GMAI_mm_bench_TEST_part_{part_num}'
                url = self.DATASET_URL[part_name]
                file_md5 = self.DATASET_MD5.get(part_name)
                tsv_path = osp.join(LMUDataRoot(), f'{part_name}.tsv')
                if not osp.exists(tsv_path) or (file_md5 and md5(tsv_path) != file_md5):
                    download_file(url, filename=tsv_path)
                local_path = tsv_path.replace('.tsv', '_local.tsv')
                if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                    from ..tools import LOCALIZE
                    LOCALIZE(tsv_path, local_path)
                tsv_path = local_path
                # 加载数据
                df = load(tsv_path)
                dfs.append(df)
            # 合并所有数据
            data = pd.concat(dfs, ignore_index=True)
            return data
        else:
            raise ValueError(f"未知的数据集：{dataset}")

    def report_acc_by_groups(self, df, group_column):
        res = defaultdict(list)

        # Check for the 'split' column
        if 'split' in df:
            splits = list(set(df['split']))
            res['split'] = splits
        else:
            df['split'] = ['none'] * len(df)
            res['split'] = ['none']

        res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]

        if group_column not in df:
            raise ValueError(f"Column '{group_column}' not found in dataframe.")  # noqa: E713

        abilities = list(set(df[group_column]))
        abilities = ['None' if isinstance(ab, float) and pd.isna(ab) else ab for ab in abilities]
        abilities.sort()

        for ab in abilities:
            ab_name = ab
            sub_df = df[df[group_column] == ab]
            res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

        return pd.DataFrame(res)

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, mcq_vanilla_eval
        nproc = judge_kwargs.pop('nproc', 4)

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

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = report_acc(data)

        for group_col in ['clinical vqa task', 'department', 'perceptual granularity']:
            acc_grouped = self.report_acc_by_groups(data, group_col)
            score_file_grouped = eval_file.replace(f'.{suffix}', f'_{group_col}_acc.csv')
            dump(acc_grouped, score_file_grouped)

        return acc


class MMERealWorld(ImageMCQDataset):

    TYPE = 'MMERealWorld'

    DATASET_MD5 = {
        'MME-RealWorld': '271c33ec814c39533c467ec6fb8a6f36',
        'MME-RealWorld-Lite': '4c17057d7d3b6c4a0d4397c3dae0881c',
        'MME-RealWorld-CN': 'daaa763d52a760a38606d5dedb3fe444',
    }
    SYS = {
        'MME-RealWorld': (
            'Select the best answer to the above multiple-choice question based on the image. '
            'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
            'The best answer is:'
        ),
        'MME-RealWorld-Lite': (
            'Select the best answer to the above multiple-choice question based on the image. '
            'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
            'The best answer is:'
        ),
        'MME-RealWorld-CN': (
            '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n'
            '最佳答案为：'
        ),
    }

    @classmethod
    def supported_datasets(cls):
        return ['MME-RealWorld', 'MME-RealWorld-CN', 'MME-RealWorld-Lite',]

    def load_data(
        self, dataset="MME-RealWorld", repo_id="yifanzhang114/MME-RealWorld-Base64"
    ):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset}.tsv")

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.DATASET_MD5[dataset]:
                return False
            return True

        def generate_tsv(pth):
            tsv_file = os.path.join(pth, f"{dataset}.tsv")

            if os.path.exists(tsv_file):
                print(f"{tsv_file} already exists.")
                return

            json_dir = os.path.join(pth, dataset)
            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

            data_list = []
            for json_file in json_files:
                with open(os.path.join(json_dir, json_file), "r") as f:
                    data = json.load(f)
                    for item in tqdm(data):
                        choice_prompt = (
                            "The choices are listed below:\n"
                            if dataset in ["MME-RealWorld", "MME-RealWorld-Lite"]
                            else "选项如下所示:\n"
                        )
                        data_list.append(
                            {
                                "index": item["index"],
                                "image": item["image"],
                                "question": item["question"],
                                "multi-choice options": choice_prompt
                                + "\n".join(item["multi-choice options"]),
                                "A": item["multi-choice options"][0][4:],
                                "B": item["multi-choice options"][1][4:],
                                "C": item["multi-choice options"][2][4:],
                                "D": item["multi-choice options"][3][4:],
                                "E": item["multi-choice options"][4][4:],
                                "answer": item["answer"],
                                "category": item["category"],
                                "l2-category": item["l2-category"],
                            }
                        )
            df = pd.DataFrame(data_list)
            df.to_csv(tsv_file, sep="\t", index=False)
            print(f"TSV file saved to {tsv_file}")

        # Check if dataset is cached and has integrity
        if dataset == "MME-RealWorld-Lite":
            url = 'https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64/resolve/main/mme_realworld_lite.tsv'  # noqa: E501
            file_md5 = (
                self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
            )
            datas = self.prepare_tsv(url, file_md5)
            choice_prompt = "The choices are listed below:\n"
            for index, item in datas.iterrows():
                options = eval(item["multi-choice options"])
                datas.loc[index, "multi-choice options"] = choice_prompt + "\n".join(
                    options
                )
                datas.loc[index, "A"] = options[0][4:]
                datas.loc[index, "B"] = options[1][4:]
                datas.loc[index, "C"] = options[2][4:]
                datas.loc[index, "D"] = options[3][4:]
                datas.loc[index, "E"] = options[4][4:]
            return datas

        update_flag = False
        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
            print(f"Using cached dataset from {cache_path}")
        else:
            from huggingface_hub import snapshot_download

            # Download or find the dataset path
            dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
            generate_tsv(dataset_path)
            update_flag = True

        data_path = os.path.join(dataset_path, f"{dataset}.tsv")
        if file_size(data_path, "GB") > 1:
            local_path = data_path.replace(".tsv", "_local.tsv")
            if (
                not osp.exists(local_path)
                or os.environ.get("FORCE_LOCAL", None)
                or update_flag
            ):
                from vlmeval.tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def post_build(self, dataset):
        self.TYPE = 'MMERealWorld'

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        choice_prompt = line['multi-choice options'] + '\n'
        question += ' ' + choice_prompt + self.SYS[self.dataset_name]

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import extract_characters_regex, get_dimension_rating
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        FAIL_MSG = 'Failed to obtain answer via API.'
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            cnt_rejected = 0
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                extract_pred = extract_characters_regex(pred)
                if extract_pred == '':
                    cnt_rejected += 1
                    data.loc[data['index'] == idx, 'score'] = 0
                else:
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {cnt_rejected} questions. '
                f'Those questions will be counted as 0 score in ALL rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating


class CVBench(ImageMCQDataset):
    """CV-Bench, composed of two sub datasets:
    CV-Bench-2D: 2D computer vision tasks
    CV-Bench-3D: 3D computer vision tasks

    Reference:
    - https://cambrian-mllm.github.io/
    - https://huggingface.co/datasets/nyu-visionx/CV-Bench

    Evaluation strategy:
        See [Cambrian-1](https://arxiv.org/pdf/2406.16860) Appendix C
    """
    DATASET_URL = {
        "CV-Bench-2D": "http://opencompass.openxlab.space/utils/VLMEval/CV-Bench-2D.tsv",
        "CV-Bench-3D": "http://opencompass.openxlab.space/utils/VLMEval/CV-Bench-3D.tsv",
    }

    DATASET_MD5 = {
        "CV-Bench-2D": "a7cff4cc2857cc237ee2b89e62bccb2d",
        "CV-Bench-3D": "bb94c0d568d652d15b60e001ac40a170",
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line["image_path"])
        else:
            tgt_path = self.dump_image(line)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        # use the prompt provided by the dataset
        msgs.append(dict(type="text", value=line["prompt"]))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import mcq_vanilla_eval, report_acc

        nproc = judge_kwargs.pop("nproc", 4)

        suffix = eval_file.split(".")[-1]
        model_name = judge_kwargs.get("model", "extract_matching")

        if model_name == "exact_matching":
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn(
                "OPENAI_API_KEY is not set properly, will use exact matching for evaluation"
            )
            model = None

        result_file = eval_file.replace(f".{suffix}", f"_{model_name}_result.pkl")

        data = load(eval_file)
        data = data.sort_values(by="index")
        data["prediction"] = [str(x) for x in data["prediction"]]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(
                k
            )

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta["index"], meta["question"])}
        data_map = {x: y for x, y in zip(data["index"], data["question"])}
        for k in data_map:
            assert (
                k in meta_q_map
            ), f"eval_file should be the same as or a subset of dataset {self.dataset_name}"

        score_file = eval_file.replace(f".{suffix}", "_acc.csv")

        if osp.exists(score_file):
            acc = load(score_file)
            return acc
        data = mcq_vanilla_eval(
            model, data, meta, nproc, result_file, self.dataset_name
        )
        dump(data, eval_file.replace(f".{suffix}", f"_{model}_result.{suffix}"))
        data = load(eval_file.replace(f".{suffix}", f"_{model}_result.{suffix}"))

        if all(data["split"] == "2D"):  # 2D
            acc = self.report_accuracy(data)
        else:  # 3D, use default evaluation strategy
            acc = report_acc(data)

        score_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(acc, score_file)

        return acc

    def report_accuracy(self, data):
        # CV-Bench-2D evaluation strategy
        # first calculate the accuracy for each source
        # then calculate the overall accuracy by averaging across all sources
        res = defaultdict(list)

        splits = list(set(data["split"]))
        res["split"] = splits

        sources = set(data["source"])
        for source in sources:
            sub_df = data[data["source"] == source]
            res[source] = [
                np.mean(sub_df[sub_df["split"] == sp]["hit"]) for sp in res["split"]
            ]
        res = pd.DataFrame(res)
        res["Overall"] = 0
        for source in sources:
            res["Overall"] += res[source]
        res["Overall"] = res["Overall"] / len(sources)
        return res


class HRBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'HRBench4K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.tsv',
        'HRBench8K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.tsv',
    }

    DATASET_MD5 = {
        'HRBench4K': 'f6b041b03d49543494b8a56d2e35be65',
        'HRBench8K': '274c9c7f89329b804a4723178a00219c',
    }

    def evaluate(self, eval_file, **judge_kwargs):
        assert os.path.exists(eval_file), '{} does not exist!'.format(eval_file)
        from .utils.multiple_choice import mcq_vanilla_eval
        from .utils.hrbench import report_acc_hrbench
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'extract_matching')
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

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')

        if osp.exists(score_file):
            acc = load(score_file)
            return acc
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = report_acc_hrbench(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc


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


class NaturalBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'NaturalBenchDataset': (
            'https://huggingface.co/datasets/BaiqiL/'
            'NaturalBench/resolve/main/NaturalBenchDataset.tsv'
        ),
    }
    DATASET_MD5 = {
        'NaturalBenchDataset':'dbe25b044bc35696426381e9ba4fe930',
    }

    def build_prompt(self, line):
        SUFFIX_FOR_VQA = {
            "yes_no": "Please answer Yes or No.",
            "multiple_choice": "Please output the letter corresponding to the correct option."
        }
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        prompt = f'{question} {SUFFIX_FOR_VQA[line["type"]]}'
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.naturalbench import extract_answer, get_scores

        data = load(eval_file)
        data = data.sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        answers = [str(x) for x in data['answer']]
        indexs = [str(x) for x in data['index']]
        meta = self.data
        types = [str(x) for x in meta['type']]
        results = {}
        assert len(predictions) == len(answers) == len(indexs) == len(types) == (1900 * 4)
        number_answered_samples = len(predictions) // 4
        for i in range(number_answered_samples):
            results[i] = {
                "q0_i0": extract_answer(predictions[i * 4], types[i * 4]),
                "q0_i1": extract_answer(predictions[i * 4 + 1], types[i * 4 + 1]),
                "q1_i0": extract_answer(predictions[i * 4 + 2], types[i * 4 + 2]),
                "q1_i1": extract_answer(predictions[i * 4 + 3], types[i * 4 + 3])
            }

        scores = get_scores(results)
        print(scores)
        score_file = 'NaturalBench_acc.csv'
        df = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score'])
        dump(df, score_file)

        return scores


class WeMath(ImageBaseDataset):
    TYPE = 'MCQ'
    DATASET_URL = {
        'WeMath': 'https://opencompass.openxlab.space/utils/VLMEval/WeMath.tsv',
        'WeMath_COT': 'https://opencompass.openxlab.space/utils/VLMEval/WeMath.tsv',
    }
    DATASET_MD5 = {'WeMath': 'b5e969a075f01290a542411fb7766388',
                   'WeMath_COT': 'b5e969a075f01290a542411fb7766388'}

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

        if 'COT' in self.dataset_name:
            requirement = line['requirement']
            if requirement is not None:
                prompt += f'\n{requirement}'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.wemath import wemath_evaluate_models, wemath_accuracy
        from .utils.multiple_choice import mcq_vanilla_eval

        # model = judge_kwargs['model']
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['exact_matching', 'gpt-4-0125', 'gpt-4-turbo', 'gpt-4o-mini'], model
        name_str_map = {'gpt-4-0125': 'gpt4', 'gpt-4-turbo': 'gpt4-turbo', 'gpt-4o-mini': 'gpt4o-mini'}
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

        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{name_str}.xlsx')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage) and model is not None:
            data = load(eval_file)
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
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

            if 'id' in data.columns:
                # 更改列名
                data.rename(columns={'id': 'ID'}, inplace=True)
            dump(data, storage)
        if osp.exists(storage):
            accuracy_scores = wemath_evaluate_models(storage)
            four_dim_scores = wemath_accuracy(storage)
        else:
            accuracy_scores = wemath_evaluate_models(eval_file)
            four_dim_scores = wemath_accuracy(eval_file)
        combine_score = {**accuracy_scores, **four_dim_scores}
        combine_score = pd.DataFrame(combine_score)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(combine_score, score_pth)
        return combine_score


class VMCBenchDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        'VMCBench_DEV': 'https://huggingface.co/datasets/suyc21/VMCBench/resolve/main/data/tsv/VMCBench_DEV.tsv',
        'VMCBench_TEST': 'https://huggingface.co/datasets/suyc21/VMCBench/resolve/main/data/tsv/VMCBench_TEST.tsv'
    }

    DATASET_MD5 = {
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
        prompt = ''
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "Answer with the option's letter from the given choices directly. \n"

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vmcbench import get_mc_score, report_vmc_acc
        suffix = eval_file.split('.')[-1]
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        data['hit'] = data.apply(get_mc_score, axis=1)
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')
        dump(data, result_file)
        acc = report_vmc_acc(data)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc


class LEGO(ImageMCQDataset):

    DATASET_URL = {
        'LEGO': 'https://opencompass.openxlab.space/utils/VLMEval/LEGO.tsv',
    }
    DATASET_MD5 = {'LEGO': 'd595f50e1fb4d4eb12cbc95297893ffc'}

    @staticmethod
    def split_LEGO(msgs):
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

    def build_prompt_sort(self, line):

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
            prompt += (
                "Please respond with only the sequence of letters (e.g., ‘BDAC’) "
                "that correctly orders the steps.\n"
            )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def build_prompt(self, line):
        if line['question_type'] == 'sort':
            msgs = self.build_prompt_sort(line)
        else:
            msgs = super().build_prompt(line)
        msgs = self.split_LEGO(msgs)
        return msgs


class VisuLogic(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_URL = {
        'VisuLogic': 'http://opencompass.openxlab.space/utils/VLMEval/VisuLogic.tsv'
    }
    DATASET_MD5 = {
        'VisuLogic': 'b0820b5ec1e01dfe3951927f0def73b6',
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        prompt = ''
        prompt += question
        prompt += "\nSolve the complex visual logical reasoning problem through step-by-step reasoning."
        prompt += "Think about the reasoning process first "
        prompt += "and answer the question following this format: Answer: \\boxed{$LETTER}"

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.visulogic import VisuLogic_acc
        from .utils.multiple_choice import mcq_vanilla_eval

        # model = judge_kwargs['model']
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['exact_matching', 'gpt-4-0125', 'gpt-4-turbo', 'gpt-4o-mini'], model
        name_str_map = {'gpt-4-0125': 'gpt4', 'gpt-4-turbo': 'gpt4-turbo', 'gpt-4o-mini': 'gpt4o-mini'}
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

        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{name_str}.xlsx')

        if osp.exists(storage):
            accuracy_scores = VisuLogic_acc(storage)
        else:
            accuracy_scores = VisuLogic_acc(eval_file)
        combine_score = {**accuracy_scores,}
        combine_score = pd.DataFrame(combine_score)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(combine_score, score_pth)
        return combine_score


class CMMU_MCQ(ImageMCQDataset):
    DATASET_URL = {
        'CMMU_MCQ': 'https://huggingface.co/datasets/Pfei111/CMMU_VAL_MCQ/resolve/main/CMMU_VAL_MCQ.tsv',
    }

    DATASET_MD5 = {
        'CMMU_MCQ': None,
    }


class PathMMU_VAL(ImageMCQDataset):
    DATASET_URL = {
        'PathMMU_VAL': 'https://huggingface.co/datasets/Pfei111/PathMMU/resolve/main/PathMMU_VAL.tsv',
    }

    DATASET_MD5 = {
        'PathMMU_VAL': None,
    }


class PathMMU_TEST(ImageMCQDataset):
    DATASET_URL = {
        'PathMMU_TEST': 'https://huggingface.co/datasets/Pfei111/PathMMU/resolve/main/PathMMU_TEST.tsv',
    }

    DATASET_MD5 = {
        'PathMMU_TEST': None,
    }


class TDBench(ImageMCQDataset):
    DATASET_URL = {
        'tdbench_rot0': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_rot0.tsv',
        'tdbench_rot90': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_rot90.tsv',
        'tdbench_rot180': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_rot180.tsv',
        'tdbench_rot270': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_rot270.tsv',
        'tdbench_cs_zoom': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/case_study_zoom_in.tsv',
        'tdbench_cs_height': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/case_study_height.tsv',
        'tdbench_cs_integrity': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/case_study_integrity.tsv',  # noqa: E501
        'tdbench_cs_depth': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/case_study_depth.tsv',
    }

    DATASET_MD5 = {
        'tdbench_rot0': '98d58436f01ca2bf2f1db1b9bfd7a947',
        'tdbench_rot90': 'd4afebfd0a4776242069e43269779f41',
        'tdbench_rot180': 'd54dd9f418f83ed612b02fd5f42f65c7',
        'tdbench_rot270': 'f95304455582de5635ff10c0400562ac',
        'tdbench_cs_zoom': '2a01618c9c1e7d1a9d86af545e943392',
        'tdbench_cs_height': 'ecbe1c5802e25749558417208164bcb3',
        'tdbench_cs_integrity': '05b2045cae2016f6edc400da48e2df4b',
        'tdbench_cs_depth': '449dbe4b24a43a06a9f680811deae517',
    }

    def evaluate(self, eval_file, **judge_kwargs):
        acc, result_file = self.do_evaluate(eval_file, **judge_kwargs)
        # For case studies (cs_x), do not do rotation eval
        if '_rot' not in self.dataset_name:
            return acc

        from .utils.tdbench import rotational_eval
        re_result = rotational_eval(result_file)
        if re_result is not None and re_result is not False:
            file_addr = osp.abspath(result_file.split('_rot')[0] + '_REresult.csv')
            link_addr = osp.join(osp.dirname(osp.dirname(result_file)), osp.basename(file_addr))
            re_result.to_csv(file_addr, index=True)
            print(tabulate(re_result, headers="keys"))
            if osp.exists(link_addr) or osp.islink(link_addr):
                os.remove(link_addr)
            os.symlink(file_addr, link_addr)

        return acc

    def do_evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, mcq_vanilla_eval
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125', 'gpt-4o-mini']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4', 'gpt-4o-mini': 'gpt4omini'}
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

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # Save evaluation results
        judged_result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}')
        dump(data, judged_result_file)

        acc = report_acc(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc, judged_result_file


class MicroBench(ImageMCQDataset):

    DATASET_URL = {'MicroBench': ''}

    DATASET_PART_URL = {
        'part_1': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_1.tsv',
        'part_2': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_2.tsv',
        'part_3': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_3.tsv',
        'part_4': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_4.tsv',
        'part_5': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_5.tsv',
        'part_6': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_6.tsv',
        'part_7': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_7.tsv',
        'part_8': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_8.tsv',
        'part_9': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_9.tsv',
        'part_10': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_10.tsv',
        'part_11': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_11.tsv',
        'part_12': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_12.tsv',
        'part_13': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_13.tsv',
        'part_14': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_14.tsv',
    }

    def load_data(self, dataset="MicroBench", repo_id="xuxuxuxuxu/MicroBench"):

        dfs = []
        for part_num in range(1, 15):
            part_name = f'part_{part_num}'
            url = self.DATASET_PART_URL[part_name]
            tsv_path = osp.join(LMUDataRoot(), f'microbench_{part_name}.tsv')
            if not osp.exists(tsv_path):
                download_file(url, filename=tsv_path)
            local_path = tsv_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                from ..tools import LOCALIZE
                LOCALIZE(tsv_path, local_path)
            tsv_path = local_path
            # 加载数据
            df = load(tsv_path)
            dfs.append(df)
        # 合并所有数据
        data = pd.concat(dfs, ignore_index=True)
        return data


class MicroVQA(ImageMCQDataset):

    DATASET_URL = {
        'MicroVQA': 'https://opencompass.openxlab.space/utils/VLMEval/MicroVQA.tsv',
    }

    DATASET_MD5 = {
        'MicroVQA': 'd7506438701a2076ec277f8bb3586c1a',
    }


class OmniMedVQA(ImageMCQDataset):

    DATASET_URL = {'OmniMedVQA': ''}

    DATASET_PART_URL = {
        'part_1': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_1.tsv',
        'part_2': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_2.tsv',
        'part_3': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_3.tsv',
        'part_4': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_4.tsv',
        'part_5': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_5.tsv',
        'part_6': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_6.tsv',
        'part_7': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_7.tsv',
        'part_8': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_8.tsv',
        'part_9': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_9.tsv',
        'part_10': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_10.tsv',
        'part_11': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_11.tsv',
        'part_12': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_12.tsv',
        'part_13': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_13.tsv',
        'part_14': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_14.tsv',
        'part_15': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_15.tsv',
        'part_16': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_16.tsv',
        'part_17': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_17.tsv',
        'part_18': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_18.tsv',
        'part_19': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_19.tsv',
        'part_20': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_20.tsv',
        'part_21': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_21.tsv',
        'part_22': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_22.tsv',
        'part_23': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_23.tsv',
        'part_24': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_24.tsv',
        'part_25': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_25.tsv',
        'part_26': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_26.tsv',
        'part_27': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_27.tsv',
        'part_28': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_28.tsv',
        'part_29': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_29.tsv',
        'part_30': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_30.tsv',
    }

    def load_data(self, dataset="OmniMedVQA", repo_id="KKYYKK/OmniMed_VLM"):

        dfs = []
        for part_num in range(1, 15):
            part_name = f'part_{part_num}'
            url = self.DATASET_PART_URL[part_name]
            tsv_path = osp.join(LMUDataRoot(), f'omnimedbench_{part_name}.tsv')
            if not osp.exists(tsv_path):
                download_file(url, filename=tsv_path)
            local_path = tsv_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                from ..tools import LOCALIZE
                LOCALIZE(tsv_path, local_path)
            tsv_path = local_path
            # 加载数据
            df = load(tsv_path)
            dfs.append(df)
        # 合并所有数据
        data = pd.concat(dfs, ignore_index=True)
        return data


class MSEarthMCQ(ImageMCQDataset):

    DATASET_URL = {
        'MSEarthMCQ': '',
    }

    def load_data(self, dataset="MSEarthMCQ", repo_id="MSEarth/MSEarth_MCQ"):
        """
        将HuggingFace parquet后缀dataset生成规范的DataFrame
        需要字段:
            index, question, A, B, C, D, answer, image (base64)
        现有字段:
            query, response, image(PIL.Image Type)
        """
        import re
        import pandas as pd
        from datasets import load_dataset
        # from PIL import Image
        from ..tools import encode_image_to_base64

        # 读取huggingface数据
        hf_ds = load_dataset("MSEarth/MSEarth_MCQ", data_dir="data", split='train')

        # 正则提取
        caption_prefix = r"^\s*<image>\s*Caption:\s*"
        options_prefix = r"\s*Options:\s*"
        option_pat = r"\s*([A-D])\.\s*([^\n]+)"

        records = []
        for idx, sample in enumerate(hf_ds):
            raw_query = sample["query"]
            raw_answer = sample["response"]

            # 解析 answer 只保留字母
            answer_letter = raw_answer.strip()[0].upper()

            # 拆分 query: 去掉 Caption 前缀 + 提取 Options
            q_without_caption = re.sub(caption_prefix, "", raw_query, flags=re.IGNORECASE)
            if "Options:" in q_without_caption:
                question_part, options_part = re.split(
                    options_prefix, q_without_caption, maxsplit=1, flags=re.IGNORECASE)
            else:
                # 极端情况，没有 Options 关键字
                question_part, options_part = q_without_caption, ""

            question_part = question_part.strip()

            options_dict = {"A": "", "B": "", "C": "", "D": ""}
            for m in re.finditer(option_pat, options_part):
                letter, text = m.groups()
                options_dict[letter] = text.strip()

            img_pil = sample["image"]
            img_base = encode_image_to_base64(img_pil)

            rec = {
                "index":    idx,  # noqa E241
                "question": question_part,  # noqa E241
                "A":        options_dict["A"],  # noqa E241
                "B":        options_dict["B"],  # noqa E241
                "C":        options_dict["C"],  # noqa E241
                "D":        options_dict["D"],  # noqa E241
                "answer":   answer_letter,  # noqa E241
                "image":    img_base,  # noqa E241
            }
            records.append(rec)

        df = pd.DataFrame(records)
        df.reset_index(drop=True, inplace=True)

        return df

    def build_prompt(self, line):
        '''
<image>
You are tasked with answering a multiple-choice question about the above given input image.

Caption:
Delineation of hazardous regions for the nine classifications for …
Question:
Which aquifer shows the highest spread of the Fe-Mn hazard?
Options:
A. Aquifer 1
B. Aquifer 2
C. Aquifer 3
D. None of the above
Based on the image, select the correct option (e.g., 'A', 'B', 'C', 'D') or \
directly state the correct option content. Do not give any explanation.
'''

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
        prompt = 'You are tasked with answering a multiple-choice question about the given input image.\n\n'
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Caption:\n {question}\n'
        if len(options):
            prompt += options_prompt
            # prompt += 'Please select the correct answer from the options above. \n'
            prompt += "Based on the image, select the correct option (e.g., 'A', 'B', 'C', 'D') or directly state the correct option content, Do not give any explaination."  # noqa E501

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
