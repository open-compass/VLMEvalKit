import warnings

from .image_base import ImageBaseDataset
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
        # Other Benchmarks
        'CCBench': 'https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv',
        'AI2D_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv',
        'MMStar': 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv',
        'RealWorldQA': 'https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv',
        'MLLMGuard_DS': 'https://opencompass.openxlab.space/utils/VLMEval/MLLMGuard_DS.tsv',
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
        # Other Benchmarks
        'CCBench': 'f5dde47f24dc5a6fb6e595b409b466ac',
        'AI2D_TEST': '0f593e0d1c7df9a3d69bf1f947e71975',
        'MMStar': 'e1ecd2140806c1b1bbf54b43372efb9e',
        'RealWorldQA': '92321028d2bc29040284b6674721e48f',
        'MLLMGuard_DS': '975fc0dd7119386e198c37d71e274b3f',
    }

    def post_build(self, dataset):
        img_root_map = {
            'MMBench_DEV_EN': 'MMBench', 'MMBench_TEST_EN': 'MMBench',
            'MMBench_DEV_CN': 'MMBench', 'MMBench_TEST_CN': 'MMBench',
            'MMBench': 'MMBench', 'MMBench_CN': 'MMBench',
            'MMBench_DEV_EN_V11': 'MMBench_V11', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_DEV_CN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_V11',
            'MMBench_V11': 'MMBench', 'MMBench_CN_V11': 'MMBench',
        }
        if dataset in img_root_map:
            ROOT = LMUDataRoot()
            self.img_root = osp.join(ROOT, 'images', img_root_map[dataset])

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


class MMMUDataset(ImageMCQDataset):

    DATASET_URL = {
        'MMMU_DEV_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv',
        'MMMU_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv',
    }

    DATASET_MD5 = {
        'MMMU_DEV_VAL': '521afc0f3bf341e6654327792781644d',
        'MMMU_TEST': 'c19875d11a2d348d07e5eb4bdf33166d',
    }

    def post_build(self, dataset):
        self.img_root = osp.join(LMUDataRoot(), 'images', 'MMMU')

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
