from ..smp import listinstr

dataset_URLs = {
    'MMBench_DEV_EN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv", 
    'MMBench_TEST_EN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv", 
    'MMBench_DEV_CN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv", 
    'MMBench_TEST_CN': "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv", 
    "MMBench": "https://opencompass.openxlab.space/utils/VLMEval/MMBench.tsv",  # Link Invalid, Internal Only
    "MMBench_CN": "https://opencompass.openxlab.space/utils/VLMEval/MMBench_CN.tsv",    # Link Invalid, Internal Only
    'CCBench': "https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv", 
    'MME': "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv", 
    'SEEDBench_IMG': "https://opencompass.openxlab.space/utils/VLMEval/SEEDBench_IMG.tsv", 
    "CORE_MM": "https://opencompass.openxlab.space/utils/VLMEval/CORE_MM.tsv",
    "MMVet": "https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv",
    "COCO_VAL": "https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL.tsv",
    "OCRVQA_TEST": "https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv",
    "OCRVQA_TESTCORE": "https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv",
    'TextVQA_VAL': "https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv",
    "MMMU_DEV_VAL": "https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv",
    "MMMU_TEST": "https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv",
    "MathVista_MINI": "https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv",
    'ChartQA_VALTEST_HUMAN': "https://opencompass.openxlab.space/utils/VLMEval/ChartQA_VALTEST_HUMAN.tsv",
    'ScienceQA_VAL': "https://opencompass.openxlab.space/utils/VLMEval/ScienceQA_VAL.tsv",
    'ScienceQA_TEST': "https://opencompass.openxlab.space/utils/VLMEval/ScienceQA_TEST.tsv",
    'HallusionBench': "https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv",
    "DocVQA_VAL": "https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv",
    'AI2D_TEST': "https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv",
    "LLaVABench": "https://opencompass.openxlab.space/utils/VLMEval/LLaVABench.tsv",
    "OCRBench": 'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv',
}

dataset_md5_dict = {
    'MMBench_DEV_EN': "b6caf1133a01c6bb705cf753bb527ed8", 
    'MMBench_TEST_EN': "6939fadb0ce626fefc0bdc9c64efc528", 
    'MMBench_DEV_CN': "08b8fc3324a5ed74155350f57be69fbd", 
    'MMBench_TEST_CN': "7e1239baf0ee4c8b513e19705a0f317e", 
    "MMBench": "4115aea3383f3dd0083be6a633e0f820",  # Link Invalid, Internal Only
    "MMBench_CN": "2e053ffc90ea598b1feae13c36dc13ee",    # Link Invalid, Internal Only
    'CCBench': "1de88b4257e7eee3f60b18d45eda6f07", 
    'MME': "b36b43c3f09801f5d368627fb92187c3", 
    'SEEDBench_IMG': "68017231464752261a2526d6ca3a10c0", 
    "CORE_MM": "8a8da2f2232e79caf98415bfdf0a202d",
    "MMVet": "f400d7f513a585a0f218cbd6882e0671",
    'COCO_VAL': "72a5079dead060269ac222c5aa5128af",
    'OCRVQA_TEST': 'ca46a6d74b403e9d6c0b670f6fc00db9',
    'OCRVQA_TESTCORE': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
    'TextVQA_VAL': 'b233b31f551bbf4056f2f955da3a92cd',
    'MMMU_DEV_VAL': "521afc0f3bf341e6654327792781644d",
    'MMMU_TEST': "c19875d11a2d348d07e5eb4bdf33166d",
    'MathVista_MINI': 'f199b98e178e5a2a20e7048f5dcb0464',
    'ChartQA_VALTEST_HUMAN':'2c90a4133408a21d57fb2ea26f77bbfc',
    'ScienceQA_VAL': '96320d05e142e585e7204e72affd29f3',
    'ScienceQA_TEST': 'e42e9e00f9c59a80d8a5db35bc32b71f',
    'HallusionBench': '0c23ac0dc9ef46832d7a24504f2a0c7c',
    "DocVQA_VAL": 'ee0d8ae5527439438d08e154ef65d735',
    "AI2D_TEST": "0f593e0d1c7df9a3d69bf1f947e71975",
    "LLaVABench": "d382a093f749a697820d3dadd61c8428",
    "OCRBench": 'e953d98a987cc6e26ef717b61260b778',
}

img_root_map = {k: k for k in dataset_URLs}
img_root_map.update({
    'MMBench_DEV_EN': "MMBench", 
    'MMBench_TEST_EN': "MMBench", 
    'MMBench_DEV_CN': "MMBench", 
    'MMBench_TEST_CN': "MMBench", 
    "MMBench_CN": "MMBench",    # Link Invalid, Internal Only
    'COCO_VAL':'COCO',
    'OCRVQA_TEST': 'OCRVQA',
    'OCRVQA_TESTCORE': 'OCRVQA',
    'TextVQA_VAL': 'TextVQA',
    'MMMU_DEV_VAL': 'MMMU',
    "MMMU_TEST": "MMMU", 
    'MathVista_MINI': 'MathVista',
    'ChartQA_VALTEST_HUMAN': 'ChartQA',
    'HallusionBench': 'Hallusion',
    'DocVQA_VAL': 'DocVQA',
    "OCRBench": 'OCRBench',
})

assert set(dataset_URLs) == set(img_root_map) == set(dataset_md5_dict)

def DATASET_TYPE(dataset):
    dataset = dataset.lower()
    if listinstr(['mmbench', 'seedbench', 'ccbench', 'mmmu', 'scienceqa', 'ai2d'], dataset):
        return 'multi-choice'
    elif listinstr(['mme', 'hallusion'], dataset):
        return 'Y/N'
    elif 'coco' in dataset:
        return 'Caption'
    elif listinstr(['ocrvqa', 'textvqa', 'chartqa', 'mathvista', 'docvqa', 'llavabench', 'mmvet', 'OCRBench'], dataset):
        return 'VQA'
    else:
        return 'QA'

def abbr2full(s):
    datasets = [x for x in img_root_map]
    ins = [s in d for d in datasets]
    if sum(ins) == 1:
        for d in datasets:
            if s in d:
                return d
    else:
        return None
