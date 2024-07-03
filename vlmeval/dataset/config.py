from ..smp import listinstr

dataset_URLs = {
    'CORE_MM': 'https://opencompass.openxlab.space/utils/VLMEval/CORE_MM.tsv',
    'MMVet': 'https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv',
    'COCO_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL.tsv',
    'OCRVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv',
    'OCRVQA_TESTCORE': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
    'TextVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
    'MMMU_DEV_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv',
    'MMMU_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv',
    'MathVista_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv',
    'DocVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
    'DocVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
    'InfoVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
    'InfoVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
    'LLaVABench': 'https://opencompass.openxlab.space/utils/VLMEval/LLaVABench.tsv',
    'OCRBench': 'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv',
    'ChartQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
}

dataset_md5_dict = {
    'CORE_MM': '8a8da2f2232e79caf98415bfdf0a202d',
    'MMVet': '748aa6d4aa9d4de798306a63718455e3',
    'COCO_VAL': '72a5079dead060269ac222c5aa5128af',
    'OCRVQA_TEST': 'ca46a6d74b403e9d6c0b670f6fc00db9',
    'OCRVQA_TESTCORE': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
    'TextVQA_VAL': 'b233b31f551bbf4056f2f955da3a92cd',
    'MMMU_DEV_VAL': '521afc0f3bf341e6654327792781644d',
    'MMMU_TEST': 'c19875d11a2d348d07e5eb4bdf33166d',
    'MathVista_MINI': 'f199b98e178e5a2a20e7048f5dcb0464',
    'DocVQA_VAL': 'd5ee77e1926ff10690d469c56b73eabf',
    'DocVQA_TEST': '6a2f28cac26ef2d3447374e8c6f6c8e9',
    'InfoVQA_VAL': '2342e9c225222f0ef4dec545ebb126fe',
    'InfoVQA_TEST': 'df535bf51b88dc9718252c34131a6227',
    'LLaVABench': 'd382a093f749a697820d3dadd61c8428',
    'OCRBench': 'e953d98a987cc6e26ef717b61260b778',
    'ChartQA_TEST': 'c902e0aa9be5582a7aad6dcf52734b42',
}

img_root_map = {k: k for k in dataset_URLs}
img_root_map.update({
    'COCO_VAL': 'COCO',
    'OCRVQA_TEST': 'OCRVQA',
    'OCRVQA_TESTCORE': 'OCRVQA',
    'TextVQA_VAL': 'TextVQA',
    'MMMU_DEV_VAL': 'MMMU',
    'MMMU_TEST': 'MMMU',
    'MathVista_MINI': 'MathVista',
    'HallusionBench': 'Hallusion',
    'DocVQA_VAL': 'DocVQA',
})


def DATASET_TYPE(dataset):
    # Dealing with Custom Dataset
    dataset = dataset.lower()
    if 'coco' in dataset:
        return 'Caption'
    elif listinstr(['ocrvqa', 'textvqa', 'chartqa', 'mathvista', 'docvqa', 'infovqa', 'llavabench',
                    'mmvet', 'ocrbench', 'mllmguard'], dataset):
        return 'VQA'
    else:
        if dataset not in dataset_URLs:
            import warnings
            warnings.warn(f"Dataset {dataset} not found in dataset_URLs, will use 'multi-choice' as the default TYPE.")
            return 'multi-choice'
        else:
            return 'QA'
