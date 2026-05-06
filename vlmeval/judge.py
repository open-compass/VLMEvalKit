def listinstr(lst, s):
    return any(item in s for item in lst)


def get_default_judge_model(dataset_name, dataset_type, judge_kwargs=None):
    """Return the default judge model for a dataset, or None if not specified."""
    judge_kwargs = judge_kwargs or {}

    if dataset_type in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
        ['moviechat1k', 'mme-reasoning'], dataset_name.lower()
    ):
        if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
            return 'gpt-4o-mini'
        if listinstr(['VisualPuzzles'], dataset_name):
            return 'exact_matching'
        if listinstr(['PuzzleVQA'], dataset_name):
            return 'exact_matching'
        if listinstr(['VisuLogic'], dataset_name):
            return 'exact_matching'
        return 'gpt-4o-mini'

    if listinstr(['MMVet', 'LLaVABench', 'MMBench_Video', 'MMBench-Video'], dataset_name):
        if listinstr(['LLaVABench_KO'], dataset_name):
            return 'gpt-4o-0806'
        return 'gpt-4-turbo'

    if listinstr(['VGRPBench'], dataset_name):
        return 'gpt-4o'

    if listinstr(
        ['MathVista', 'MathVerse', 'MathVision', 'LENS', 'DynaMath', 'VL-RewardBench',
         'LogicVista', 'MOAT', 'OCR_Reasoning', 'VTCBench', 'Asclepius',
         'MMSafetyBench', 'MSSBench', 'SIUO', 'SIUO_GEN', 'XSTest', 'Flames'], dataset_name
    ):
        return 'gpt-4o-mini'

    if listinstr(['OlympiadBench'], dataset_name):
        if judge_kwargs.get('olympiad_use_api_judger', False):
            return 'gpt-4o-mini'
        return None

    if listinstr(
        ['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench',
         'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name
    ):
        return 'gpt-4o'

    if listinstr(['ChartMimic'], dataset_name):
        return 'gpt-4o'
    if listinstr(['VDC'], dataset_name):
        return 'llama31-8b'
    if listinstr(['Video_MMLU_QA', 'Video_MMLU_CAP'], dataset_name):
        return 'qwen-72b'
    if listinstr(['MMVMBench'], dataset_name):
        return 'gpt-4o'
    if listinstr(['CVQA_EN', 'CVQA_LOC'], dataset_name):
        return 'gpt-4.1'
    if listinstr(['M4Bench'], dataset_name):
        return 'gpt-4o'
    if listinstr(['AyaVisionBench'], dataset_name):
        return 'gpt-4.1'
    if listinstr(['MathCanvas'], dataset_name):
        return 'gpt-4.1-2025-04-14'
    if listinstr(['MMReason'], dataset_name):
        return 'gpt-4.1'
    if listinstr(['CoreCognition'], dataset_name):
        return 'gpt-4.1'
    if listinstr(['WorldVQA'], dataset_name):
        return 'gpt-4o-1120'
    if listinstr(['Video-MME'], dataset_name):
        return 'gpt-4o-mini'
    if listinstr(['MaCBench'], dataset_name):
        return 'gpt-4o-mini'
    if listinstr(['SciDocBench'], dataset_name):
        return 'gpt-4o-mini'

    return None
