DEFAULT_TYPE_JUDGE_MODELS = {
    'MCQ': 'gpt-4o-mini',
    'Y/N': 'gpt-4o-mini',
    'MCQ_MMMU_Pro': 'gpt-4o-mini',
}


def get_default_judge_model(dataset, dataset_type=None, judge_kwargs=None):
    """Return the default judge model for a dataset, or None if not specified."""
    judge_kwargs = judge_kwargs or {}
    judge_model = getattr(dataset, 'DEFAULT_JUDGE_MODEL', None)
    if isinstance(judge_model, dict):
        for judge_arg, model in judge_model.items():
            if judge_kwargs.get(judge_arg, False):
                return model
        return None
    if judge_model is not None:
        return judge_model

    dataset_type = dataset_type or getattr(dataset, 'TYPE', None)
    return DEFAULT_TYPE_JUDGE_MODELS.get(dataset_type)
