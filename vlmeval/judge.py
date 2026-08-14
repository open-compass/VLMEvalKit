from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

DEFAULT_TYPE_JUDGE_MODELS = {
    'MCQ': 'gpt-4o-mini',
    'Y/N': 'gpt-4o-mini',
    'MCQ_MMMU_Pro': 'gpt-4o-mini',
}

_NO_JUDGE = object()


@dataclass(frozen=True)
class DefaultJudgeModel:
    """A benchmark's fixed, conditional, or explicitly disabled default judge."""

    model: Optional[str] = None
    flag: Optional[str] = None
    disabled: bool = False

    def __post_init__(self):
        if self.disabled:
            if self.model is not None or self.flag is not None:
                raise ValueError('A disabled judge default cannot define a model or flag.')
            return
        if not isinstance(self.model, str) or not self.model:
            raise TypeError('A judge default must define a non-empty model string.')
        if self.flag is not None and (not isinstance(self.flag, str) or not self.flag):
            raise TypeError('A conditional judge flag must be a non-empty string.')

    @classmethod
    def fixed(cls, model):
        return cls(model=model)

    @classmethod
    def exact_matching(cls):
        return cls.fixed('exact_matching')

    @classmethod
    def none(cls):
        return cls(disabled=True)

    @classmethod
    def when(cls, flag, model):
        return cls(model=model, flag=flag)

    def resolve(self, judge_kwargs=None):
        """Return the selected model, or None when its condition is inactive."""
        if self.disabled:
            return None
        if self.flag is not None and not (judge_kwargs or {}).get(self.flag, False):
            return None
        return self.model


def _validate_model(model):
    if model is None:
        return model
    if isinstance(model, str) and model:
        return model
    raise TypeError('A default judge model must be a non-empty string or None.')


def _declared_default(dataset, judge_kwargs):
    """Resolve a dataset's declaration without applying type fallback."""
    declaration = getattr(dataset, 'DEFAULT_JUDGE_MODEL', None)
    if isinstance(declaration, DefaultJudgeModel):
        if declaration.disabled:
            return _NO_JUDGE
        return declaration.resolve(judge_kwargs)
    if not isinstance(declaration, Mapping):
        return _validate_model(declaration)

    # Dictionaries retain their historical conditional meaning.  When more
    # than one flag is enabled, the first declaration wins as it did before.
    for flag, model in declaration.items():
        if not isinstance(flag, str) or not flag:
            raise TypeError('Conditional judge flags must be non-empty strings.')
        if judge_kwargs.get(flag, False):
            return _validate_model(model)
    return None


def _resolve_leaf_model(dataset, judge_kwargs):
    default = _declared_default(dataset, judge_kwargs)

    # A rule-only evaluator must not receive an explicit model inherited from
    # a concat invocation, nor fall through to the generic MCQ/Y/N default.
    if default is _NO_JUDGE:
        return _NO_JUDGE

    explicit_model = judge_kwargs.get('model')
    if explicit_model is not None:
        return _validate_model(explicit_model)
    if default is not None:
        return default
    return DEFAULT_TYPE_JUDGE_MODELS.get(getattr(dataset, 'TYPE', None))


def _dataset_map(dataset):
    children = getattr(dataset, 'dataset_map', None)
    return children if isinstance(children, Mapping) else None


def _status_value(dataset, judge_kwargs):
    children = _dataset_map(dataset)
    if children is not None:
        return {
            name: _status_value(child, judge_kwargs)
            for name, child in children.items()
        }

    model = _resolve_leaf_model(dataset, judge_kwargs)
    return None if model is _NO_JUDGE else model


def _apply_leaf_model(kwargs, model):
    if model is None or model is _NO_JUDGE:
        kwargs.pop('model', None)
    else:
        kwargs['model'] = model
    return kwargs


def resolve_judge_kwargs(dataset, judge_kwargs=None):
    """Return an independent kwargs dictionary for one dataset evaluation."""
    kwargs = dict(judge_kwargs or {})
    if kwargs.get('model') is None:
        kwargs.pop('model', None)

    # A nested concat resolves each of its own children when it dispatches
    # them.  Keeping only an explicit model here is sufficient for recursion.
    if _dataset_map(dataset) is not None:
        return kwargs
    return _apply_leaf_model(kwargs, _resolve_leaf_model(dataset, kwargs))


def resolve_judge_config(dataset, judge_kwargs=None):
    """Return top-level runtime kwargs and a JSON-serializable judge status."""
    kwargs = dict(judge_kwargs or {})
    if kwargs.get('model') is None:
        kwargs.pop('model', None)

    children = _dataset_map(dataset)
    if children is not None:
        return kwargs, _status_value(dataset, kwargs)

    model = _resolve_leaf_model(dataset, kwargs)
    return _apply_leaf_model(kwargs, model), None if model is _NO_JUDGE else model
