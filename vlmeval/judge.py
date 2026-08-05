from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_TYPE_JUDGE_MODELS = {
    'MCQ': 'gpt-4o-mini',
    'Y/N': 'gpt-4o-mini',
    'MCQ_MMMU_Pro': 'gpt-4o-mini',
}

_INHERITED_POLICY_KEY = '_default_judge_model_policy'
_MISSING = object()


@dataclass(frozen=True)
class DefaultJudgeModel:
    """A typed default judge/scorer policy.

    Policies stay structured while a concat dataset is being resolved and are
    converted to a scalar ``model`` only immediately before a leaf evaluator is
    called.  ``unresolved`` is an internal fifth state used to distinguish a
    missing declaration from an explicit :meth:`none` declaration.
    """

    kind: str
    value: object = None

    @classmethod
    def fixed(cls, model):
        if not isinstance(model, str) or not model:
            raise TypeError('A fixed default judge model must be a non-empty string.')
        return cls('fixed', model)

    @classmethod
    def exact_matching(cls):
        return cls.fixed('exact_matching')

    @classmethod
    def none(cls):
        return cls('none')

    @classmethod
    def unresolved(cls):
        return cls('unresolved')

    @classmethod
    def when(cls, flag, model):
        return cls.conditional({flag: model})

    @classmethod
    def conditional(cls, conditions):
        if not isinstance(conditions, Mapping) or not conditions:
            raise TypeError('Conditional judge defaults require a non-empty mapping.')
        normalized = []
        for flag, model in conditions.items():
            if not isinstance(flag, str) or not flag:
                raise TypeError('Conditional judge flags must be non-empty strings.')
            normalized.append((flag, cls.normalize(model)))
        return cls('conditional', tuple(normalized))

    @classmethod
    def by_dataset(cls, policies):
        if not isinstance(policies, Mapping):
            raise TypeError('Per-dataset judge defaults require a mapping.')
        normalized = []
        for dataset_name, policy in policies.items():
            if not isinstance(dataset_name, str) or not dataset_name:
                raise TypeError('Dataset names must be non-empty strings.')
            normalized.append((dataset_name, cls.normalize(policy)))
        return cls('by_dataset', tuple(normalized))

    @classmethod
    def normalize(cls, declaration):
        """Normalize class declarations while retaining legacy compatibility.

        Plain strings are fixed policies.  Legacy dictionaries retain their
        historical conditional meaning; per-dataset mappings must therefore be
        declared explicitly with :meth:`by_dataset`.
        """
        if isinstance(declaration, cls):
            return declaration
        if declaration is None:
            return cls.unresolved()
        if isinstance(declaration, str):
            return cls.fixed(declaration)
        if isinstance(declaration, Mapping):
            return cls.conditional(declaration)
        raise TypeError(
            'DEFAULT_JUDGE_MODEL must be a string, mapping, DefaultJudgeModel, or None.'
        )

    @classmethod
    def resolve(cls, dataset, judge_kwargs=None, dataset_type=None):
        """Resolve a dataset (recursively for concat datasets) into a policy."""
        judge_kwargs = dict(judge_kwargs or {})
        explicit_model = judge_kwargs.get('model', _MISSING)
        explicit_override = cls.unresolved()
        if explicit_model is not _MISSING and explicit_model is not None:
            explicit_override = cls.fixed(explicit_model)
        return cls._resolve_dataset(
            dataset=dataset,
            judge_kwargs=judge_kwargs,
            overrides=(),
            fallbacks=(),
            explicit_override=explicit_override,
            dataset_type=dataset_type,
        )

    @classmethod
    def _resolve_dataset(
        cls,
        dataset,
        judge_kwargs,
        overrides,
        fallbacks,
        explicit_override,
        dataset_type=None,
    ):
        dataset_map = getattr(dataset, 'dataset_map', None)
        declaration = cls.normalize(getattr(dataset, 'DEFAULT_JUDGE_MODEL', None))

        if isinstance(dataset_map, Mapping):
            resolved_declaration = declaration._materialize(judge_kwargs)
            children = {}
            for child_name, child in dataset_map.items():
                child_overrides = []
                for override in overrides:
                    override = override._materialize(judge_kwargs)
                    if override.kind == 'unresolved':
                        continue
                    if override.kind == 'by_dataset':
                        child_override = override.for_child(child_name)
                        if child_override.kind != 'unresolved':
                            child_overrides.append(child_override)
                    else:
                        child_overrides.append(override)

                child_fallbacks = fallbacks
                if resolved_declaration.kind == 'by_dataset':
                    child_override = resolved_declaration.for_child(child_name)
                    if child_override.kind != 'unresolved':
                        child_overrides.append(child_override)
                elif resolved_declaration.kind != 'unresolved':
                    child_fallbacks = (resolved_declaration,) + child_fallbacks

                children[child_name] = cls._resolve_dataset(
                    dataset=child,
                    judge_kwargs=judge_kwargs,
                    overrides=tuple(child_overrides),
                    fallbacks=child_fallbacks,
                    explicit_override=explicit_override,
                )
            return cls.by_dataset(children)

        candidates = overrides + (declaration,) + fallbacks
        resolved = cls.unresolved()
        for candidate in candidates:
            candidate = candidate._materialize(judge_kwargs)
            if candidate.kind == 'unresolved':
                continue
            if candidate.kind == 'by_dataset':
                raise ValueError('A leaf dataset cannot resolve to a by-dataset judge policy.')
            resolved = candidate
            break

        if resolved.kind == 'unresolved':
            leaf_type = dataset_type if dataset_type is not None else getattr(dataset, 'TYPE', None)
            if isinstance(leaf_type, str) and leaf_type in DEFAULT_TYPE_JUDGE_MODELS:
                resolved = cls.fixed(DEFAULT_TYPE_JUDGE_MODELS[leaf_type])

        # ``none`` is terminal: the leaf evaluator does not consume a judge
        # model, so a user-selected model must not change its effective policy.
        if resolved.kind != 'none' and explicit_override.kind != 'unresolved':
            return explicit_override
        return resolved

    def _materialize(self, judge_kwargs):
        if self.kind != 'conditional':
            return self
        matched = [(flag, policy) for flag, policy in self.value if judge_kwargs.get(flag, False)]
        if len(matched) > 1:
            flags = ', '.join(flag for flag, _ in matched)
            raise ValueError(f'Multiple conditional default judge models matched: {flags}')
        if not matched:
            return self.unresolved()
        return matched[0][1]._materialize(judge_kwargs)

    def for_child(self, dataset_name):
        """Return a child's policy, or this scalar policy when it applies to all children."""
        if self.kind != 'by_dataset':
            return self
        return dict(self.value).get(dataset_name, self.unresolved())

    def apply(self, judge_kwargs=None):
        """Copy kwargs and apply a resolved scalar policy without mutating the caller."""
        kwargs = dict(judge_kwargs or {})
        policy = self._materialize(kwargs)
        if policy.kind == 'by_dataset':
            raise ValueError('A by-dataset judge policy cannot be applied to a leaf evaluator.')
        if policy.kind == 'fixed':
            if kwargs.get('model') is None:
                kwargs['model'] = policy.value
        elif policy.kind == 'none':
            # A terminal no-judge policy must also remove an explicit model
            # inherited from a concat parent before calling the leaf evaluator.
            kwargs.pop('model', None)
        elif kwargs.get('model') is None:
            kwargs.pop('model', None)
        return kwargs

    def to_status_value(self):
        """Return a JSON-serializable scalar or per-dataset representation."""
        if self.kind == 'fixed':
            return self.value
        if self.kind in ('none', 'unresolved'):
            return None
        if self.kind == 'by_dataset':
            return {name: policy.to_status_value() for name, policy in self.value}
        raise ValueError('A conditional judge policy must be resolved before serialization.')


def resolve_judge_policy(dataset, judge_kwargs=None):
    """Return a resolved policy and a clean copy of its runtime kwargs.

    The private inherited-policy key is used only between nested concat
    evaluators.  It is consumed here and is never forwarded to a leaf.
    """
    kwargs = dict(judge_kwargs or {})
    inherited_policy = kwargs.pop(_INHERITED_POLICY_KEY, None)
    if inherited_policy is None:
        policy = DefaultJudgeModel.resolve(dataset, kwargs)
    elif isinstance(inherited_policy, DefaultJudgeModel):
        policy = inherited_policy
    else:
        raise TypeError('The inherited default judge policy has an invalid type.')
    return policy, kwargs


def resolve_judge_kwargs(dataset, judge_kwargs=None, policy=None):
    """Return an independent kwargs dict resolved for one child dataset."""
    kwargs = dict(judge_kwargs or {})
    inherited_policy = kwargs.pop(_INHERITED_POLICY_KEY, None)
    if policy is None:
        policy = inherited_policy or DefaultJudgeModel.resolve(dataset, kwargs)
    if not isinstance(policy, DefaultJudgeModel):
        raise TypeError('The default judge policy has an invalid type.')

    if kwargs.get('model') is None:
        kwargs.pop('model', None)
    if policy.kind == 'by_dataset':
        kwargs[_INHERITED_POLICY_KEY] = policy
        return kwargs
    return policy.apply(kwargs)


def resolve_judge_config(dataset, judge_kwargs=None, dataset_type=None):
    """Resolve top-level evaluation kwargs and their status/cache value.

    Leaf defaults are applied to a copied kwargs dictionary.  Composite
    policies remain structured until concat evaluators dispatch their children,
    while the returned status value always describes the complete resolved
    policy in a JSON-serializable form.
    """
    kwargs = dict(judge_kwargs or {})
    if kwargs.get('model') is None:
        kwargs.pop('model', None)
    policy = DefaultJudgeModel.resolve(dataset, kwargs, dataset_type=dataset_type)
    if policy.kind != 'by_dataset':
        kwargs = policy.apply(kwargs)
    return kwargs, policy.to_status_value()


def get_default_judge_model(dataset, dataset_type=None, judge_kwargs=None):
    """Return a scalar default for legacy leaf-dataset callers.

    Concat datasets intentionally return ``None`` here: their complete policy
    is dispatched by their evaluator instead of being collapsed into one model.
    """
    policy = DefaultJudgeModel.resolve(dataset, judge_kwargs, dataset_type)
    value = policy.to_status_value()
    return value if isinstance(value, str) else None
