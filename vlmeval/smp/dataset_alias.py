import copy as cp
import re
from dataclasses import dataclass

SAFE_DATASET_ALIAS_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


@dataclass(frozen=True)
class DatasetSpec:
    dataset_alias_name: str
    dataset_name: str
    dataset_class_name: str | None
    build_config: dict
    source: str


@dataclass(frozen=True)
class DatasetAliasContext:
    dataset_name: str
    dataset_alias_name: str
    dataset_class_name: str | None = None


def _non_empty_str(value, field_name, display_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f'`{field_name}` must be a non-empty string for dataset config {display_name}'
        )
    return value


def validate_dataset_alias_name(dataset_alias_name):
    if not isinstance(dataset_alias_name, str) or not dataset_alias_name.strip():
        raise ValueError('Dataset alias must be a non-empty string')
    if not SAFE_DATASET_ALIAS_RE.fullmatch(dataset_alias_name):
        raise ValueError(
            f'Dataset alias {dataset_alias_name} is not a safe filename component. '
            'Use only letters, digits, ".", "_", and "-", and start with a letter or digit.'
        )
    return dataset_alias_name


def _copy_spec(spec, *, dataset_alias_name=None, source=None):
    return DatasetSpec(
        dataset_alias_name=dataset_alias_name or spec.dataset_alias_name,
        dataset_name=spec.dataset_name,
        dataset_class_name=spec.dataset_class_name,
        build_config=cp.deepcopy(spec.build_config),
        source=source or spec.source,
    )


def get_predefined_dataset_spec(name):
    # Delayed import avoids a vlmeval.smp -> vlmeval.dataset -> vlmeval.smp cycle.
    from vlmeval.dataset.video_dataset_config import PREDEFINED_DATASET_SPECS

    spec = PREDEFINED_DATASET_SPECS.get(name)
    if spec is None:
        return None
    return _copy_spec(spec)


def _resolve_preset_config(dataset_alias_name, value):
    preset_name = value.get('preset')
    if not isinstance(preset_name, str) or not preset_name.strip():
        raise ValueError(
            f'`preset` must be a non-empty string for dataset config {dataset_alias_name}'
        )
    if 'class' in value:
        raise ValueError(
            f'`class` cannot be set when using preset for dataset config {dataset_alias_name}'
        )
    if 'dataset' in value:
        raise ValueError(
            f'`dataset` cannot be set when using preset for dataset config {dataset_alias_name}'
        )

    preset_spec = get_predefined_dataset_spec(preset_name)
    if preset_spec is None:
        raise ValueError(
            f'Unknown dataset preset {preset_name} for dataset config {dataset_alias_name}'
        )

    build_config = cp.deepcopy(preset_spec.build_config)
    build_config.update({k: v for k, v in value.items() if k != 'preset'})
    return DatasetSpec(
        dataset_alias_name=dataset_alias_name,
        dataset_name=preset_spec.dataset_name,
        dataset_class_name=preset_spec.dataset_class_name,
        build_config=build_config,
        source='preset_config',
    )


def _resolve_explicit_config(dataset_alias_name, value):
    if value == {}:
        raise ValueError(
            f'Empty dataset config {dataset_alias_name} is not supported. '
            'Use direct --data shortcut or set a preset explicitly.'
        )
    cls_name = _non_empty_str(value.get('class'), 'class', dataset_alias_name)
    dataset_name = _non_empty_str(value.get('dataset'), 'dataset', dataset_alias_name)
    build_config = cp.deepcopy(value)
    return DatasetSpec(
        dataset_alias_name=dataset_alias_name,
        dataset_name=dataset_name,
        dataset_class_name=cls_name,
        build_config=build_config,
        source='explicit_config',
    )


def resolve_dataset_spec(dataset_alias_name, data_config=None):
    dataset_alias_name = validate_dataset_alias_name(dataset_alias_name)
    config = data_config or {}
    if dataset_alias_name in config:
        value = config[dataset_alias_name]
        if not isinstance(value, dict):
            raise ValueError(f'Dataset config {dataset_alias_name} must be a dict')
        if 'preset' in value:
            return _resolve_preset_config(dataset_alias_name, value)
        return _resolve_explicit_config(dataset_alias_name, value)

    predefined_spec = get_predefined_dataset_spec(dataset_alias_name)
    if predefined_spec is not None:
        return predefined_spec

    return DatasetSpec(
        dataset_alias_name=dataset_alias_name,
        dataset_name=dataset_alias_name,
        dataset_class_name=None,
        build_config={'dataset': dataset_alias_name},
        source='direct_dataset',
    )


def resolve_dataset_alias(dataset_alias_name, data_config=None):
    spec = resolve_dataset_spec(dataset_alias_name, data_config)
    return DatasetAliasContext(
        dataset_name=spec.dataset_name,
        dataset_alias_name=spec.dataset_alias_name,
        dataset_class_name=spec.dataset_class_name,
    )


def resolve_dataset_alias_name(dataset_name, dataset_alias_name=None):
    resolved_name = dataset_alias_name or dataset_name
    if resolved_name is None:
        return None
    return validate_dataset_alias_name(resolved_name)
