from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetAliasContext:
    dataset_name: str
    dataset_alias_name: str


def resolve_dataset_alias(dataset_alias_name, data_config=None):
    config = data_config or {}
    dataset_name = dataset_alias_name
    if dataset_alias_name in config:
        value = config[dataset_alias_name]
        if isinstance(value, dict):
            dataset_name = value.get('dataset', dataset_alias_name)
    return DatasetAliasContext(
        dataset_name=dataset_name,
        dataset_alias_name=dataset_alias_name,
    )


def resolve_dataset_alias_name(dataset_name, dataset_alias_name=None):
    return dataset_alias_name or dataset_name
