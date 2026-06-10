"""Structured registry entrypoints for datasets, metrics, and tasks."""


class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(item):
            if name in self._registry:
                raise ValueError(f"Item {name} already registered.")
            self._registry[name] = item
            return item

        return decorator

    def get(self, name):
        if name not in self._registry:
            raise ValueError(f"Item {name} not found in registry.")
        return self._registry[name]

    def list_items(self):
        return list(self._registry.keys())


EVAL_TASK_REGISTRY = Registry()
METRIC_REGISTRY = Registry()
DATASET_REGISTRY = Registry()


def load_default_registrations():
    # Import side-effect modules once so decorator-based registrations are populated.
    import src.dataset  # noqa: F401
    import src.metrics  # noqa: F401
    import src.core.pipeline_eval  # noqa: F401


def describe_registries():
    load_default_registrations()
    return {
        "datasets": sorted(DATASET_REGISTRY.list_items()),
        "metrics": sorted(METRIC_REGISTRY.list_items()),
        "tasks": sorted(EVAL_TASK_REGISTRY.list_items()),
    }


__all__ = [
    "Registry",
    "DATASET_REGISTRY",
    "EVAL_TASK_REGISTRY",
    "METRIC_REGISTRY",
    "describe_registries",
    "load_default_registrations",
]
