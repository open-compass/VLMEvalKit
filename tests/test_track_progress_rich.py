import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def mp_util(monkeypatch):
    """Import task execution without optional model and dataset dependencies."""
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('vlmeval')
    package.__path__ = [str(root / 'vlmeval')]
    smp = types.ModuleType('vlmeval.smp')
    smp.dump = lambda data, path: Path(path).write_text(json.dumps(data))
    smp.load = lambda path: json.loads(Path(path).read_text())
    monkeypatch.setitem(sys.modules, 'vlmeval', package)
    monkeypatch.setitem(sys.modules, 'vlmeval.smp', smp)
    spec = importlib.util.spec_from_file_location('vlmeval.utils.mp_util', root / 'vlmeval/utils/mp_util.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    'task_factory',
    [list, tuple, iter, lambda tasks: (task for task in tasks), lambda tasks: map(tuple, tasks)],
    ids=['list', 'tuple', 'iterator', 'generator', 'map'],
)
def test_iterable_tasks_preserve_result_order(mp_util, task_factory):
    tasks = task_factory([(2, 3), (3, 2), (2, 0)])

    assert mp_util.track_progress_rich(pow, tasks, nproc=2) == [8, 9, 1]


def test_generator_tasks_save_results_with_keys(mp_util, tmp_path):
    tasks = ({'base': 2, 'exp': exponent} for exponent in range(3))
    path = tmp_path / 'results.json'

    results = mp_util.track_progress_rich(pow, tasks, nproc=2, save=str(path), keys=['a', 'b', 'c'])

    assert results == [1, 2, 4]
    assert json.loads(path.read_text()) == {'a': 1, 'b': 2, 'c': 4}


def test_empty_task_iterator(mp_util):
    assert mp_util.track_progress_rich(pow, iter(()), nproc=2) == []
