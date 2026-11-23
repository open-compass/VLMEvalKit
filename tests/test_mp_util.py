import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def mp_util(monkeypatch):
    """Load the worker helper without importing optional model dependencies."""
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('vlmeval')
    package.__path__ = [str(root / 'vlmeval')]
    smp = types.ModuleType('vlmeval.smp')
    smp.dump = None
    smp.load = None
    monkeypatch.setitem(sys.modules, 'vlmeval', package)
    monkeypatch.setitem(sys.modules, 'vlmeval.smp', smp)
    spec = importlib.util.spec_from_file_location('vlmeval.utils.mp_util', root / 'vlmeval/utils/mp_util.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('quota', [-1, 50000, 250000])
def test_auto_workers_execute_tasks_with_cgroup_quota(mp_util, tmp_path, monkeypatch, quota):
    (tmp_path / 'cpu.cfs_period_us').write_text('100000')
    (tmp_path / 'cpu.cfs_quota_us').write_text(str(quota))
    monkeypatch.setattr(mp_util, 'Path', lambda path: tmp_path / Path(path).name)
    monkeypatch.setattr(os, 'cpu_count', lambda: 2)

    assert mp_util.track_progress_rich(pow, [(2, 0), (2, 1), (2, 2)], nproc=None) == [1, 2, 4]
