"""US-001 — lazy-import invariant.

``vlmeval/vlm/rbln/AGENTS.md`` requires that importing the RBLN package
never pulls ``optimum.rbln`` (or the lower-level ``rebel`` /
``torch_neuronx`` runtime) into the process: every optimum-rbln import
lives inside a wrapper method so the package imports cleanly on machines
without the RBLN runtime installed.

We verify this in a *fresh* subprocess — the pytest process itself may
have imported optimum.rbln via another test, so an in-process
``sys.modules`` check would be unreliable.
"""

from __future__ import annotations
import subprocess
import sys

import pytest

from .conftest import REPO_ROOT

_FORBIDDEN = ('optimum.rbln', 'torch_neuronx', 'rebel')


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _assert_clean_import(import_line: str):
    """Run ``import_line`` in a fresh interpreter and assert none of the
    RBLN runtime modules got pulled in at module-load time."""
    forbidden = ', '.join(repr(m) for m in _FORBIDDEN)
    code = (
        'import sys\n'
        f'{import_line}\n'
        f'bad = [m for m in ({forbidden},) if m in sys.modules]\n'
        'assert not bad, "RBLN runtime imported at module load: %s" % bad\n'
        'print("OK")\n'
    )
    res = _run(code)
    assert res.returncode == 0, (
        f'subprocess failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}'
    )
    assert 'OK' in res.stdout


@pytest.mark.parametrize('import_line', [
    'import vlmeval.vlm.rbln  # noqa: F401',
    'from vlmeval.vlm.rbln import *  # noqa: F401,F403',
], ids=['package', 'star'])
def test_import_does_not_load_runtime(import_line):
    _assert_clean_import(import_line)


def test_wrapper_classes_are_exported():
    """All 11 concrete wrapper families + base resolve from the package."""
    code = (
        'import vlmeval.vlm.rbln as m\n'
        'names = ["RBLNVLMBase","RBLNQwen2VL","RBLNQwen3VL","RBLNLlava",\n'
        '         "RBLNLlavaNext","RBLNIdefics3","RBLNGemma3","RBLNPixtral",\n'
        '         "RBLNPaliGemma","RBLNPaliGemma2","RBLNBlip2","RBLNCosmosReason1"]\n'
        'missing = [n for n in names if not hasattr(m, n)]\n'
        'assert not missing, missing\n'
        'print("OK")\n'
    )
    res = _run(code)
    assert res.returncode == 0, (
        f'subprocess failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}'
    )
    assert 'OK' in res.stdout
