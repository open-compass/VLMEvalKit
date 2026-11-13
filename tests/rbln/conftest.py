"""Shared fixtures and markers for the RBLN backend test suite.

These tests verify the hardware-independent logic of the RBLN wrappers
(prompt construction, auto-dispatch, config merging, registry integrity)
without loading any optimum-rbln model or touching the NPU. The few
hardware-gated checks use the markers registered below so they skip
cleanly when a prerequisite (NPU device, video decoder, LLM judge) is
absent.
"""

from __future__ import annotations
import os

import pytest

# Repository root = two levels up from this file (tests/rbln/conftest.py).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'requires_npu: test needs a real RBLN NPU device (/dev/rbln*).',
    )
    config.addinivalue_line(
        'markers',
        'requires_video_deps: test needs a working video decoder (decord).',
    )
    config.addinivalue_line(
        'markers',
        'requires_judge: test needs an external LLM judge (OPENAI_API_KEY / LOCAL_LLM).',
    )


def _npu_present() -> bool:
    # RBLN NPUs appear as /dev/rbln0../dev/rblnN (not /dev/rsd*).
    try:
        return any(n.startswith('rbln') for n in os.listdir('/dev'))
    except OSError:
        return False


@pytest.fixture(scope='session')
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope='session')
def npu_present() -> bool:
    return _npu_present()
