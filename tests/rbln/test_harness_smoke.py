"""US-006 — Tier 1 harness smoke with a mock model.

Drives the real ``vlmeval.inference.infer_data`` pipeline (dataset load
-> build_prompt -> model.generate -> result-file write) with a
``MockRBLNModel`` that returns canned strings. No NPU, no real model,
no optimum-rbln. Confirms the harness wiring the RBLN wrappers plug into
actually produces a prediction file.

Uses ``ChartQA_TEST`` because its TSV is cached under LMUData and it
needs no LLM judge.
"""

from __future__ import annotations
import os

import pytest

from vlmeval.smp import load
from vlmeval.vlm.base import BaseModel

pytestmark = pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(os.environ.get('LMUData', ''), 'ChartQA_TEST.tsv')
    ),
    reason='ChartQA_TEST.tsv not cached under LMUData',
)


class MockRBLNModel(BaseModel):
    """Canned-response stand-in matching the BaseModel interface that the
    RBLN wrappers implement (``generate_inner``)."""

    is_api = False
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, reply='mock-answer', **kwargs):
        super().__init__()
        self._reply = reply
        self.calls = 0

    def generate_inner(self, message, dataset=None):
        self.calls += 1
        return self._reply


def test_mock_harness_writes_predictions(tmp_path):
    from vlmeval.dataset import build_dataset
    from vlmeval.inference import infer_data

    dataset = build_dataset('ChartQA_TEST')
    # Keep it tiny — emulates `--limit 3`.
    dataset.data = dataset.data.iloc[:3].copy()

    model = MockRBLNModel(reply='42')
    out_file = str(tmp_path / 'MockRBLN_ChartQA_TEST.pkl')

    infer_data(
        model, 'MockRBLN', str(tmp_path), dataset, out_file,
        verbose=False, api_nproc=1,
    )

    assert os.path.isfile(out_file), 'prediction file was not written'
    res = load(out_file)
    assert len(res) == 3
    assert all(isinstance(v, str) and v for v in res.values())
    assert model.calls == 3, f'expected 3 generate calls, got {model.calls}'
