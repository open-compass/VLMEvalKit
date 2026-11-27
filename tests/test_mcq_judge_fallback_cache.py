import os
from unittest import mock

import pandas as pd
import pytest

from vlmeval.dataset import image_mcq
from vlmeval.dataset.image_mcq import CustomMCQDataset


class FakeJudge:

    def __init__(self, up):
        self.up = up

    def working(self):
        return self.up

    def generate(self, prompt, **kwargs):
        return 'B'


@pytest.mark.parametrize('prediction', [
    'It is a small feline sitting on a sofa.',
    'Looks like a kitten to me.',
])
def test_judge_rerun_after_exact_matching_fallback(tmp_path, monkeypatch, prediction):
    # A judge that fails its startup check makes evaluate() fall back to exact matching.
    # Those results must not be cached under the judge's name, or a rerun with a working
    # judge keeps the exact-matching misses.
    monkeypatch.setenv('LMUData', str(tmp_path))
    question = dict(question='What animal is in the picture?', A='dog', B='cat', C='horse', D='bird')
    rows = [dict(index=i, answer='B', **question) for i in range(4)]
    pd.DataFrame(rows).to_csv(tmp_path / 'ReproMCQ.tsv', sep='\t', index=False)
    pred_file = str(tmp_path / 'Model_ReproMCQ.xlsx')
    pd.DataFrame([dict(r, prediction=prediction) for r in rows]).to_excel(pred_file, index=False)
    dataset = CustomMCQDataset('ReproMCQ')

    with mock.patch.object(image_mcq, 'build_judge', return_value=FakeJudge(up=False)):
        acc = dataset.evaluate(pred_file, model='gpt-4o-mini', nproc=1)
    assert acc['Overall'][0] == 0
    assert not os.path.exists(str(tmp_path / 'Model_ReproMCQ_gpt-4o-mini_result.pkl'))

    with mock.patch.object(image_mcq, 'build_judge', return_value=FakeJudge(up=True)):
        acc = dataset.evaluate(pred_file, model='gpt-4o-mini', nproc=1)
    assert acc['Overall'][0] == 1
