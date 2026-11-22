import warnings

import pytest

from vlmeval.dataset.utils import judge_util
from vlmeval.dataset.utils.megabench.scoring.vlm_as_judge import OpenAIVLMJudger


@pytest.mark.parametrize('model', [
    'gpt-4o-0806', 'gpt-4o-1120', 'gpt-4o-mini', 'gpt-4.1',
    'gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14',
    'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18',
])
def test_build_judge_warns_for_retiring_models(monkeypatch, model):
    import vlmeval.api

    monkeypatch.delenv('LOCAL_LLM', raising=False)
    monkeypatch.setattr(judge_util, 'load_env', lambda: None)
    monkeypatch.setattr(vlmeval.api, 'OpenAIWrapper', lambda model, **kwargs: model)

    with pytest.warns(UserWarning, match='January 14, 2027.*gpt-6-luna'):
        judge_util.build_judge(model=model)


@pytest.mark.parametrize('model', ['gpt-4o', 'gpt-5.2', 'gpt-6-luna'])
def test_build_judge_does_not_warn_for_other_models(monkeypatch, model):
    import vlmeval.api

    monkeypatch.delenv('LOCAL_LLM', raising=False)
    monkeypatch.setattr(judge_util, 'load_env', lambda: None)
    monkeypatch.setattr(vlmeval.api, 'OpenAIWrapper', lambda model, **kwargs: model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        judge_util.build_judge(model=model)
    assert not caught


def test_local_override_checks_actual_model(monkeypatch):
    import vlmeval.api

    monkeypatch.setenv('LOCAL_LLM', 'gpt-6-luna')
    monkeypatch.setattr(judge_util, 'load_env', lambda: None)
    monkeypatch.setattr(vlmeval.api, 'OpenAIWrapper', lambda model, **kwargs: model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert judge_util.build_judge(model='gpt-4o-mini') == 'gpt-6-luna'
    assert not caught


def test_megabench_judge_warns():
    with pytest.warns(UserWarning, match='January 14, 2027.*gpt-6-luna'):
        judge = OpenAIVLMJudger(None)
    assert judge.model == 'gpt-4o-2024-08-06'
