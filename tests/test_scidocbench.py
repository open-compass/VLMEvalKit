import json
from types import SimpleNamespace

import pandas as pd

from vlmeval.dataset import scidocbench


def _intermediate_file(path, suffix, extension=None):
    extension = extension or 'tsv'
    return f'{path}{suffix}.{extension}'


def test_answer_worker_does_not_run_reasoning(monkeypatch):
    monkeypatch.setattr(
        scidocbench,
        'eval_reasoning',
        lambda *args: (_ for _ in ()).throw(AssertionError('unexpected call')),
    )
    item = json.dumps({
        'pair_qid': 'p11b_001',
        'prediction': '<think>hidden</think>Yes',
        'answer': '"Yes"',
        '_enable_reasoning_diagnostic': True,
    })

    score, note = scidocbench._eval_one_item(None, item)

    assert score == 1.0
    assert 'reasoning' not in note.lower()


def test_reasoning_worker_runs_only_the_reasoning_judge(monkeypatch):
    judge = object()
    monkeypatch.setattr(
        scidocbench,
        'eval_reasoning',
        lambda model, prediction, question: (
            0.75 if (model, prediction, question) == (judge, 'full output', 'question') else 0.0,
            'reasoning note',
        ),
    )
    item = json.dumps({
        'pair_qid': 'euler-stratification_001',
        'prediction': 'full output',
        'question': 'question',
    })

    assert scidocbench._eval_one_reasoning(judge, item) == (0.75, 'reasoning note')


def test_cache_reuse_depends_only_on_failure_note():
    assert scidocbench._cache_result_is_reusable(
        (2.0, 'completed'), scidocbench.ANSWER_EVAL_FAILURE_MARKERS)
    assert scidocbench._cache_result_is_reusable(
        (-1.0, 'completed'), scidocbench.REASONING_EVAL_FAILURE_MARKERS)
    assert not scidocbench._cache_result_is_reusable(
        (0.0, 'Eval error: temporary'),
        scidocbench.ANSWER_EVAL_FAILURE_MARKERS)
    assert not scidocbench._cache_result_is_reusable(
        (0.0, 'Reasoning eval error: temporary'),
        scidocbench.REASONING_EVAL_FAILURE_MARKERS)
    assert not scidocbench._cache_result_is_reusable(
        (0.0, scidocbench.INFERENCE_FAILURE_NOTE),
        scidocbench.ANSWER_EVAL_FAILURE_MARKERS)
    assert not scidocbench._cache_result_is_reusable(
        (0.0, None, 'completed'), scidocbench.ANSWER_EVAL_FAILURE_MARKERS)
    assert scidocbench._scored_result_is_reusable(
        pd.DataFrame({'eval_note': ['completed']}), include_reasoning=True)


def test_reasoning_diagnostic_uses_separate_progress_cache(monkeypatch):
    eval_file = 'predictions.tsv'
    data = pd.DataFrame([{
        'index': '0',
        'qid': 'euler-stratification_001__en_all_first',
        'pair_qid': 'euler-stratification_001',
        'partition': 'en_all_first',
        'mode': 'all_first',
        'language': 'en',
        'category': 'A1',
        'eval_method': 'judge',
        'prediction': '{"answer": "Yes", "reasoning": "because"}',
        'answer': '{"answer": "Yes"}',
        'question': 'Is it symmetric?',
        'judge_prompt': '',
    }])
    saved = {}
    progress_calls = []
    judge_calls = []

    def fake_load(path):
        return data if path == eval_file else saved[path]

    def fake_dump(value, path):
        saved[path] = value

    def fake_progress(func, tasks, keys, save, **kwargs):
        progress_calls.append((func.__name__, save))
        if func is scidocbench._eval_one_item:
            results = [(0.5, 'answer note') for _ in tasks]
        else:
            results = [(0.75, 'reasoning note') for _ in tasks]
        cache = saved.setdefault(save, {})
        cache.update(dict(zip(keys, results)))
        return results

    def fake_build_judge(**kwargs):
        judge_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        scidocbench, 'osp', SimpleNamespace(exists=lambda path: path in saved))
    monkeypatch.setattr(scidocbench, 'load', fake_load)
    monkeypatch.setattr(scidocbench, 'dump', fake_dump)
    monkeypatch.setattr(
        scidocbench, 'get_intermediate_file_path', _intermediate_file)
    monkeypatch.setattr(scidocbench, '_configure_content_cache', lambda *args: None)
    monkeypatch.setattr(scidocbench, 'build_judge', fake_build_judge)
    monkeypatch.setattr(scidocbench, 'track_progress_rich', fake_progress)

    dataset = scidocbench.SciDocBench.__new__(scidocbench.SciDocBench)
    dataset.enable_reasoning_diagnostic = False
    dataset.evaluate(eval_file, model='fake-judge', nproc=1)

    assert [name for name, _ in progress_calls] == ['_eval_one_item']
    answer_cache = progress_calls[0][1]
    assert scidocbench.SCORER_VERSION in answer_cache

    dataset.enable_reasoning_diagnostic = True
    summary = dataset.evaluate(
        eval_file, model='fake-judge', nproc=1, max_tokens=4096)

    assert [name for name, _ in progress_calls] == [
        '_eval_one_item',
        '_eval_one_reasoning',
    ]
    reasoning_cache = progress_calls[1][1]
    assert reasoning_cache != answer_cache
    assert scidocbench.REASONING_DIAGNOSTIC_VERSION in reasoning_cache
    reasoning_row = summary[
        summary['Category'] == 'Reasoning (question whitelist)'
    ].iloc[0]
    assert reasoning_row['Score'] == 75.0
    assert [call['max_tokens'] for call in judge_calls] == [8192, 4096]


def test_failed_cached_evaluations_are_retried(monkeypatch):
    eval_file = 'predictions.tsv'
    rows = []
    for index in ('0', '1'):
        rows.append({
            'index': index,
            'qid': f'euler-stratification_001__{index}',
            'pair_qid': 'euler-stratification_001',
            'partition': 'en_all_first',
            'mode': 'all_first',
            'language': 'en',
            'category': 'A1',
            'eval_method': 'judge',
            'prediction': '{"answer": "Yes", "reasoning": "because"}',
            'answer': '{"answer": "Yes"}',
            'question': 'Is it symmetric?',
            'judge_prompt': '',
        })
    data = pd.DataFrame(rows)
    answer_cache = _intermediate_file(
        eval_file, f'_fake-judge_{scidocbench.SCORER_VERSION}', 'pkl')
    reasoning_cache = _intermediate_file(
        eval_file, f'_fake-judge_{scidocbench.REASONING_DIAGNOSTIC_VERSION}', 'pkl')
    storage = _intermediate_file(
        eval_file,
        f'_fake-judge_{scidocbench.SCORER_VERSION}_'
        f'{scidocbench.REASONING_DIAGNOSTIC_VERSION}',
    )
    api_failure = scidocbench.INFERENCE_FAILURE_NOTE
    saved = {
        answer_cache: {
            '0': (0.0, 'Eval error: temporary failure'),
            '1': (0.0, api_failure),
        },
        reasoning_cache: {
            '0': (0.0, 'Failed to parse reasoning response: invalid'),
            '1': (0.0, api_failure),
        },
        storage: pd.DataFrame({
            'score': [0.0],
            'eval_note': ['Eval error: temporary failure'],
            'pair_qid': ['euler-stratification_001'],
            'reasoning_score': [None],
        }),
    }
    progress_calls = []

    def fake_load(path):
        return data if path == eval_file else saved[path]

    def fake_dump(value, path):
        saved[path] = value

    def fake_progress(func, tasks, keys, save, **kwargs):
        progress_calls.append((func.__name__, list(keys)))
        if func is scidocbench._eval_one_item:
            results = [(0.5, 'answer note') for _ in tasks]
        else:
            results = [(0.75, 'reasoning note') for _ in tasks]
        saved[save].update(dict(zip(keys, results)))
        return results

    monkeypatch.setattr(
        scidocbench, 'osp', SimpleNamespace(exists=lambda path: path in saved))
    monkeypatch.setattr(scidocbench, 'load', fake_load)
    monkeypatch.setattr(scidocbench, 'dump', fake_dump)
    monkeypatch.setattr(
        scidocbench, 'get_intermediate_file_path', _intermediate_file)
    monkeypatch.setattr(scidocbench, '_configure_content_cache', lambda *args: None)
    monkeypatch.setattr(scidocbench, 'build_judge', lambda **kwargs: object())
    monkeypatch.setattr(scidocbench, 'track_progress_rich', fake_progress)

    dataset = scidocbench.SciDocBench.__new__(scidocbench.SciDocBench)
    dataset.enable_reasoning_diagnostic = True
    dataset.evaluate(eval_file, model='fake-judge', nproc=1)

    assert progress_calls == [
        ('_eval_one_item', ['0', '1']),
        ('_eval_one_reasoning', ['0', '1']),
    ]
    assert scidocbench._cache_result_is_reusable(
        saved[answer_cache]['1'], scidocbench.ANSWER_EVAL_FAILURE_MARKERS)
    assert scidocbench._cache_result_is_reusable(
        saved[reasoning_cache]['1'],
        scidocbench.REASONING_EVAL_FAILURE_MARKERS)


def test_failed_retry_is_saved_with_original_notes(monkeypatch):
    eval_file = 'predictions.tsv'
    data = pd.DataFrame([{
        'index': '0',
        'qid': 'euler-stratification_001__en_all_first',
        'pair_qid': 'euler-stratification_001',
        'partition': 'en_all_first',
        'mode': 'all_first',
        'language': 'en',
        'category': 'A1',
        'eval_method': 'judge',
        'prediction': 'prediction',
        'answer': 'answer',
        'question': 'question',
        'judge_prompt': '',
    }])
    storage = _intermediate_file(
        eval_file,
        f'_fake-judge_{scidocbench.SCORER_VERSION}_'
        f'{scidocbench.REASONING_DIAGNOSTIC_VERSION}',
    )
    saved = {}
    answer_error = 'Eval error: answer retry still failed'
    reasoning_error = 'Reasoning eval error: reasoning retry still failed'

    def fake_load(path):
        return data if path == eval_file else saved[path]

    def fake_dump(value, path):
        saved[path] = value

    def fake_progress(func, tasks, keys, save, **kwargs):
        if func is scidocbench._eval_one_item:
            results = [(0.0, answer_error) for _ in tasks]
        else:
            results = [(None, reasoning_error) for _ in tasks]
        saved.setdefault(save, {}).update(dict(zip(keys, results)))
        return results

    monkeypatch.setattr(
        scidocbench, 'osp', SimpleNamespace(exists=lambda path: path in saved))
    monkeypatch.setattr(scidocbench, 'load', fake_load)
    monkeypatch.setattr(scidocbench, 'dump', fake_dump)
    monkeypatch.setattr(
        scidocbench, 'get_intermediate_file_path', _intermediate_file)
    monkeypatch.setattr(scidocbench, '_configure_content_cache', lambda *args: None)
    monkeypatch.setattr(scidocbench, 'build_judge', lambda **kwargs: object())
    monkeypatch.setattr(scidocbench, 'track_progress_rich', fake_progress)

    dataset = scidocbench.SciDocBench.__new__(scidocbench.SciDocBench)
    dataset.enable_reasoning_diagnostic = True
    dataset.evaluate(eval_file, model='fake-judge', nproc=1)

    result = saved[storage]
    assert result.iloc[0]['score'] == 0.0
    assert pd.isna(result.iloc[0]['reasoning_score'])
    assert answer_error in result.iloc[0]['eval_note']
    assert reasoning_error in result.iloc[0]['eval_note']
    assert not scidocbench._scored_result_is_reusable(
        result, include_reasoning=True)
