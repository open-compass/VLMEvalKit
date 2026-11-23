import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from vlmeval.dataset import DATASET_CLASSES, build_dataset, gdp_pdf, gdp_pdf_aa
from vlmeval.dataset.gdp_pdf import FAIL_MSG, GDPPDF
from vlmeval.dataset.gdp_pdf_aa import GDPPDFAA
from vlmeval.dataset.utils.judge_cache import dump_judge_cache
from vlmeval.dataset.utils.repeat import wrap_dataset_runs
from vlmeval.smp import dump, load, md5


@pytest.fixture
def prepared_data(monkeypatch, tmp_path):
    fitz = pytest.importorskip('fitz')
    monkeypatch.setenv('LMUData', str(tmp_path))
    for name in ('GDP_pdf', 'GDP_pdf_AA', 'GDP_pdf_AA_text'):
        directory = tmp_path / name
        directory.mkdir()
        pdf = fitz.open()
        for number in range(5):
            page = pdf.new_page(width=72, height=96)
            page.insert_text((8, 20), f'Page {number + 1}')
        pdf.save(str(directory / 'document.pdf'))
        pdf.close()
        pd.DataFrame([
            dict(task_id='a', prompt='Task A', pdf_path='document.pdf', domain='law',
                 **{'rubric - 1. criterion': 'pass', 'rubric - 2. criterion': 'fail'}),
            dict(task_id='b', prompt='Task B', pdf_path='document.pdf', domain='law',
                 **{'rubric - 1. criterion': 'pass'}),
            dict(task_id='c', prompt='Task C', pdf_path='document.pdf', domain='finance',
                 **{'rubric - 1. criterion': 'pass'}),
        ]).to_parquet(directory / 'data.parquet')
    monkeypatch.setattr(GDPPDF, 'HF_PARQUET_MD5', md5(str(directory / 'data.parquet')))
    calls = []

    def parse(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({'pages': [
            dict(page=i, text=f'  Full text {i}\nEnd {i}  ') for i in range(1, 6)
        ]}))

    executable = tmp_path / 'liteparse' / 'dist' / 'cli.js'
    executable.parent.mkdir(parents=True)
    executable.write_text('')
    (executable.parent.parent / 'package.json').write_text(json.dumps(
        dict(name='@llamaindex/liteparse', version='2.5.0')
    ))

    def which(command):
        assert command == 'lit'
        return str(executable)

    monkeypatch.setattr(gdp_pdf_aa.shutil, 'which', which)
    monkeypatch.setattr(gdp_pdf_aa.subprocess, 'run', parse)
    return tmp_path, calls


@pytest.mark.parametrize('dataset_name', ['GDP_pdf_AA', 'GDP_pdf_AA_text'])
def test_registration_and_complete_ocr_preparation(prepared_data, dataset_name):
    directory, calls = prepared_data
    assert GDPPDFAA in DATASET_CLASSES
    dataset = build_dataset(dataset_name)
    assert isinstance(dataset, GDPPDFAA)
    assert len(dataset) == 3  # Exactly one attempt per source task.
    assert dataset.dpi == 150
    assert len(calls) == 1  # Tasks sharing a document reuse its text.
    command = calls[0]
    assert command[command.index('--ocr-language') + 1] == 'eng'
    assert command[command.index('--max-pages') + 1] == '5'
    assert '--no-ocr' not in command and '--target-pages' not in command
    for number in range(1, 6):
        expected = f'--- Page {number} ---\n  Full text {number}\nEnd {number}  '
        assert expected in dataset.data.iloc[0]['document_text']
    assert list((directory / dataset_name / 'liteparse_2.5.0_eng').glob('*.json'))
    build_dataset(dataset_name)
    assert len(calls) == 1


def test_ocr_rejects_missing_pages_and_does_not_cache(prepared_data, monkeypatch):
    directory, _ = prepared_data
    monkeypatch.setattr(gdp_pdf_aa.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout=json.dumps({'pages': [dict(page=1, text='only one page')]})))
    with pytest.raises(ValueError, match='every page in order'):
        build_dataset('GDP_pdf_AA')
    assert not (directory / 'GDP_pdf_AA' / 'liteparse_2.5.0_eng').exists()


@pytest.mark.parametrize('dataset_name', ['GDP_pdf_AA', 'GDP_pdf_AA_text'])
def test_missing_liteparse_reports_not_installed(prepared_data, monkeypatch, dataset_name):
    _, calls = prepared_data
    monkeypatch.setattr(gdp_pdf_aa.shutil, 'which', lambda command: None)
    with pytest.raises(ImportError, match='LiteParse is not installed'):
        build_dataset(dataset_name)
    assert not calls


def test_ocr_cache_invalidates_when_pdf_changes(prepared_data):
    directory, calls = prepared_data
    dataset = build_dataset('GDP_pdf_AA')
    pdf = directory / 'GDP_pdf_AA' / 'document.pdf'
    with pdf.open('ab') as stream:
        stream.write(b'\n%changed\n')
    dataset._extract_pages('document.pdf')
    assert len(calls) == 2


def test_original_surge_prompt_and_defaults_are_preserved(prepared_data):
    _, calls = prepared_data
    dataset = build_dataset('GDP_pdf')
    assert type(dataset) is GDPPDF
    assert dataset.dpi == 144
    assert dataset.DEFAULT_JUDGE_MODEL == 'gemini-3.5-flash'
    assert 'document_text' not in dataset.data
    messages = dataset.build_prompt(0)
    assert messages[0] == dict(type='text', value='Task A')
    assert len(messages) == 6
    assert not calls


@pytest.mark.parametrize('kwargs,count,coverage,composite', [
    ({}, 5, 5, False),
    ({'max_images': 3}, 3, 5, True),
    ({'max_images': 2}, 2, 5, True),
    ({'max_images': 1}, 1, 4, True),
])
def test_prompt_images_first_and_full_text_survives_adaptation(
    prepared_data, kwargs, count, coverage, composite
):
    dataset = build_dataset('GDP_pdf_AA', **kwargs)
    messages = dataset.build_prompt(0)
    assert [message['type'] for message in messages] == ['image'] * count + ['text']
    text = messages[-1]['value']
    assert text.startswith('Task A\n\n')
    assert text.endswith(dataset.data.iloc[0]['document_text'])
    assert ('composites' in text) is composite
    if coverage:
        assert f'pages 1 through {coverage}' in text
    assert 'Full text 5' in text
    for message in messages[:-1]:
        assert Image.open(message['value']).mode == 'RGB'


def test_text_dataset_keeps_complete_text_without_rendering(prepared_data, monkeypatch):
    def unexpected_render(*args, **kwargs):
        pytest.fail('The text dataset must not render page images')

    monkeypatch.setattr(GDPPDF, 'dump_image', unexpected_render)
    dataset = build_dataset('GDP_pdf_AA_text')
    line = dataset.data.iloc[0]
    assert dataset.dump_image(line) == []
    assert dataset.build_prompt(0) == [dict(
        type='text', value='Task A\n\nDocument text:\n' + line['document_text']
    )]
    assert 'Full text 5' in line['document_text']


@pytest.mark.parametrize('kwargs', [
    {'dpi': 71}, {'dpi': 151}, {'dpi': True}, {'max_images': 0},
    {'max_images': True},
])
def test_invalid_configuration_rejected_before_loading(kwargs):
    with pytest.raises(ValueError):
        GDPPDFAA(**kwargs)


def test_liteparse_checks_manifest_instead_of_broken_cli_version(monkeypatch, tmp_path):
    executable = tmp_path / 'package' / 'dist' / 'cli.js'
    executable.parent.mkdir(parents=True)
    executable.write_text('')
    manifest = executable.parent.parent / 'package.json'
    manifest.write_text(json.dumps(dict(name='@llamaindex/liteparse', version='2.5.0')))
    monkeypatch.setattr(gdp_pdf_aa.shutil, 'which', lambda command: str(executable))
    dataset = GDPPDFAA.__new__(GDPPDFAA)
    dataset._liteparse_executable = None
    assert dataset._get_liteparse() == str(executable)
    dataset._liteparse_executable = None
    manifest.write_text(json.dumps(dict(name='@llamaindex/liteparse', version='2.6.0')))
    with pytest.raises(ValueError, match='version 2.5.0'):
        dataset._get_liteparse()
    monkeypatch.setattr(gdp_pdf_aa.shutil, 'which', lambda command: None)
    with pytest.raises(ImportError, match='2.5.0'):
        dataset._get_liteparse()


def _judge_inputs(prompt):
    assert prompt.startswith('Task prompt:\n')
    question, rubric_prompt = prompt[len('Task prompt:\n'):].split('\n\n', 1)
    response, rest = rubric_prompt.split('  response: ', 1)[1].split(',\n  criteria: ', 1)
    criterion = rest.split(',\n}', 1)[0]
    return dict(task=question, answer=response, criterion=criterion)


class RecordingJudge:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        payload = _judge_inputs(prompt)
        if payload['answer'] == 'unparseable' or (
            payload['answer'] == 'partial' and payload['criterion'] == 'fail'
        ):
            return 'invalid verdict'
        score = '1' if payload['criterion'] == 'pass' else '0'
        return json.dumps(dict(score=score, rationale='test'))


@pytest.fixture
def judge(monkeypatch):
    model = RecordingJudge()
    options = []

    def build(**kwargs):
        options.append(kwargs)
        return model

    def run(func, tasks, keys, cache_file, **kwargs):
        cache = gdp_pdf.load_judge_cache(cache_file)
        cache.update({key: func(*task) for key, task in zip(keys, tasks)})
        dump_judge_cache(cache, cache_file)
        return cache

    monkeypatch.setattr(gdp_pdf, 'build_judge', build)
    monkeypatch.setattr(gdp_pdf, 'run_cached_tasks', run)
    monkeypatch.setattr(gdp_pdf_aa, 'build_judge', build)
    monkeypatch.setattr(gdp_pdf_aa, 'run_cached_tasks', run)
    return model, options


@pytest.mark.parametrize('dataset_name', ['GDP_pdf_AA', 'GDP_pdf_AA_text'])
def test_aa_judging_missing_predictions_macro_mean_and_cache(prepared_data, judge, dataset_name):
    directory, _ = prepared_data
    model, options = judge
    dataset = build_dataset(dataset_name)
    prediction = str(directory / 'prediction.tsv')
    dump(pd.DataFrame([
        dict(index='b', prediction='answer B'), dict(index='a', prediction='answer A'),
    ]), prediction)
    result = dataset.evaluate(prediction, nproc=1)
    assert result['all_pass'] == pytest.approx(1 / 3)
    assert result['mean_pass'] == pytest.approx((0.5 + 1 + 0) / 3)
    assert result['domain/law/mean_pass'] == 0.75
    assert result['domain/finance/mean_pass'] == 0
    assert not any(key.startswith('domain/') and key.endswith('/all_pass') for key in result)
    assert result['num_samples'] == 3
    assert options[0]['model'] == 'gpt-5.6-luna'
    assert options[0]['reasoning_effort'] == 'medium'
    assert len(model.calls) == 3
    for prompt, _ in model.calls:
        payload = _judge_inputs(prompt)
        assert payload['task'] == ('Task A' if payload['answer'] == 'answer A' else 'Task B')
        assert set(payload) == {'task', 'answer', 'criterion'}
        assert prompt == (
            f'Task prompt:\n{payload["task"]}\n\n'
            + gdp_pdf._judge_prompt(payload['answer'], payload['criterion'])
        )
    assert dataset.report_primary_metric(result) == {'all_pass': pytest.approx(1 / 3)}
    dataset.evaluate(prediction, nproc=1)
    assert len(model.calls) == 3
    dataset.data.loc[dataset.data['index'] == 'b', 'question'] = 'Updated Task B'
    dataset.evaluate(prediction, nproc=1)
    assert len(model.calls) == 4  # Task prompt is part of AA cache identity.
    detail = next(directory.glob('*aa_rubric_v2*detail.tsv'))
    assert list(load(str(detail))['index']) == ['a', 'b', 'c']


@pytest.mark.parametrize('dataset_name', ['GDP_pdf_AA', 'GDP_pdf_AA_text'])
def test_repeated_aa_evaluation_and_judge_cache(prepared_data, judge, dataset_name):
    directory, _ = prepared_data
    model, _ = judge
    dataset = wrap_dataset_runs(build_dataset(dataset_name))
    assert dataset.n_runs == 5
    assert len(dataset) == 15 and dataset.data['index'].is_unique
    assert dataset.data.groupby('orig_index').size().tolist() == [5, 5, 5]
    predictions = dataset.data.copy()
    predictions['prediction'] = [
        'answer' if index == 'a' or sample < (1 if index == 'b' else 2) else FAIL_MSG
        for index, sample in zip(predictions['orig_index'], predictions['sample_id'])
    ]
    prediction = str(directory / 'repeated.tsv')
    dump(predictions, prediction)

    result = dataset.evaluate(prediction, nproc=1)
    assert result['all_pass'] == pytest.approx(3 / 15)
    assert result['mean_pass'] == pytest.approx(5.5 / 15)
    assert result['pass@5'] == pytest.approx(2 / 3)
    assert result['pass^5'] == 0
    assert result['num_samples'] == 3
    assert dataset.report_primary_metric(result) == {'all_pass': pytest.approx(3 / 15)}
    detail = load(str(directory / 'repeated_results.csv'))
    assert len(detail) == 15
    assert not detail.duplicated(['orig_index', 'sample_id']).any()
    assert len(list((directory / 'repeated_runs').glob('*aa_rubric_v2*detail.tsv'))) == 5
    assert len(model.calls) == 13
    assert dataset.evaluate(prediction, nproc=1) == result
    assert len(model.calls) == 13


@pytest.mark.parametrize('answer,unscored,first_calls,total_calls', [
    ('unparseable', 2, 10, 20), ('partial', 1, 6, 11),
])
def test_incomplete_grading_is_not_reported_and_is_retried(
    prepared_data, judge, answer, unscored, first_calls, total_calls
):
    directory, _ = prepared_data
    model, _ = judge
    dataset = build_dataset('GDP_pdf_AA_text')
    prediction = str(directory / 'incomplete.tsv')
    dump(pd.DataFrame([dict(index='a', prediction=answer)]), prediction)
    with pytest.raises(ValueError, match='grading is incomplete'):
        dataset.evaluate(prediction, nproc=1)
    detail = load(str(next(directory.glob('*aa_rubric_v2*detail.tsv'))))
    assert detail['num_unscored_criteria'].sum() == unscored
    assert len(detail) == 3
    assert not list(directory.glob('*score*.json'))
    assert len(model.calls) == first_calls
    with pytest.raises(ValueError, match='grading is incomplete'):
        dataset.evaluate(prediction, nproc=1)
    assert len(model.calls) == total_calls


@pytest.mark.parametrize('score', ['0', '0.99994', '0.99996', '1'])
def test_aa_reuses_surge_parser_retries_and_threshold(score):
    class Judge:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            if len(self.calls) == 1:
                return 'invalid verdict'
            return json.dumps(dict(score=score, rationale='same parsing rules'))

    surge, aa = Judge(), Judge()
    expected = gdp_pdf._score_criterion(surge, 'response', 'criterion')
    actual = gdp_pdf_aa._score_aa_criterion(aa, 'response', 'criterion', 'original task')
    assert actual == expected
    assert len(aa.calls) == len(surge.calls) == 2
    assert aa.calls[0] == ('Task prompt:\noriginal task\n\n' + surge.calls[0][0],
                           surge.calls[0][1])


def test_empty_predictions_zero_and_unknown_or_duplicate_indices_rejected(prepared_data, judge):
    directory, _ = prepared_data
    model, _ = judge
    dataset = build_dataset('GDP_pdf_AA_text')
    prediction = str(directory / 'empty.tsv')
    dump(pd.DataFrame([dict(index='a', prediction='   ')]), prediction)
    assert dataset.evaluate(prediction, nproc=1)['mean_pass'] == 0
    assert not model.calls
    for indices in (['a', 'a'], ['unknown']):
        dump(pd.DataFrame(dict(index=indices, prediction=['x'] * len(indices))), prediction)
        with pytest.raises(ValueError, match='unique, known'):
            dataset.evaluate(prediction)
    surge = build_dataset('GDP_pdf')
    with pytest.raises(ValueError, match='exactly once'):
        surge.evaluate(prediction)


def test_surge_judge_protocol_and_metrics_unchanged(prepared_data, monkeypatch):
    directory, _ = prepared_data

    class SurgeJudge:
        def generate(self, prompt, **kwargs):
            assert 'Read this json object:' in prompt and 'Task A' not in prompt
            return json.dumps(dict(score='0.99996', rationale='rounded as in Surge'))

    monkeypatch.setattr(gdp_pdf, 'build_judge', lambda **kwargs: SurgeJudge())
    dataset = build_dataset('GDP_pdf')
    prediction = str(directory / 'surge.tsv')
    dump(pd.DataFrame(dict(index=['a', 'b', 'c'], prediction=['answer', FAIL_MSG, FAIL_MSG])),
         prediction)
    result = dataset.evaluate(prediction, nproc=1)
    assert result['all_pass'] == pytest.approx(1 / 3)
    assert result['mean_criteria'] == pytest.approx(1 / 3)
    assert result['domain/law/all_pass'] == 0.5
    assert 'mean_pass' not in result
    assert list(directory.glob('*rubric_v1*cache.pkl'))
