"""Offline regressions for the MMMU_TEST submission file.

MMMU_result_transfer derived its output path with
`result_path.replace('.xlsx', '.json')` and read the table with a raw
`load()`. Both assume the .xlsx default, but README strongly recommends
`PRED_FORMAT=tsv` for models with long responses, and get_pred_file_path
honours tsv/xlsx/json.

Under PRED_FORMAT=tsv the replace was a no-op, so the submission dict was
dumped to the prediction file's own .tsv path; dump() then failed, fell back
to a .pkl with only a warning, and the function returned the *prediction*
file path as though it were the submission JSON.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_smp_file():
    """Prefer the installed package; fall back to loading the smp chain."""
    try:
        import vlmeval.smp as smp
        return smp
    except Exception:
        pass
    for name in ('vlmeval', 'vlmeval.smp', 'vlmeval.utils',
                 'vlmeval.dataset', 'vlmeval.dataset.utils'):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod

    def _load(dotted, rel):
        spec = importlib.util.spec_from_file_location(dotted, REPO / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    _load('vlmeval.smp.log', 'vlmeval/smp/log.py')
    _load('vlmeval.smp.misc', 'vlmeval/smp/misc.py')
    _load('vlmeval.smp.vlm', 'vlmeval/smp/vlm.py')
    return _load('vlmeval.smp.file', 'vlmeval/smp/file.py')


@pytest.fixture(scope='module')
def transfer():
    smp_file = _load_smp_file()

    smp = sys.modules['vlmeval.smp']
    for attr in ('dump', 'load', 'load_prediction_file', 'get_intermediate_file_path',
                 'get_file_extension'):
        if not hasattr(smp, attr) and hasattr(smp_file, attr):
            setattr(smp, attr, getattr(smp_file, attr))
    if not hasattr(smp, 'gpt_key_set'):
        smp.gpt_key_set = lambda *a, **k: False
    log = sys.modules.setdefault('vlmeval.smp.log', types.ModuleType('vlmeval.smp.log'))
    if not hasattr(log, 'get_logger'):
        log.get_logger = lambda *a, **k: types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None
        )

    # Judge plumbing is only used by MMTBench_result_transfer.
    ju = types.ModuleType('vlmeval.dataset.utils.judge_util')
    ju.build_judge = lambda *a, **k: None
    ju.warn_if_judge_model_changed = lambda *a, **k: None
    sys.modules['vlmeval.dataset.utils.judge_util'] = ju
    mc = types.ModuleType('vlmeval.dataset.utils.multiple_choice')
    mc.extract_answer_from_item = lambda *a, **k: {'opt': ''}
    sys.modules['vlmeval.dataset.utils.multiple_choice'] = mc
    mp = types.ModuleType('vlmeval.utils.mp_util')
    mp.track_progress_rich = lambda *a, **k: []
    sys.modules['vlmeval.utils.mp_util'] = mp
    # can_infer is real logic used by MMMU_result_transfer, so load it for real.
    spec = importlib.util.spec_from_file_location(
        'vlmeval.utils.matching_util', REPO / 'vlmeval/utils/matching_util.py'
    )
    matching = importlib.util.module_from_spec(spec)
    sys.modules['vlmeval.utils.matching_util'] = matching
    spec.loader.exec_module(matching)

    spec = importlib.util.spec_from_file_location(
        'vlmeval.utils.result_transfer', REPO / 'vlmeval/utils/result_transfer.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['vlmeval.utils.result_transfer'] = module
    spec.loader.exec_module(module)
    return module


def _frame():
    """One MCQ row and one open-ended row, as MMMU_TEST predictions look."""
    return pd.DataFrame([
        {'index': 0, 'id': 'validation_Art_1', 'prediction': 'B',
         'A': 'first', 'B': 'second', 'C': 'third', 'D': 'fourth'},
        {'index': 1, 'id': 'validation_Math_2', 'prediction': '42',
         'A': None, 'B': None, 'C': None, 'D': None},
    ])


def _write(frame, path):
    suffix = path.suffix
    if suffix == '.xlsx':
        frame.to_excel(path, index=False)
    elif suffix == '.tsv':
        frame.to_csv(path, sep='\t', index=False)
    elif suffix == '.json':
        path.write_text(json.dumps(frame.to_dict('records')))
    else:
        raise AssertionError(suffix)


@pytest.mark.parametrize('name', [
    'Model_MMMU_TEST.tsv',
    'Model_MMMU_TEST.json',
    'Model_MMMU_TEST.xlsx',
])
def test_submission_file_is_json_and_distinct(transfer, tmp_path, name):
    pred = tmp_path / name
    _write(_frame(), pred)

    out = Path(transfer.MMMU_result_transfer(str(pred)))

    assert out.suffix == '.json', f'submission must be JSON, got {out.name}'
    assert out != pred, 'submission must not overwrite the prediction file'
    assert out.exists(), f'{out.name} was not written'

    payload = json.loads(out.read_text())
    assert payload == {'validation_Art_1': 'B', 'validation_Math_2': '42'}


@pytest.mark.parametrize('name', [
    'Model_MMMU_TEST.tsv',
    'Model_MMMU_TEST.json',
    'Model_MMMU_TEST.xlsx',
])
def test_prediction_file_is_left_intact(transfer, tmp_path, name):
    pred = tmp_path / name
    _write(_frame(), pred)
    before = pred.read_bytes()

    transfer.MMMU_result_transfer(str(pred))

    assert pred.read_bytes() == before


@pytest.mark.parametrize('name', [
    'Model_MMMU_TEST.tsv',
    'Model_MMMU_TEST.json',
    'Model_MMMU_TEST.xlsx',
])
def test_no_pkl_fallback_is_triggered(transfer, tmp_path, name):
    """dump() silently degrades to .pkl on failure; nothing should hit that."""
    pred = tmp_path / name
    _write(_frame(), pred)

    transfer.MMMU_result_transfer(str(pred))

    assert list(tmp_path.glob('*.pkl')) == []
