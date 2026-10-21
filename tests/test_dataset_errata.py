import sys
import types
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from vlmeval.dataset import ConcatDataset
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_mcq import ImageMCQDataset
from vlmeval.dataset.image_vqa import MTVQADataset
from vlmeval.dataset.utils.data_errata import ISSUE_1754_DATA_ERRATA, filter_unscorable_samples
from vlmeval.smp import dump, get_intermediate_file_path, load

EXPECTED_INDICES = {
    'MMBench_DEV_EN': {1770, 1849, 1852},
    'MMBench_DEV_CN': {1770, 1849, 1852, 1879, 1880},
    'MMBench_dev_ar': {710, 1849, 1852},
    'MMBench_dev_cn': {1770, 1849, 1852, 1879, 1880},
    'MMBench_dev_en': {1770, 1849, 1852},
    'MMBench_dev_pt': {710, 712, 1849, 1852},
    'MMBench_dev_ru': {710, 712, 1849, 1852},
    'MMBench_dev_tr': {712, 1849, 1852},
    'MMMB_ar': {1141, 1286, 1634},
    'MMMB_cn': {1141, 1286, 1634},
    'MMMB_pt': {
        12, 26, 31, 61, 63, 138, 153, 154, 162, 227, 242, 249, 270, 288,
        302, 306, 309, 311, 314, 337, 338, 356, 369, 371, 375, 406, 419,
        458, 1141, 1286, 1547, 1634, 1723,
    },
    'MMMB_tr': {1141, 1286, 1634},
    'MTVQA_TEST': {
        743, 744, 745, 746, 747, 1720, 1721, 2034, 2037, 2065, 2138, 2364,
        2367, 2863, 3093, 3124, 3125, 3126, 3127, 3796, 3797, 3798, 3799,
        3800, 3820, 3821, 5044, 5046, 5047, 5048, 5853, 5855, 5856,
    },
    'SEEDBench2_Plus': {1592},
    'AI2D_TEST': {225703},
    'AI2D_TEST_NO_MASK': {225703},
    'MMT-Bench_VAL': {8279, 8289, 8293, 20554},
    'MMT-Bench_VAL_MI': {8279, 8289, 8293, 20554},
}


def make_rows(indices):
    return pd.DataFrame([
        dict(index=str(index), question='Same question', A='clima', B='clima',
             answer='A', category='DE', split='test')
        for index in indices
    ])


def test_errata_matches_issue_list_and_registered_versions():
    assert {name: set(errata[1]) for name, errata in ISSUE_1754_DATA_ERRATA.items()} == EXPECTED_INDICES
    assert sum(len(indices) for name, indices in EXPECTED_INDICES.items() if name != 'MMT-Bench_VAL_MI') == 112
    assert sum(len(indices) for indices in EXPECTED_INDICES.values()) == 116
    for name, (release_md5, indices) in ISSUE_1754_DATA_ERRATA.items():
        dataset_class = MTVQADataset if name == 'MTVQA_TEST' else ImageMCQDataset
        assert release_md5 == dataset_class.DATASET_MD5[name]
        assert len(indices) == len(set(indices))


@pytest.mark.parametrize('dataset_name', EXPECTED_INDICES)
def test_constructor_excludes_only_issue_indices(dataset_name, tmp_path):
    original = make_rows([*sorted(EXPECTED_INDICES[dataset_name]), 999998])
    dataset_class = MTVQADataset if dataset_name == 'MTVQA_TEST' else ImageMCQDataset
    with patch.object(ImageBaseDataset, 'load_data', return_value=original.copy()), \
            patch('vlmeval.dataset.image_base.LMUDataRoot', return_value=str(tmp_path)), \
            pytest.warns(UserWarning, match='Issue #1754'):
        dataset = dataset_class(dataset_name)
    assert dataset.data['index'].tolist() == [999998]
    assert dataset.data.iloc[0]['A'] == dataset.data.iloc[0]['B']
    assert len(original) == len(EXPECTED_INDICES[dataset_name]) + 1


def test_image_reference_is_resolved_before_exclusion(tmp_path):
    original = make_rows([1770, 2])
    image = 'base64-image-content' * 10
    original['image'] = [image, '1770']
    with patch.object(ImageBaseDataset, 'load_data', return_value=original), \
            patch('vlmeval.dataset.image_base.LMUDataRoot', return_value=str(tmp_path)), \
            pytest.warns(UserWarning, match='Issue #1754'):
        dataset = ImageMCQDataset('MMBench_DEV_EN')
    assert dataset.data['index'].tolist() == [2]
    assert dataset.data.iloc[0]['image'] == image
    with patch.object(dataset, 'dump_image', return_value=['image.png']):
        prompt = dataset.build_prompt(0)
    assert prompt[0] == {'type': 'image', 'value': 'image.png'}
    assert 'Same question' in prompt[1]['value']


@pytest.mark.parametrize('use_group_index', [False, True])
def test_circular_variants_are_excluded_together(use_group_index):
    original = make_rows([1770, 1001770, 2001770, 2])
    if use_group_index:
        original['index'] = ['1770', '900001', '900002', '2']
        original['g_index'] = ['1770', '1770', '1770', '2']
    with pytest.warns(UserWarning, match='Excluding 3'):
        filtered = filter_unscorable_samples(original, 'MMBench_DEV_EN', ImageMCQDataset.DATASET_MD5['MMBench_DEV_EN'])
    assert filtered['index'].tolist() == ['2']


@pytest.mark.parametrize('dataset_name,index', [('MMMB_pt', 12), ('AI2D_TEST', 225703), ('MTVQA_TEST', 743)])
def test_non_circular_datasets_use_exact_indices(dataset_name, index):
    original = make_rows([index, index + 1000000])
    release_md5 = ISSUE_1754_DATA_ERRATA[dataset_name][0]
    with pytest.warns(UserWarning, match='Excluding 1'):
        filtered = filter_unscorable_samples(original, dataset_name, release_md5)
    assert filtered['index'].tolist() == [str(index + 1000000)]


@pytest.mark.parametrize('dataset_name', ['MMBench_DEV_EN_V11', 'MMMB_en', 'MMMB_ru', 'CustomDataset'])
def test_unlisted_datasets_are_unchanged(dataset_name):
    original = make_rows([12, 1770, 1849, 1852])
    assert filter_unscorable_samples(original, dataset_name, 'any-release') is original


def test_changed_release_and_no_matching_indices_are_unchanged():
    original = make_rows([1770, 2])
    assert filter_unscorable_samples(original, 'MMBench_DEV_EN', 'corrected-release') is original
    valid = make_rows([2, 3])
    filtered = filter_unscorable_samples(valid, 'MMBench_DEV_EN', ImageMCQDataset.DATASET_MD5['MMBench_DEV_EN'])
    assert filtered is valid


def test_empty_and_all_excluded_subsets_preserve_schema():
    original = make_rows([1770])
    with pytest.warns(UserWarning, match='Excluding 1'):
        filtered = filter_unscorable_samples(original, 'MMBench_DEV_EN', ImageMCQDataset.DATASET_MD5['MMBench_DEV_EN'])
    assert filtered.empty
    assert filtered.dtypes.equals(original.dtypes)
    empty_result = filter_unscorable_samples(filtered, 'MMBench_DEV_EN', ImageMCQDataset.DATASET_MD5['MMBench_DEV_EN'])
    assert empty_result is filtered


def test_old_mcq_prediction_is_filtered_before_metadata_check(tmp_path, monkeypatch):
    monkeypatch.setenv('PRED_FORMAT', 'tsv')
    original = make_rows([225703, 999998])
    with patch.object(ImageBaseDataset, 'load_data', return_value=original.copy()), \
            patch('vlmeval.dataset.image_base.LMUDataRoot', return_value=str(tmp_path)), \
            pytest.warns(UserWarning, match='Issue #1754'):
        dataset = ImageMCQDataset('AI2D_TEST')
    original['prediction'] = ['B', 'A']
    eval_file = str(tmp_path / 'old_prediction.tsv')
    dump(original, eval_file)
    with pytest.warns(UserWarning, match='Issue #1754'):
        result = dataset.evaluate(eval_file, model='exact_matching', nproc=1)
    assert result['Overall'].tolist() == [1.0]
    detail = load(get_intermediate_file_path(eval_file, '_exact_matching_result'))
    assert detail['index'].tolist() == [999998]
    assert load(eval_file)['index'].tolist() == [225703, 999998]


def test_mtvqa_old_prediction_and_classmethod_are_compatible(tmp_path):
    original = make_rows([743, 744, 8001])
    original['answer'] = ['wrong annotation', 'another annotation', 'Berlin.']
    original['prediction'] = ['no match', 'no match', 'berlin']
    eval_file = str(tmp_path / 'mtvqa.tsv')
    dump(original, eval_file)
    with pytest.warns(UserWarning, match='Excluding 2'):
        result = MTVQADataset.evaluate(eval_file)
    assert result == {'DE': 100.0, 'Average': 100.0}
    assert len(load(eval_file)) == 3
    with patch.dict(MTVQADataset.DATASET_MD5, {'MTVQA_TEST': 'corrected-release'}):
        unfiltered = MTVQADataset.evaluate(eval_file)
    assert unfiltered['Average'] == pytest.approx(100 / 3)


def test_print_vanilla_filters_old_circular_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv('PRED_FORMAT', 'tsv')
    monkeypatch.setenv('PRINT_VANILLA', '1')
    original = make_rows([1770, 1001770, 2, 1000002])
    original['g_index'] = [1770, 1770, 2, 2]
    with patch.object(ImageBaseDataset, 'load_data', return_value=original.copy()), \
            patch('vlmeval.dataset.image_base.LMUDataRoot', return_value=str(tmp_path)), \
            pytest.warns(UserWarning, match='Issue #1754'):
        dataset = ImageMCQDataset('MMBench_DEV_EN')
    original['prediction'] = 'A'
    eval_file = str(tmp_path / 'circular.tsv')
    dump(original, eval_file)
    with pytest.warns(UserWarning, match='Issue #1754'):
        result = dataset.evaluate(eval_file, model='exact_matching', nproc=1)
    assert result['Overall'].tolist() == [1.0]
    for suffix, indices in [('_result', [2]), ('_vanilla_circ0_result', [2]), ('_vanilla_all_result', [2, 1000002])]:
        detail = load(get_intermediate_file_path(eval_file, '_exact_matching' + suffix))
        assert detail['index'].tolist() == indices


def test_verifier_never_receives_excluded_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv('PRED_FORMAT', 'tsv')
    original = make_rows([225703, 999998])
    original['prediction'] = 'A'
    eval_file = str(tmp_path / 'verifier.tsv')
    dump(original, eval_file)
    verifier = Mock()
    verifier.evaluate.return_value = True
    module = types.ModuleType('vlmeval.dataset.utils.verifier')
    module.Verifier = Mock(return_value=verifier)
    with patch.object(ImageBaseDataset, 'load_data', return_value=original.copy()), \
            patch('vlmeval.dataset.image_base.LMUDataRoot', return_value=str(tmp_path)), \
            pytest.warns(UserWarning, match='Issue #1754'):
        dataset = ImageMCQDataset('AI2D_TEST')
    with patch.dict(sys.modules, {'vlmeval.dataset.utils.verifier': module}), \
            pytest.warns(UserWarning, match='Issue #1754'):
        dataset.evaluate(eval_file, use_verifier=True)
    assert verifier.evaluate.call_count == 1
    detail = load(get_intermediate_file_path(eval_file, '_detailed_results'))
    assert detail['index'].tolist() == [999998]


@pytest.mark.parametrize('dataset_name', ['MMMB', 'MTL_MMBench_DEV'])
def test_concat_dataset_uses_filtered_child_data(dataset_name, tmp_path):
    def load_child(dataset, name):
        indices = EXPECTED_INDICES.get(name, set())
        return make_rows([*sorted(indices), 999998])

    with patch.object(ImageBaseDataset, 'load_data', load_child), \
            patch('vlmeval.dataset.image_base.LMUDataRoot', return_value=str(tmp_path)), \
            pytest.warns(UserWarning, match='Issue #1754'):
        dataset = ConcatDataset(dataset_name)
    assert len(dataset) == 6
    assert all(child.data['index'].tolist() == [999998] for child in dataset.dataset_map.values())
