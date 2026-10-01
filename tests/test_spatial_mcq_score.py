import importlib.util
import sys
import types
import unittest
from unittest import mock

import pandas as pd


def _load_matching_func():
    spec = importlib.util.spec_from_file_location(
        'matching_func',
        'vlmeval/dataset/utils/spatial_bench/matching_func.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cal_scores():
    """Load ``cal_scores`` in isolation.

    Only ``compute_mcq_score`` is exercised here. It needs ``can_match_option``
    from ``matching_func``; the judge / LLM extraction imports are irrelevant to
    that function and are replaced with placeholders so the module can be
    imported without the rest of the package's heavy dependencies.
    """
    matching = _load_matching_func()

    pkg = 'vlmeval.dataset.utils.spatial_bench'
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    utils = types.ModuleType('vlmeval.dataset.utils')
    utils.__path__ = ['vlmeval/dataset/utils']
    spatial = types.ModuleType(pkg)
    spatial.__path__ = ['vlmeval/dataset/utils/spatial_bench']
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.get_intermediate_file_path = lambda *a, **k: ''
    smp_vlm = types.ModuleType('vlmeval.smp.vlm')
    smp_vlm.gpt_key_set = lambda: False
    judge_util = types.ModuleType('vlmeval.dataset.utils.judge_util')
    judge_util.build_judge = lambda *a, **k: None
    llm_extract = types.ModuleType(pkg + '.llm_extract')
    llm_extract.parallel_llm_extract = lambda *a, **k: None
    matching_mod = types.ModuleType(pkg + '.matching_func')
    matching_mod.can_match_option = matching.can_match_option
    matching_mod.can_match_na = matching.can_match_na
    tools = types.ModuleType(pkg + '.tools')
    tools.__path__ = []
    tools_files = types.ModuleType(pkg + '.tools.files')
    tools_files.build_eval_paths = lambda *a, **k: (None, None)
    tools_files.get_judge_tag_from_score_fn = lambda *a, **k: None

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.utils': utils,
        'vlmeval.dataset.utils.spatial_bench': spatial,
        'vlmeval.smp': smp,
        'vlmeval.smp.file': smp_file,
        'vlmeval.smp.vlm': smp_vlm,
        'vlmeval.dataset.utils.judge_util': judge_util,
        pkg + '.llm_extract': llm_extract,
        pkg + '.matching_func': matching_mod,
        pkg + '.tools': tools,
        pkg + '.tools.files': tools_files,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.s