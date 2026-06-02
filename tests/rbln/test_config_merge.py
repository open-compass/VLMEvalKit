"""US-003 — rbln_config merge + compile-vs-load behaviour in base.py.

These exercise ``RBLNVLMBase._merge_rbln_config`` and
``RBLNVLMBase._from_pretrained`` directly via ``__new__`` (so no
``__init__`` model load happens) and a fake optimum-rbln class that
records the kwargs it is called with.

Key invariants (mirroring commit d2065ce — "Don't re-pass compile-time
rbln_config when loading cached artifact"):

* Loading from a compiled directory forces ``export=False`` and passes
  ONLY the runtime config — never the baked-in compile-time keys (which
  would raise ``ValueError`` in optimum-rbln).
* Compiling fresh passes the merged compile config.
* ``_merge_rbln_config`` does a one-level nested merge so per-submodule
  runtime overrides don't clobber sibling compile-time keys.
"""

from __future__ import annotations

from vlmeval.vlm.rbln.base import RBLNVLMBase


class _FakeRBLN:
    """Stand-in for an optimum-rbln model class."""

    def __init__(self):
        self.captured = None

    def from_pretrained(self, path, **kwargs):
        self.captured = {'path': path, **kwargs}
        return 'FAKE_MODEL'


def _bare_instance(**attrs):
    obj = RBLNVLMBase.__new__(RBLNVLMBase)
    obj.rbln_config = {}
    obj.rbln_runtime_config = {}
    obj.rbln_export = None
    obj.verbose = False
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


# ----------------------------------------------------------------------
# _merge_rbln_config
# ----------------------------------------------------------------------

def test_merge_returns_compile_config_when_no_runtime():
    obj = _bare_instance(
        rbln_config={'tensor_parallel_size': 8, 'visual': {'max_seq_lens': 6400}},
        rbln_runtime_config={},
    )
    merged = obj._merge_rbln_config()
    assert merged == {'tensor_parallel_size': 8, 'visual': {'max_seq_lens': 6400}}


def test_merge_one_level_nested_does_not_clobber_compile_keys():
    obj = _bare_instance(
        rbln_config={'tensor_parallel_size': 8, 'visual': {'max_seq_lens': 6400}},
        rbln_runtime_config={'visual': {'device': 0}, 'device': [0, 1]},
        rbln_export=False,
    )
    merged = obj._merge_rbln_config()
    # compile-time visual.max_seq_lens preserved, runtime visual.device added
    assert merged['visual'] == {'max_seq_lens': 6400, 'device': 0}
    assert merged['tensor_parallel_size'] == 8
    assert merged['device'] == [0, 1]


def test_merge_export_true_ignores_runtime():
    obj = _bare_instance(
        rbln_config={'tensor_parallel_size': 8},
        rbln_runtime_config={'device': [0]},
        rbln_export=True,
    )
    merged = obj._merge_rbln_config()
    assert merged == {'tensor_parallel_size': 8}


# ----------------------------------------------------------------------
# _from_pretrained: cached load
# ----------------------------------------------------------------------

def _compiled_dir(tmp_path):
    d = tmp_path / 'Model-Compiled'
    d.mkdir()
    (d / 'compiled.rbln').write_bytes(b'')  # marks it as a compiled artifact
    return str(d)


def test_cached_load_forces_export_false_and_drops_compile_config(tmp_path):
    fake = _FakeRBLN()
    obj = _bare_instance(
        model_path=_compiled_dir(tmp_path),
        rbln_config={'tensor_parallel_size': 8, 'max_seq_len': 114688},
        rbln_runtime_config={},
        rbln_export=None,
    )
    obj._from_pretrained(fake)
    assert fake.captured['export'] is False
    # compile-time rbln_config MUST NOT be re-passed on a cached load
    assert 'rbln_config' not in fake.captured


def test_cached_load_passes_only_runtime_config(tmp_path):
    fake = _FakeRBLN()
    obj = _bare_instance(
        model_path=_compiled_dir(tmp_path),
        rbln_config={'tensor_parallel_size': 8},
        rbln_runtime_config={'device': [0, 1]},
        rbln_export=None,
    )
    obj._from_pretrained(fake)
    assert fake.captured['export'] is False
    assert fake.captured['rbln_config'] == {'device': [0, 1]}
    # the compile-time tensor_parallel_size is absent
    assert 'tensor_parallel_size' not in fake.captured['rbln_config']


# ----------------------------------------------------------------------
# _from_pretrained: fresh compile
# ----------------------------------------------------------------------

def _fresh_dir(tmp_path):
    d = tmp_path / 'Model-Source'
    d.mkdir()  # no *.rbln -> not a compiled artifact
    return str(d)


def test_fresh_compile_passes_merged_compile_config(tmp_path):
    fake = _FakeRBLN()
    obj = _bare_instance(
        model_path=_fresh_dir(tmp_path),
        rbln_config={'tensor_parallel_size': 8, 'visual': {'max_seq_lens': 6400}},
        rbln_runtime_config={},
        rbln_export=None,
    )
    obj._from_pretrained(fake)
    # fresh dir -> export not forced to False; compile config flows through
    assert fake.captured.get('export') is None
    assert fake.captured['rbln_config']['tensor_parallel_size'] == 8
    assert fake.captured['rbln_config']['visual'] == {'max_seq_lens': 6400}
