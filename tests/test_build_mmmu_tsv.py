import importlib.util
from pathlib import Path


def _load_builder():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'build_mmmu_tsv.py'
    spec = importlib.util.spec_from_file_location('build_mmmu_tsv', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sanitize_text_restores_latex_escapes():
    builder = _load_builder()

    text = '$\x0crac{a}\x08egin{b}\\right\t imes\\left[1\r\n2\right]$'

    assert builder.sanitize_text(text) == '$\\frac{a}\\begin{b}\\right\\t imes\\left[1\n2\\right]$'


def test_drop_trailing_empty_options():
    builder = _load_builder()

    assert builder.drop_trailing_empty_options(['A', 'B', 'None', '']) == ['A', 'B']
    assert builder.drop_trailing_empty_options(['None of the answers are correct']) == [
        'None of the answers are correct'
    ]
