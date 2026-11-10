import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "assert_score.py"


def run_assert_score(dataset, base_score):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            dataset,
            "--base_score",
            base_score,
            "--model-name",
            "test-model",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_invalid_base_score_returns_nonzero():
    result = run_assert_score("MMBench_V11_MINI", "not-a-dictionary")

    assert result.returncode != 0
    assert "Parameter parsing error:" in result.stderr


def test_empty_dataset_list_returns_nonzero():
    result = run_assert_score("   ", "{}")

    assert result.returncode != 0
    assert "must contain at least one dataset" in result.stderr
