import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GROUPS = {
    'choice': ['XLRS-Bench-lite', 'MMMU_DEV_VAL', 'MathVista_MINI', 'AI2D_TEST', 'MMStar'],
    'bool': ['HallusionBench'],
    'score': ['MMVet'],
    'text': ['OCRBench'],
    'extended': [
        'hle_verified', 'MMMU_Pro_10c', 'ChartQAPro', 'SciDocBench',
        'MicroVQA', 'SFE', 'OmniMat_QA', 'OmniMat_CAL',
    ],
}


@pytest.mark.parametrize('datasets', [[], ['SFE']])
def test_launcher_routes_datasets_through_one_checkout_server(tmp_path, datasets):
    """Exercise the actual launcher with a small runner double and real HTTP server."""
    shim_dir = tmp_path / 'bin'
    shim_dir.mkdir()
    shim = shim_dir / 'python'
    shim.write_text(f'#!{sys.executable}\n' + textwrap.dedent('''\
        import json
        import os
        import sys
        import urllib.request
        from pathlib import Path

        args = sys.argv[1:]
        if args[:2] != ['-u', 'run.py']:
            os.execv(sys.executable, [sys.executable, *args])

        def option(name):
            return args[args.index(name) + 1]

        dataset = option('--data')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for stage, routing, endpoint in [
            ('infer', '--extra-body', '--base-url'),
            ('judge', '--judge-args', '--judge-base-url'),
        ]:
            payload = json.loads(option(routing))
            assert payload == {'mock_dataset': dataset, 'mock_stage': stage}
            payload.update(model='mock-model', messages=[{'role': 'user', 'content': 'hello'}])
            request = urllib.request.Request(
                option(endpoint) + '/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'},
            )
            with opener.open(request, timeout=5) as response:
                assert json.load(response)['choices'][0]['message']['content']

        output = Path(option('--work-dir'))
        output.mkdir(parents=True, exist_ok=True)
        (output / f'{dataset}.json').write_text(json.dumps({
            'args': args, 'prediction_format': os.environ['PRED_FORMAT'],
        }))
    '''))
    shim.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    output = tmp_path / 'run'
    env = dict(os.environ, PATH=f'{shim_dir}:{os.environ["PATH"]}',
               EVAL_ROOT=str(output), MOCK_PORT=str(port))
    for name in ('MOCK_SERVER', 'MOCK_DATA_CONFIG'):
        env.pop(name, None)
    result = subprocess.run(
        ['bash', '.github/scripts/run_mock_api_tests.sh', *datasets],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {'extended': datasets} if datasets else GROUPS
    actual = {}
    from vlmeval.smp.dataset_alias import resolve_dataset_spec

    for group, names in expected.items():
        paths = list((output / f'mock_api_{group}').glob('*.json'))
        assert {path.stem for path in paths} == set(names)
        for path in paths:
            record = json.loads(path.read_text())
            assert record['prediction_format'] == 'tsv'
            args = record['args']
            config = json.loads(args[args.index('--data-config') + 1])
            for name in GROUPS['extended']:
                spec = resolve_dataset_spec(name, config)
                assert spec.dataset_name == name
            assert config['hle_verified']['skip_noimg'] is True
            assert config['SciDocBench']['enable_reasoning_diagnostic'] is True
            assert config['OmniMat_QA']['skip_noimg'] is False
            assert config['OmniMat_CAL']['skip_noimg'] is False
            assert '--api-mode' in args
            assert '--local-media' in args
            actual[path.stem] = {'infer': 1, 'judge': 1}
    health = json.loads((output / 'logs/mock_requests.json').read_text())
    assert health['requests'] == actual
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', port)) != 0


def test_comparator_detects_extended_dataset_regressions(tmp_path):
    baseline = tmp_path / 'baseline'
    current = tmp_path / 'current'
    for root in (baseline, current):
        for group in GROUPS:
            model = root / f'mock_api_{group}' / 'mock--mock-api'
            model.mkdir(parents=True)
            (model / 'score.csv').write_text('dataset,score\nexample,0.5\n')
    command = ['bash', '.github/scripts/compare_mock_api_baseline.sh',
               str(current), str(baseline)]
    env = dict(os.environ, PATH=f'{Path(sys.executable).parent}:{os.environ["PATH"]}')
    passed = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    extended = current / 'mock_api_extended/mock--mock-api/score.csv'
    extended.write_text('dataset,score\nexample,0.6\n')
    failed = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert 'Comparing mock_api_extended' in failed.stdout
    assert 'Differences in score.csv' in failed.stdout
