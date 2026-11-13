import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def load_video_infer_data():
    run_path = Path(__file__).resolve().parents[1] / 'vlmeval' / 'inference_video.py'
    run_tree = ast.parse(run_path.read_text(encoding='utf-8'))
    function = next(
        node for node in run_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'infer_data'
    )
    namespace = {
        'np': SimpleNamespace(all=all),
        'osp': SimpleNamespace(exists=lambda path: path.endswith('_PREV.pkl')),
        'get_rank_and_world_size': lambda: (1, 2),
        'resolve_dataset_alias_name': lambda name, alias: alias or name,
        'load': lambda _: {0: 'cached zero', 1: 'cached one'},
        'dump': Mock(),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(run_path), 'exec'), namespace)
    return namespace['infer_data'], namespace['dump']


def test_video_resume_writes_completed_rank_partition(tmp_path):
    infer_data, dump = load_video_infer_data()

    class DatasetData:
        def __getitem__(self, key):
            if key == 'index':
                return [0, 1]
            raise KeyError(key)

        def __len__(self):
            return 2

    dataset = SimpleNamespace(dataset_name='Example', data=DatasetData(), pack=False)
    model = object()
    out_file = str(tmp_path / 'rank-1.pkl')

    result = infer_data(model, 'mock', str(tmp_path), dataset, out_file)

    assert result is model
    dump.assert_called_once_with({1: 'cached one'}, out_file)
