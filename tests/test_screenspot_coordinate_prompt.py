import ast
import re
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


def load_screenspot():
    source = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/GUI/screenspot.py'
    nodes = []
    names = {'SYSTEM_PROMPT', 'SYSTEM_PROMPT_V2', 'USER_INSTRUCTION', 'USER_INSTRUCTION_V2'}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets):
            nodes.append(node)
        elif (isinstance(node, ast.FunctionDef)
              and node.name in ('parse_bbox_aguvis', 'convert_bbox')):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'ScreenSpot':
            nodes.append(node)
    namespace = {'ImageBaseDataset': object, 'pd': pd, 'ast': ast, 're': re, 'Image': Image}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('re_type', ['functional', 'composite'])
@pytest.mark.parametrize('indexed', [False, True])
def test_screenspot_prompts_state_fractional_coordinate_units(re_type, indexed):
    module = load_screenspot()
    cls = module['ScreenSpot']
    dataset = cls.__new__(cls)
    dataset.RE_TYPE = re_type
    dataset.data = pd.DataFrame([{'question': 'Activate the button', 'description': 'Save'}])
    dataset.dump_image = lambda line: ['screenshot.png']
    messages = dataset.build_prompt(0 if indexed else dataset.data.iloc[0])
    instruction = messages[-1]['value']
    assert 'between 0 and 1' in instruction
    assert 'image width' in instruction and 'image height' in instruction
    assert 'pyautogui.click' in instruction
    assert messages[1] == {'type': 'image', 'value': 'screenshot.png'}
    assert dataset.data.iloc[0]['question'] == 'Activate the button'


def test_fractional_point_uses_the_same_units_as_normalized_ground_truth(tmp_path):
    module = load_screenspot()
    image = tmp_path / 'screenshot.png'
    Image.new('RGB', (100, 200)).save(image)
    bbox = module['convert_bbox']([20, 40, 10, 10], str(image))
    point = module['parse_bbox_aguvis']('pyautogui.click(x=0.25, y=0.22)')
    assert bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
