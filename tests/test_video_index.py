import random

import pandas as pd

from vlmeval.dataset import video_index
from vlmeval.dataset.utils import video_index as vi
from vlmeval.dataset.video_index import VideoIndex

OPTIONS = ['a red cup', 'a knife', 'the phone']


def test_video_index_reply_parsing():
    assert vi.extract_letter('B', OPTIONS) == 'B'
    assert vi.extract_letter('(c) the phone', OPTIONS) == 'C'
    assert vi.extract_letter('The answer is B.', OPTIONS) == 'B'
    assert vi.extract_letter('The answer is A. Wait, the answer is C', OPTIONS) == 'C'
    assert vi.extract_letter('the phone', OPTIONS) == 'C'
    assert vi.extract_letter('A person picks up the phone', OPTIONS) == 'C'
    assert vi.extract_letter('I cannot tell', OPTIONS) is None
    assert vi.extract_letter('', OPTIONS) is None
    assert vi.score_reply('C', 'C', OPTIONS) == 1.0
    assert vi.score_reply('I cannot tell', 'C', OPTIONS) == 0.0


def test_video_index_permutations_are_fixed_per_item():
    perms = vi.permutations_for('Demo_1', 4)
    rng = random.Random('42|Demo_1')
    assert perms == [rng.sample(range(4), 4) for _ in range(4)]
    assert all(sorted(perm) == [0, 1, 2, 3] for perm in perms)


def test_video_index_frame_rule():
    assert vi.one_fps_indices(40, 2.0, 20.0) == list(range(0, 40, 2))
    assert vi.one_fps_indices(10, 0.5, 20.0) == list(range(10))
    indices = vi.one_fps_indices(2000, 2.0, 1000.0, cap=512)
    assert len(indices) == 512 and indices[0] == 0 and indices[-1] == 1998
    assert len(vi.one_fps_indices(40, 2.0, 20.0, cap=8)) == 8


def test_video_index_blind_rows_are_averaged_per_item(monkeypatch):
    groups = ['perception', 'temporal']
    rows = []
    for item in range(2):
        for perm_idx, perm in enumerate(vi.permutations_for(f'Demo_{item}', 3)):
            answer = vi.LETTERS[perm.index(2)]
            # item 0: every reply names the marked option; item 1: one reply of four does
            correct = item == 0 or perm_idx == 0
            rows.append({
                'index': len(rows),
                'item_id': f'Demo_{item}',
                'capability_group': groups[item],
                'candidates': repr([OPTIONS[j] for j in perm]),
                'answer': answer,
                'prediction': answer if correct else 'I cannot tell',
            })
    dumped = {}
    monkeypatch.setattr(video_index, 'load', lambda _path: pd.DataFrame(rows))
    monkeypatch.setattr(video_index, 'dump', lambda data, path: dumped.update({path: data}))
    monkeypatch.setattr(video_index, 'get_intermediate_file_path', lambda path, suffix, _ext=None: f'{path}{suffix}')

    dataset = VideoIndex.__new__(VideoIndex)
    rating = dataset.evaluate('pred.xlsx')

    assert rating['Overall'] == 62.5
    assert rating['Perception'] == 100.0 and rating['Temporal'] == 25.0
    assert 'Spatial' not in rating
    assert rating['Items'] == 2 and rating['Replies'] == 8 and rating['Replies without an option'] == 3
    assert VideoIndex.report_primary_metric(rating) == {'Overall': 62.5}


class _Frame:

    def __init__(self, value):
        self.value = value

    def asnumpy(self):
        return self.value


class _Reader:
    """Stands for a decord VideoReader; counts the frames decoded in stream order."""

    def __init__(self):
        self.position = None

    def seek(self, position):
        self.position = position

    def next(self):
        self.position += 1
        return _Frame(self.position - 1)


def test_video_index_frames_are_decoded_in_stream_order():
    reader = _Reader()
    assert vi.read_frames(reader, [0, 3, 7]) == [0, 3, 7]
    assert reader.position == 8
    assert vi.read_frames(reader, []) == []
