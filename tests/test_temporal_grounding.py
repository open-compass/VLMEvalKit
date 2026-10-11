import pytest

from vlmeval.dataset.utils import temporal_grounding as tg


@pytest.mark.parametrize('text, expected', [
    ('The event happens in 18.5 - 23.0 seconds.', [(18.5, 23.0)]),
    ('The event happens from 18 to 23 seconds.', [(18.0, 23.0)]),
    ('It starts at 01:05 and ends at 01:12.5.', [(65.0, 72.5)]),
    ('Between 1:02:03 and 1:02:10', [(3723.0, 3730.0)]),
    ('[[12.0, 30.5], [40, 52]]', [(12.0, 30.5), (40.0, 52.0)]),
    ('Starting time: 0.8 seconds. Ending time: 1.1 seconds', [(0.8, 1.1)]),
    ('I cannot find it.', []),
])
def test_extract_time_spans(text, expected):
    assert tg.extract_time_spans(text) == expected


def test_temporal_iou():
    assert tg.temporal_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert tg.temporal_iou((0, 10), (0, 10)) == 1.0
    assert tg.temporal_iou((0, 4), (6, 9)) == 0.0
    assert tg.temporal_iou((10, 5), (5, 15)) == 0.0  # invalid prediction
    assert tg.temporal_iou(None, (5, 15)) == 0.0


def test_recall_at():
    r = tg.recall_at([0.2, 0.35, 0.6, 0.9])
    assert r == {'R1@0.3': 75.0, 'R1@0.5': 50.0, 'R1@0.7': 25.0}


def test_sampled_frame_times_matches_video_base_sampling():
    # 10 s at 30 fps sampled at 2 fps -> frames 0, 15, 30, ... -> 0.0, 0.5, 1.0, ... s
    times = tg.sampled_frame_times(300, 30.0, fps=2.0)
    assert len(times) == 20 and times[:3] == [0.0, 0.5, 1.0]
    # 4 uniform frames of a 300-frame video -> frames 60, 120, 180, 240
    assert tg.sampled_frame_times(300, 30.0, nframe=4) == [2.0, 4.0, 6.0, 8.0]


def test_frames_with_timestamps():
    msg = tg.frames_with_timestamps(['a.jpg', 'b.jpg'], [0.0, 0.5])
    assert msg == [
        dict(type='text', value='0.0'),
        dict(type='image', value='a.jpg'),
        dict(type='text', value='0.5'),
        dict(type='image', value='b.jpg')
    ]


def test_normalize_query_matches_official_loader():
    assert tg.normalize_query('  A man  opens\nthe door. ') == 'A man opens the door'
    assert tg.normalize_query('A person is laughing...') == 'A person is laughing'


def test_round_spans_matches_official_rounding():
    # Python round() sends halves to the even number, as in the official evaluation
    assert tg.round_spans([(2.5, 7.5), (0.4, 9.6)]) == [(2, 8), (0, 10)]
    assert tg.temporal_iou(tg.round_spans([(2.5, 7.5)])[0], (0, 10)) == pytest.approx(0.6)
