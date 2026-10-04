import pytest

from vlmeval.dataset.momentseeker import MomentSeeker


def test_r1_uses_first_interval_only():
    gts = [[10, 20]]
    assert MomentSeeker.score_one([(10, 20), (50, 60)], gts)[0] == 1.0
    assert MomentSeeker.score_one([(50, 60), (10, 20)], gts)[0] == 0.0
    assert MomentSeeker.score_one([], gts) == (0.0, 0.0)


def test_r1_threshold_is_strict():
    # IoU of (0, 3) with (0, 10) is exactly 0.3, which does not exceed the threshold
    assert MomentSeeker.score_one([(0, 3)], [[0, 10]])[0] == 0.0
    assert MomentSeeker.score_one([(0, 3.1)], [[0, 10]])[0] == 1.0


def test_ap5_ranking_and_one_to_one_matching():
    gts = [[10, 20], [40, 50]]
    # hits at ranks 1 and 3 -> precisions 1/1 and 2/3
    assert MomentSeeker.score_one([(10, 20), (70, 80), (40, 50)], gts)[1] == pytest.approx((1 + 2 / 3) / 2)
    # the same ground truth cannot be matched twice
    assert MomentSeeker.score_one([(10, 20), (11, 20)], [[10, 20]])[1] == pytest.approx(1.0)
    assert MomentSeeker.score_one([(11, 20), (10, 20)], [[10, 20]])[1] == pytest.approx(1.0)
    # only the first 5 intervals count
    assert MomentSeeker.score_one([(0, 1)] * 5 + [(10, 20)], [[10, 20]])[1] == 0.0


def test_primary_metric_nested_and_flattened():
    assert MomentSeeker.report_primary_metric({'overall': {'R@1': 20.5, 'mAP@5': 21.0}}) == {'R@1': 20.5}
    assert MomentSeeker.report_primary_metric({'overall|R@1': 20.5, 'overall|mAP@5': 21.0}) == {'R@1': 20.5}
