import pytest

from vlmeval.dataset.vue_tr import vue_tr_auc, vue_tr_query_scores


def test_query_scores_floor_ceil_and_merge():
    # predictions are floored / ceiled: [9.6, 20.2] -> [9, 21]
    iou, precision, recall = vue_tr_query_scores([(9.6, 20.2)], [[10, 20]])
    assert iou == pytest.approx(10 / 12)
    assert precision == pytest.approx(10 / 12)
    assert recall == pytest.approx(1.0)
    # overlapping predictions are merged for IoU
    assert vue_tr_query_scores([(0, 6), (5, 10)], [[0, 10]])[0] == pytest.approx(1.0)


def test_query_scores_empty_prediction():
    iou, precision, recall = vue_tr_query_scores([], [[0, 10]])
    assert iou == 0.0 and precision is None and recall == 0.0
    # version 2: no prediction for no ground truth counts as precise
    assert vue_tr_query_scores([], [], version=1)[1] is None
    assert vue_tr_query_scores([], [], version=2)[1] == 1.0


def test_auc():
    # Trapezoid over t = 0, 0.01, ..., 1 (as in the official script).
    res = vue_tr_auc([1.0, 0.0], [1.0, None], [1.0, 0.0])
    # IoU > t holds for half of the queries for t < 1 and for none at t = 1
    assert res['IoU'] == pytest.approx(49.75)
    # precision >= t for every t; the undefined (None) precision is left out
    assert res['Precision'] == pytest.approx(100.0)
    # recall >= t holds for both queries at t = 0 and for one of them for t > 0
    assert res['Recall'] == pytest.approx(50.25)


def test_primary_metric_nested_and_flattened():
    from vlmeval.dataset.vue_tr import VUETR
    assert VUETR.report_primary_metric({'overall': {'IoU': 32.4, 'Precision': 39.7}}) == {'IoU': 32.4}
    assert VUETR.report_primary_metric({'overall|IoU': 32.4, 'overall|Precision': 39.7}) == {'IoU': 32.4}


def test_reversed_interval_counts_for_precision_and_recall():
    # qa_eval.py reads [2, 1] as [1, 2] for precision / recall; IoU drops it
    iou, precision, recall = vue_tr_query_scores([(2.0, 1.0)], [[0, 4]])
    assert iou == 0.0
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.25)


def test_tsv_is_reused_until_more_videos_arrive(tmp_path, monkeypatch):
    import json
    import os

    from vlmeval.dataset.vue_tr import VUETR
    from vlmeval.smp import load

    monkeypatch.setenv('LMUData', str(tmp_path))
    root = tmp_path / 'VUE-TR'
    (root / 'videos').mkdir(parents=True)
    gt = [dict(query_id=i, video_id=f'v{i}', query='q', gt=[[0, 1]], duration=60.0, duration_category='short',
               query_format='keyword', query_modality='vision') for i in range(2)]
    (root / 'VUE-TR_ground_truth.json').write_text(json.dumps(gt))
    (root / 'videos' / 'v0.mp4').touch()

    dataset = object.__new__(VUETR)
    data_file = dataset.prepare_dataset('VUE-TR')['data_file']
    assert len(load(data_file)) == 1
    os.utime(data_file, (1e10, 1e10))  # mark the file so a rewrite would show
    with pytest.warns(UserWarning):
        dataset.prepare_dataset('VUE-TR')
    assert os.stat(data_file).st_mtime == 1e10  # reused, not written again

    (root / 'videos' / 'v1.mp4').touch()
    dataset.prepare_dataset('VUE-TR')
    assert len(load(data_file)) == 2  # rebuilt once the second video is there
