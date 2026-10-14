import json

import pandas as pd
import pytest

from vlmeval.dataset import etbench as etb
from vlmeval.smp import dump


@pytest.mark.parametrize('answer, expected', [
    ('The event happens in 10.2 - 12.8 seconds.', [(10.2, 12.8)]),
    ('It happens from 5 to 9 seconds.', [(5.0, 9.0)]),
    ('The event starts at 4 seconds.\nIt ends at 8 seconds.', [(4.0, 8.0)]),
    ('The event starts at 4s.', None),  # no word boundary in "4s", as in the official parser
    ('12.8 - 10.2', [(10.2, 12.8)]),
    ('No idea.', None),
])
def test_parse_spans(answer, expected):
    assert etb.parse_spans(answer) == expected


def test_parse_events():
    times, caps = etb.parse_events('0 - 5 seconds, add salt.\n5 - 10 seconds, stir the soup.')
    assert times == [[0.0, 5.0], [5.0, 10.0]] and caps == ['add salt.', 'stir the soup.']
    # A single time t is read as [t - 20, t + 20].
    assert etb.parse_events('At 25 seconds, add salt.') == ([[5.0, 45.0]], ['add salt.'])
    assert etb.parse_events('nothing here') == (None, None)


def test_score_grounding():
    samples = [dict(tgt=[0, 10], a='0 - 10 seconds'), dict(tgt=[0, 10], a='5 - 15 seconds')]
    res = etb.score_grounding(samples)  # IoUs 1 and 1/3
    assert res['F1'] == pytest.approx((1 + 1 + 0.5 + 0.5) / 4)
    assert res['mIoU'] == pytest.approx(0.66667)


def test_score_highlight():
    samples = [dict(tgt=[[[2, 4]], [[10, 12]]], a='The highlight moment happens in 11s.'),
               dict(tgt=[[2, 4]], a='none')]
    assert etb.score_highlight(samples) == dict(Total=2, Failed=1, F1=0.5)


def test_score_action_localization_and_summarization():
    tal = etb.score_action_localization([dict(tgt=[[0, 10], [20, 30]], a='0 - 10, 50 - 60 seconds')])
    assert tal['F1'] == pytest.approx(0.5)  # precision 1/2 and recall 1/2 at every threshold
    evs = etb.score_summarization([dict(tgt=[[0, 10]], a='0 - 5 seconds')])
    assert evs['F1'] == pytest.approx(0.66667)  # recall 1/2, precision 1


class FirstOption:
    def best_option(self, answer, options):
        return 0


def test_score_choice():
    o = ['a cat', 'a dog', 'a bird', 'a fish']
    samples = [dict(o=o, p=1, a=a) for a in ('Best Option: (B)', 'B. a dog', '', 'the second one')]
    res = etb.score_choice(samples, FirstOption())
    assert res == dict(Total=4, Failed=2, Acc=0.5)


def test_evaluate(tmp_path):
    rows = [
        dict(task='tvg', source='charades_sta', tgt=[0, 10], prediction='The event happens in 0 - 10 seconds.'),
        dict(task='tvg', source='charades_sta', tgt=[0, 10], prediction='I do not know.'),
        dict(task='tal', source='thumos14', tgt=[[0, 10]], prediction='The action happens in 0 - 10 seconds.'),
        dict(task='rar', source='perception_test', p=1, o=['a cat', 'a dog'], prediction='Best Option: (B)'),
    ]
    data = pd.DataFrame([
        dict(index=i, idx=i, video=f'v{i}', question='q', duration=30.0, g='', src='', task=r['task'],
             source=r['source'], prediction=r['prediction'], tgt=json.dumps(r['tgt']) if 'tgt' in r else '',
             p=r.get('p'), o=json.dumps(r['o']) if 'o' in r else '') for i, r in enumerate(rows)
    ])
    eval_file = tmp_path / 'model_ETBench_1fps.xlsx'
    dump(data, str(eval_file))
    metrics = object.__new__(etb.ETBench).evaluate(str(eval_file))
    assert metrics['TVG_F1'] == 50.0 and metrics['TAL_F1'] == 100.0
    assert metrics['RAR_Acc'] == 100.0  # the answer index is read back from the file as a number
    assert metrics['per_source']['tvg_charades_sta']['Failed'] == 1
    assert metrics['num_samples'] == 4


def test_broken_videos_are_repaired_once(tmp_path, monkeypatch):
    data_file = tmp_path / 'ETBench.tsv'
    dump(pd.DataFrame(dict(index=[0], video=['summe/cooking'], question=['q'])), str(data_file))
    (tmp_path / 'summe').mkdir()
    (tmp_path / 'summe' / 'cooking.mp4').write_bytes(b'video')
    calls = []

    def fake_reencode(path):
        calls.append(path.name)
        path.rename(path.with_suffix('.orig.mp4'))
        path.write_bytes(b'fixed')

    monkeypatch.setattr(etb.ETBench, '_reencode', staticmethod(fake_reencode))
    dataset = object.__new__(etb.ETBench)
    dataset._repair_videos(tmp_path, data_file)
    dataset._repair_videos(tmp_path, data_file)  # the kept original marks the video as repaired
    assert calls == ['cooking.mp4']
