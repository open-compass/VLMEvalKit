import json

import pandas as pd
import pytest

from vlmeval.dataset import etbench as etb
from vlmeval.smp import dump, load


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


def test_build_tsv_keeps_both_prompt_styles(tmp_path):
    txt = [dict(idx=0, task='eca', source='a', video='a/v0.mp4', duration=10, q='around 4.6 seconds', src=[4.574],
                p=1, o=['1.0 - 2.0 seconds', '3.0 - 4.0 seconds']),
           dict(idx=0, task='tvg', source='b', video='b/v1.mp4', duration=20, q='Answer in seconds.', tgt=[1, 2])]
    vid = [dict(a, q=a['q'].replace('4.6 seconds', '<vid>')) for a in txt]
    vid[0]['o'] = ['1.0 - 2.0', '3.0 - 4.0']
    vid[1]['q'] = 'Answer.'
    for name, annos in (('txt.json', txt), ('vid.json', vid), ('subset.json', dict(tvg=dict(b=[0])))):
        (tmp_path / name).write_text(json.dumps(annos))
    data_file = tmp_path / 'ETBench.tsv'
    etb.ETBench._build_tsv(tmp_path / 'txt.json', tmp_path / 'vid.json', tmp_path / 'subset.json', data_file)
    data = load(str(data_file))
    assert list(data['question']) == ['around 4.6 seconds', 'Answer in seconds.']
    assert list(data['question_vid']) == ['around <vid>', 'Answer.']
    assert json.loads(data['src'][0]) == [4.574]
    assert json.loads(data['o'][0]) == txt[0]['o'] and json.loads(data['o_vid'][0]) == vid[0]['o']
    annotations = (tmp_path / 'txt.json', tmp_path / 'vid.json')
    assert etb.ETBench._tsv_up_to_date(data_file, annotations)
    dump(data.drop(columns=['question_vid']), str(data_file))  # a file built before the column existed
    assert not etb.ETBench._tsv_up_to_date(data_file, annotations)

    subset_file = tmp_path / 'ETBench-Subset.tsv'
    etb.ETBench._build_tsv(tmp_path / 'txt.json', tmp_path / 'vid.json', tmp_path / 'subset.json', subset_file,
                           subset=True)
    assert list(load(str(subset_file))['question_vid']) == ['Answer.']


def test_vid_prompt_style():
    dataset = object.__new__(etb.ETBench)
    dataset.data_root, dataset.fps, dataset.nframe = '/videos', 1.0, 0
    line = pd.Series(dict(video='a/v0', task='rar', question='around 4.6 seconds', question_vid='around <vid>',
                          src='[4.574]'))
    dataset.prompt_style = 'txt'
    assert dataset.build_prompt(line, video_llm=True)[1] == dict(type='text', value='around 4.6 seconds')
    dataset.prompt_style = 'vid'
    video, text = dataset.build_prompt(line, video_llm=True)
    assert video == dict(type='video', value='/videos/a/v0.mp4', fps=1.0)
    assert text == dict(type='text', value='around <vid>', task='rar', src=[4.574])
    no_src = dataset.build_prompt(pd.concat([line.drop('src'), pd.Series(dict(src=float('nan')))]), video_llm=True)
    assert no_src[1]['src'] is None  # an empty cell is read back as NaN
    with pytest.raises(ValueError):
        dataset.build_prompt(line, video_llm=False)


class SameWords:
    def best_option(self, answer, options):  # the option that shares the most words with the answer
        return max(range(len(options)), key=lambda i: len(set(answer.split()) & set(options[i].split())))


def test_vid_style_is_scored_with_its_own_options(tmp_path, monkeypatch):
    monkeypatch.setattr(etb, 'SentenceSimilarity', SameWords)
    rows = [dict(index=0, idx=0, video='v0', question='q', duration=30.0, task='eca', source='charades_sta',
                 prediction='Best Option: (A) red', p=1, o=json.dumps(['red', 'blue']),
                 o_vid=json.dumps(['green', 'red']), tgt='', g='', src='')]
    eval_file = tmp_path / 'model_ETBench_1fps_vid.xlsx'
    dump(pd.DataFrame(rows), str(eval_file))
    dataset = object.__new__(etb.ETBench)
    dataset.prompt_style = 'txt'
    assert dataset.evaluate(str(eval_file))['ECA_Acc'] == 0.0
    dataset.prompt_style = 'vid'  # the wrong letter is matched to the most similar option text: "red"
    assert dataset.evaluate(str(eval_file))['ECA_Acc'] == 100.0
