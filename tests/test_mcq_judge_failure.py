import re

import pandas as pd

from vlmeval.dataset.utils.multiple_choice import (extract_answer_from_item, mcq_circular_eval,
                                                   mcq_vanilla_eval)

FAIL_MSG = 'Failed to obtain answer via API.'


class DownJudge:

    def generate(self, prompt, **kwargs):
        return FAIL_MSG


class OptionJudge:
    """Answers with the label whose option text is '2', i.e. what 'two apples' means."""

    def generate(self, prompt, **kwargs):
        return re.search(r'^([A-Z])\. 2$', prompt, re.M).group(1)


def apples(index, answer, options=('1', '2', '3', '4'), prediction='There are two apples.'):
    row = dict(index=index, question='How many apples?', answer=answer, prediction=prediction)
    row.update(zip('ABCD', options))
    return row


def test_judge_outage_gives_z_not_a_random_option():
    item = pd.Series(apples(0, 'B', prediction="I can't tell."))

    picks = {extract_answer_from_item(DownJudge(), item)['opt'] for _ in range(50)}
    res = extract_answer_from_item(DownJudge(), item)

    assert picks == {'Z'}
    assert res['judge_failed']


def test_unmatched_judge_answer_is_z_and_final():
    class UnsureJudge:
        def generate(self, prompt, **kwargs):
            return 'none of them'

    res = extract_answer_from_item(UnsureJudge(), pd.Series(apples(0, 'B', prediction="I can't tell.")))

    assert res['opt'] == 'Z'
    assert not res.get('judge_failed')


def test_judge_outage_with_a_real_z_option_does_not_crash():
    item = pd.Series(dict(apples(0, 'B', prediction="I can't tell."), Z='none of these'))

    assert extract_answer_from_item(DownJudge(), item)['opt'] == 'Z'


def test_vanilla_rerun_judges_failed_items_again(tmp_path):
    data = pd.DataFrame([apples(i, 'B') for i in range(4)])
    meta = data[['index', 'answer']]
    cache = str(tmp_path / 'judge.pkl')

    down = mcq_vanilla_eval(DownJudge(), data.copy(), meta, 1, cache, 'MMBench_DEV_EN')
    back = mcq_vanilla_eval(OptionJudge(), data.copy(), meta, 1, cache, 'MMBench_DEV_EN')

    assert down['hit'].tolist() == [0, 0, 0, 0]
    assert back['hit'].tolist() == [1, 1, 1, 1]


def test_vanilla_rerun_replaces_random_results_from_older_versions(tmp_path):
    data = pd.DataFrame([apples(0, 'B')])
    cache = str(tmp_path / 'judge.pkl')
    pd.to_pickle({0: dict(hit=1, log='Match Log: Failed to predict, thus randomly generate one. . ')}, cache)

    back = mcq_vanilla_eval(OptionJudge(), data, data[['index', 'answer']], 1, cache, 'MMBench_DEV_EN')

    assert back['log'].iloc[0] == 'Match Log: B. '


def test_circular_rerun_judges_failed_groups_again(tmp_path):
    # one question asked twice, with the options rotated
    data = pd.DataFrame([apples(1, 'B'), apples(1000001, 'A', options=('2', '3', '4', '1'))])
    meta = data[['index', 'answer']]
    cache = str(tmp_path / 'judge.pkl')

    down = mcq_circular_eval(DownJudge(), data.copy(), meta, 1, cache, 'MMBench_DEV_EN')
    back = mcq_circular_eval(OptionJudge(), data.copy(), meta, 1, cache, 'MMBench_DEV_EN')

    assert down['hit'].tolist() == [0]
    assert back['hit'].tolist() == [1]
