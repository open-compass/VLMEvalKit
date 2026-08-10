from vlmeval.dataset.c4_bench import normalize_answer, parse_task_answer


def test_c4_direct_answer_parsing():
    assert normalize_answer(' “一叶障目”。 ') == '一叶障目'
    assert parse_task_answer('H0', '一叶障目') == ('一叶障目', None)


def test_c4_explanation_json_and_recovery():
    valid = '```json\n{"answer": "一叶障目", "explanation": "..."}\n```'
    assert parse_task_answer('E0', valid) == ('一叶障目', True)
    assert parse_task_answer('E0', '推理很长，最终答案：一叶障目') == ('一叶障目', False)
