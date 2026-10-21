from vlmeval.dataset.video_odyssey import VideoOdysseyV, _extract_answer


def test_default_judge_model_is_exact_matching():
    assert VideoOdysseyV.DEFAULT_JUDGE_MODEL == 'exact_matching'


def test_extract_answer_uses_response_after_think():
    assert _extract_answer('<think>answer: B</think>\nC') == 'C'


def test_extract_answer_does_not_treat_article_as_option_a():
    assert _extract_answer('This is a difficult case. I choose C.') == 'C'


def test_extract_answer_prefers_last_explicit_choice():
    prediction = 'Option B seems plausible, but the final answer is D.'
    assert _extract_answer(prediction) == 'D'


def test_extract_answer_supports_json_and_fenced_json():
    assert _extract_answer('{"predicted_answer": "B"}') == 'B'
    assert _extract_answer('```json\n{"answer": "C"}\n```') == 'C'


def test_extract_answer_rejects_failure_message():
    assert _extract_answer('Failed to obtain answer via API.') == ''
