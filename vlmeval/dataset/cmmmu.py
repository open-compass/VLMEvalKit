from .image_base import ImageBaseDataset
import re
import random
from collections import Counter


def get_multi_choice_prediction(response, all_choices, index2ans):
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    candidates = []

    for choice in all_choices:  # (A) (B) (C) (D)
        # Add the choice to candidates each time it appears in the response
        candidates.extend([choice for _ in range(response.count(f'({choice})'))])

    if len(candidates) == 0:
        for choice in all_choices:  # A B C D
            # Similarly, add the choice for each occurrence
            candidates.extend([choice for _ in range(response.count(f'{choice}'))])

    if len(candidates) == 0 and len(response.split()) >= 1:
        for index, ans in index2ans.items():
            # Add index for each occurrence of ans in response
            candidates.extend([index for _ in range(response.count(ans))])

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) >= 1:
        for index, ans in index2ans.items():
            if ans in response:
                candidates.append(index)
                # index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        return random.choice(all_choices)
        # return ''
    else:
        # Count the occurrence of each candidate
        candidate_counts = Counter(candidates)

        # Select the most frequent candidates
        max_count = max(candidate_counts.values())
        most_frequent_candidates = [c for c in all_choices if candidate_counts.get(c, 0) == max_count]

        # Combine the most frequent candidates in ABCD order
        return ''.join(most_frequent_candidates)


def extract_numbers(string):
    # Pattern for numbers with Chinese commas
    pattern_commas = r'-?\d{1,3}(?:，\d{3})+'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without Chinese commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+)(?![eE][+-]?\d+)(?!，\d)'

    # Extract numbers with Chinese commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without Chinese commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_is_number(string):
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def count_letters(string):
    return sum(c.isalpha() and 'a' <= c <= 'z' or 'A' <= c <= 'Z' for c in string)


def normalize_str(string, answer):
    # check if characters in the string

    # if number, numerize it.
    if string is None:
        return [string]
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        if len(string) > len(answer) + 20 or count_letters(string) > count_letters(answer) + 2:
            return []
        return [string]


def get_fill_blank_prediction(response, answer):
    """get the prediction from the generated response,
    return a list of predicted strings or numbers"""

    def get_key_subresponses(response):
        response = response.strip("。").strip()
        sub_responses = re.split(r'。|\n', response)
        indicators_of_keys = ['是', '为', '所以', '等于', '方案', '选择',
                              '正确答案', '因此', '最后', '答案', '结果']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None
            # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i], answer))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_TF_prediction(response):
    """get the prediction from the generated response,
    return a list of predicted strings or numbers"""

    def get_key_subresponses(response):
        response = response.strip("。").strip()
        sub_responses = re.split(r'。|\n', response)
        indicators_of_keys = ['是', '为', '所以', '判断',
                              '陈述', '说法', '表达', '答案', '结果']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            shortest_key_response = None
            # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


class CMMMU(ImageBaseDataset):

    DATASET_URL = {
        'CMMMU_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/CMMMU_TEST.tsv',
    }

    DATASET_MD5 = {
        'CMMMU_TEST': '521afc0f3bf341e6654327792781644d',
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        pass

    def build_prompt(self, line):

        pass
