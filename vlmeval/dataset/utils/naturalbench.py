import re


def extract_answer(output_string, task_type="yes_no"):
    """
    Extracts the answer from the output string based on the task type.

    Parameters:
    output_string (str): The output string.
    task_type (str): The type of task. Must be either "yes_no" or "multiple_choice".

    Returns:
    int:
        1 if "yes" or "A"
        0 if "no" or "B"
        -1 if no relevant answer is found.
        Raises a ValueError if an unsupported task_type is provided.
    """

    def find_word_position(string, word):
        pattern = r'\b' + re.escape(word) + r'\b'
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return match.start()
        return -1

    if task_type not in ["yes_no", "multiple_choice"]:
        raise ValueError(f"Task type {task_type} not supported. Must be 'yes_no' or 'multiple_choice'.")

    if task_type == "yes_no":
        position_yes_and_a = find_word_position(output_string, "yes")
        position_no_and_b = find_word_position(output_string, "no")
    elif task_type == "multiple_choice":
        position_yes_and_a = find_word_position(output_string, "A")
        position_no_and_b = find_word_position(output_string, "B")

    if position_yes_and_a == -1 and position_no_and_b == -1:
        print(f"No answer found in the output string: {output_string}.")
        return -1
    elif position_yes_and_a != -1 and position_no_and_b != -1:
        return 1 if position_yes_and_a < position_no_and_b else 0
    else:
        return 0 if position_yes_and_a == -1 else 1


def get_scores(scores):
    """
    Calculate various scores based on the given results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and images:
        - "q0_i0" means question_0 on image_0
        - "q0_i1" means question_0 on image_1
        - "q1_i0" means question_1 on image_0
        - "q1_i1" means question_1 on image_1

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'question_score': Average question score
            - 'image_score': Average image score
            - 'binary_score': Average binary VQA score
            - 'group_score': Average group score
    """
    question_score = 0.0
    image_score = 0.0
    binary_score = 0.0
    group = 0.0

    num_samples = len(scores)

    def calculate_image_score(result):
        image_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q1_i0"] == 0.0:
                image_correct += 1
            if result["q1_i1"] == 1.0 and result["q0_i1"] == 0.0:
                image_correct += 1
        elif isinstance(result, list):
            if result[0] == 1.0 and result[2] == 0.0:
                image_correct += 1
            if result[3] == 1.0 and result[1] == 0.0:
                image_correct += 1
        return image_correct

    def calculate_question_score(result):
        text_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q0_i1"] == 0.0:
                text_correct += 1
            if result["q1_i1"] == 1.0 and result["q1_i0"] == 0.0:
                text_correct += 1
        else:
            if result[0] == 1.0 and result[1] == 0.0:
                text_correct += 1
            if result[3] == 1.0 and result[2] == 0.0:
                text_correct += 1
        return text_correct

    def calculate_binary_score(result):
        binary_score_correct = 0
        if isinstance(result, dict):
            binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
            binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0
        else:
            binary_score_correct += 1 if result[0] == 1.0 else 0
            binary_score_correct += 1 if result[1] == 0.0 else 0
            binary_score_correct += 1 if result[2] == 0.0 else 0
            binary_score_correct += 1 if result[3] == 1.0 else 0

        return binary_score_correct

    def calculate_group(result):
        group_correct = 0
        if calculate_question_score(result) == 2 and calculate_image_score(result) == 2:
            group_correct += 1

        return group_correct

    if isinstance(scores, dict):
        for _, result in scores.items():
            question_score += calculate_question_score(result)
            image_score += calculate_image_score(result)
            binary_score += calculate_binary_score(result)
            group += calculate_group(result)
    else:
        for result in scores:
            question_score += calculate_question_score(result)
            image_score += calculate_image_score(result)
            binary_score += calculate_binary_score(result)
            group += calculate_group(result)

    results = {
        'question_score': question_score / float(num_samples * 2),
        'image_score': image_score / float(num_samples * 2),
        'binary_score': binary_score / float(num_samples * 4),
        'group_score': group / num_samples
    }

    return results