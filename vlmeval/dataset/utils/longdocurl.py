import ast
import re
from math import isclose


EXTRACTOR_PROMPT = """Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis. 
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer". 
- Answers may not meet visibility requirements within questions, such as, one question requires to find answers 'between 7-th to 20-th images' but invisible information in 'image 23' is included in analysis. So, these invisible components should be deprecated and removed from final answers.
- Please use <concise_answer> and </concise_answer> tokens at the start and end of the extracted answer. For example, if the extracted answer is number 3, the format is <concise_answer>3</concise_answer>.
- Please use <answer_format> and </answer_format> tokens at the start and end of the answer format. For example, if the answer format is List, the format is <answer_format>List</answer_format>.
- Please make your response as concise as possible. Also note that your response should be formatted as below: 
```
Extracted answer: <concise_answer>[answer]</concise_answer>
Answer format: <answer_format>[answer format]</answer_format>
```

Please read the following example, then extract the answer from the model response and type it at the end of the prompt. 

---
Question: List the primary questions asked about the services in this report.
Analysis: The primary questions asked about the services in the report for The Limes Residential Home are: \n\n1. Is the service safe? \n\n2. Is the service effective? \n\n3. Is the service caring? \n\n4. Is the service responsive? \n\n5. Is the service well-led? 
Extracted answer: <concise_answer>['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']</concise_answer>
Answer format: <answer_format>List</answer_format>

---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:

1. Regulation 13: Safeguarding service users from abuse and improper treatment

2. Regulation 12: Safe care and treatment

3. Regulation 18: Staffing

4. Regulation 11: Need for consent

5. Regulation 10: Dignity and respect

6. Regulation 9: Person-centred care

7. Regulation 17: Good governance

8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents

9. Regulation 18: Failure to maintain an accurate and up-to-date care plan

10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively

These breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.
Extracted answer: <concise_answer>10</concise_answer>
Answer format: <answer_format>Integer</answer_format>

---
Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump's election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question.
Extracted answer: <concise_answer>Not answerable</concise_answer>
Answer format: <answer_format>None</answer_format>

---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately.
Extracted answer: <concise_answer>Fail to answer</concise_answer>
Answer format: <answer_format>None</answer_format>

---"""


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage=False, is_close=False):
    def get_precision(gt_ans):
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip('%').strip())
    try:
        prediction = float(str(prediction).strip().rstrip('%').strip())
    except Exception:
        return False

    gt_result = [reference / 100, reference, reference * 100] if include_percentage else [reference]
    for item in gt_result:
        try:
            if is_close and isclose(item, prediction, rel_tol=0.01):
                return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    s = s.replace(',', '')
    for suffix in [
        'kg', 'mm', 'meters', 'minutes', 'miles', 'mile',
        'acres', 'million', 'thousand', 'billion', 'm',
    ]:
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
            break
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    s = re.sub(r"^['\"]|['\"]$", '', s).strip()
    s = s.strip().lstrip('$').strip()
    s = s.strip().lstrip('£').strip()
    s = s.strip().rstrip('%').strip()
    return s


def is_exact_match(s):
    if 'https://' in s:
        return True
    if s.endswith('.py') or s.endswith('ipynb'):
        return True
    if s.startswith('page'):
        return True
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        return True
    if 'a.m.' in s or 'p.m.' in s:
        return True
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        return True
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        return True
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        return True
    return False


def isfloat(num):
    try:
        float(num)
        return True
    except Exception:
        return False


def _maybe_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def parse_extracted_answer(response):
    try:
        concise_answer = re.findall(r'<concise_answer>(.*?)</concise_answer>', response, re.DOTALL)[0]
        answer_format = re.findall(r'<answer_format>(.*?)</answer_format>', response, re.DOTALL)[0]
    except Exception:
        return 'Fail to extract', 'None'

    try:
        parsed_answer = ast.literal_eval(concise_answer)
        if isinstance(parsed_answer, set):
            parsed_answer = list(parsed_answer)
    except Exception:
        parsed_answer = concise_answer
    return parsed_answer, answer_format


def build_extraction_prompt(question, analysis, system_prompt):
    return system_prompt + EXTRACTOR_PROMPT + '\nQuestion: ' + str(question) + '\nAnalysis: ' + str(analysis)


def eval_score(gt, pred, answer_type):
    if answer_type == 'Integer':
        try:
            gt = get_clean_string(str(gt))
            if len(re.findall(r'\d+,\s*\d+', gt, re.DOTALL)) > 0:
                gt = ''.join([_.strip() for _ in gt.split(',')])
            gt = int(gt)
        except Exception:
            pass
        try:
            pred = get_clean_string(str(pred))
            if len(re.findall(r'\d+,\s*\d+', pred, re.DOTALL)) > 0:
                pred = ''.join([_.strip() for _ in pred.split(',')])
            pred = int(pred)
        except Exception:
            pred = ''
        score = (gt == pred)
    elif answer_type == 'Float':
        gt = get_clean_string(str(gt))
        pred = get_clean_string(str(pred))

        if len(re.findall(r'\d+,\s*\d+', gt, re.DOTALL)) > 0:
            gt = ''.join([_.strip() for _ in gt.split(',')])
        try:
            gt = float(gt)
        except Exception:
            pass

        if len(re.findall(r'\d+,\s*\d+', pred, re.DOTALL)) > 0:
            pred = ''.join([_.strip() for _ in pred.split(',')])
        try:
            pred = float(pred)
        except Exception:
            pred = str(pred)

        try:
            score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
        except Exception:
            score = 0
    elif answer_type in ['String', 'None']:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt == pred)
        else:
            score = anls_compute(gt, pred)
    else:
        gt = _maybe_list(gt)
        pred = _maybe_list(pred)
        if not isinstance(gt, list):
            gt = [gt]
        if not isinstance(pred, list):
            pred = [pred]
        if not pred:
            return 0.0

        if isinstance(gt[0], dict):
            gt = ['-'.join([str(value) for _, value in item.items()]) for item in gt]
        if isinstance(pred[0], dict):
            pred = ['-'.join([str(value) for _, value in item.items()]) for item in pred]

        gt = [get_clean_string(a) for a in gt]
        pred = [get_clean_string(a) for a in pred]
        if isfloat(gt[0]) or is_exact_match(gt[0]):
            score = ('-'.join(gt) == '-'.join(pred))
        else:
            greedy_scores = [max([anls_compute(str(gt_v), str(pred_v)) for pred_v in pred]) for gt_v in gt]
            score = sum(greedy_scores) / len(gt) * min(1, len(gt) / len(pred)) ** 0.5

    return float(score)
