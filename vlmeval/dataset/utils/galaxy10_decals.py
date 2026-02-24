import re
from tqdm import tqdm
from ...smp import *
from collections import defaultdict


def check_format(llm_output):
    # check <think> ... </think> result
    if '<think>' not in llm_output or '</think>' not in llm_output:
        return 0.0
    if llm_output.count('<think>') != llm_output.count('</think>') and llm_output.count('<think>') != 1:
        return 0.0
    final_answer = llm_output.split('</think>')[1]
    if final_answer!=None and final_answer!='':
        return 1.0
    else:
        return 0.0

def remove_boxed(s): 
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

def Galaxy10DECaLS_auxeval(line):
    label_dict = {
        "Disturbed Galaxies": 0,
        "Merging Galaxies": 1,
        "Round Smooth Galaxies": 2,
        "In-between Round Smooth Galaxies": 3,
        "Cigar Shaped Smooth Galaxies": 4,
        "Barred Spiral Galaxies": 5,
        "Unbarred Tight Spiral Galaxies": 6,
        "Unbarred Loose Spiral Galaxies": 7,
        "Edge-on Galaxies without Bulge": 8,
        "Edge-on Galaxies with Bulge": 9
    }
    log = ''
    response = line['prediction']
    if not response or not isinstance(response, str):
        log += 'Invalid response format, returning False.'
        return dict(log=log, res=False)
    
    if '</think>' in response:
        final_ans = response.replace('\x08','\\b').split('</think>')[1]
    else:
        final_ans = response.replace('\x08','\\b')
        
    ground_truth = line['answer']
    # print(final_ans, ground_truth)
    if final_ans is None or not isinstance(final_ans, str) or final_ans.strip() == "":
        log += 'Invalid final answer, returning False.'
        return dict(log=log, res=False)
    boxed_answer = last_boxed_only_string(final_ans)
    if boxed_answer is None or boxed_answer == '':
        log += 'Invalid extract answer, returning False.'
        return dict(log=log, res=False)
    else:
        boxed_answer = remove_boxed(boxed_answer)
        if boxed_answer in label_dict and label_dict[boxed_answer] == ground_truth:
            log += 'Prefetch succeed. Rule evaluate Succeessfully.'
            return dict(log=log, res=True)
        else:
            log += 'Boxed answer does not match ground truth.'
            return dict(log=log, res=False)
                  
def Galaxy10DECaLS_acc(result_file):
    data = load(result_file)
    if result_file.endswith('.json'):
        data = pd.DataFrame(data)
    tot = defaultdict(int)
    hit = defaultdict(int)
    fetch = defaultdict(int)
    lt = len(data)

    for i in tqdm(range(lt)):
        item = data.iloc[i]
        tot['Overall'] += 1
        if 'log' in item:
            log_value = item['log']
        else:
            log_value = item['answer_match_log']
        if 'Prefetch succeed' in log_value:
            fetch['Overall'] += 1
        if item.get('result'):
            hit['Overall'] += 1
            
    res = defaultdict(list)
    for k in tot:
        res['total'].append(tot[k])
        # res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100 if tot[k] else 0.0)
        # res['rule_prefetch'].append(fetch[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)

    return pd.DataFrame(res)