#Copyright (2024) Bytedance Ltd. 
import json
import numpy as np
from vlmeval.smp import *


def evaluate_exact_match_accuracy(entry):
    if isinstance(entry['answer'], str):
        entry['answer'] = [entry['answer']]
    score = max([
        (1.0 if ann.strip().lower() in entry['prediction'].strip().lower().replace(".","") else 0.0)
        for ann in entry['answer']
    ])
    return score

def process_entry(entry):
    # Assuming entry is a dictionary with 'answer', 'prediction', and 'category' keys
    score = evaluate_exact_match_accuracy(entry)
    return entry['category'], score

def MTVQA_eval(eval_file, **kwargs):
    logger = get_logger('Evaluation')
    data = load(eval_file)
    assert 'answer' in data and 'prediction' in data and 'category' in data
    data['prediction'] = [str(x) for x in data['prediction']]
    data['answer'] = [str(x) for x in data['answer']]
    lt = len(data)
    
    # Convert the data into a list of dictionaries
    entries = [{'answer': data['answer'][i], 'prediction': data['prediction'][i], 'category': data['category'][i]} for i in range(lt)]
    
    # Create a multiprocessing pool
    with mp.Pool(processes=16) as pool:
        # Map the process_entry function to the list of entries
        results = pool.map(process_entry, entries)
    
    # Group the results by category and calculate the average score for each category
    category_scores = {}
    for category, score in results:
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)
    
    # Calculate the average score for each category
    category_averages = {category: np.mean(scores) for category, scores in category_scores.items()}
    
    # Convert the averages to a DataFrame if needed
    ret = d2df(category_averages)
    ret.round(2)
    
    suffix = eval_file.split('.')[-1]
    result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
    logger.info(f'MTVQA Eval Finished. Saved to {result_file}. ')
    logger.info(ret)
    dump(ret, result_file)

    return category_averages