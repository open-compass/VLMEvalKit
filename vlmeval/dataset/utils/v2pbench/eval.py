from .prompt import Recall_Evaluation_Prompt, Precision_Evaluation_Prompt, Answer_Extraction_Prompt_part1, Answer_Extraction_Prompt_part2, Answer_Scoring_Prompt_part1, Answer_Scoring_Prompt_part2
from openai import OpenAI
import base64
import json
import copy
from tqdm import tqdm
import os
import pandas as pd
import concurrent.futures
import re
import ast
from pathlib import Path
import argparse

def get_output_wo_image(prompt: str, max_tokens: int=2048):
    """OpenAI API Function Call"""
    while True:
        try:
            client = OpenAI(api_key='<KEY>')
            client.api_key = os.environ.get('OPENAI_API_KEY', '<YOUR_API_KEY>')
            client.base_url = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed, retrying... Error message:{e}")

def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def save_json(data, file_path, indent=4):
    with open(file_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding="utf-8") as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")

def calculate_time_iou(interval1, interval2):
    
    start1, end1 = interval1
    start2, end2 = interval2
    
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start) 
    
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    
    if union == 0:
        return 0  
    iou = intersection / union
    return iou

def is_valid_time_interval(interval_str):
    
    try:
        interval = ast.literal_eval(interval_str)
        if isinstance(interval, list) and len(interval) == 2:
            if all(isinstance(x, (int, float)) for x in interval):
                return True
        return False
    except (ValueError, SyntaxError):
        return False

def is_valid_space_interval(s):
    if not (s.startswith('[') and s.endswith(']')):
        return False
    content = s[1:-1]
    parts = content.split(',')
    if len(parts) != 4:
        return False
    for part in parts:
        try:
            int(part.strip()) 
        except ValueError:
            return False  
    return True


def string_to_list(s):
    content = s[1:-1]
    return [int(part.strip()) for part in content.split(',')]


def extract_json_between_backticks(s):
    # pattern = r'```json\n(.*?)```'  
    # match = re.search(pattern, s, re.DOTALL)   
    # if not match:
    #     raise ValueError("No JSON content wrapped by ``` was found.")
    # json_str = match.group(1).strip() 
    json_str = s
    
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted content is not valid JSON: {e}")


def calculate_recall(json_object):

    stats = {
        "Video Description Steps": {"Matched": 0, "Unmatched": 0},
        "Logical Inference Steps": {"Matched": 0, "Unmatched": 0},
        "Background Review Steps": {"Matched": 0, "Unmatched": 0}
    }
    for item in json_object:
        step_type = item["step_type"]
        judgement = item["judgment"]
        stats[step_type][judgement] += 1
    
    return stats


def calculate_space_iou(box1, box2):
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0 

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)


    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area
    return iou


def calculate_precision(json_object):
    stats = {
        "Video Description Steps": {"Matched": 0, "Wrong": 0, "Redundant": 0},
        "Logical Inference Steps": {"Matched": 0, "Wrong": 0, "Redundant": 0},
        "Background Review Steps": {"Matched": 0, "Wrong": 0, "Redundant": 0}
    }
    for item in json_object:
        step_type = item["step_type"]
        judgement = item["judgment"]
        stats[step_type][judgement] += 1
    
    return stats

def extract_answer(item):
    processed_item = copy.deepcopy(item)
    prompt_new = Answer_Extraction_Prompt_part1 + Answer_Extraction_Prompt_part2.format(question=item['question'], response=item['response'])
    processed_item['extract_answer'] = get_output_wo_image(prompt=prompt_new)
    return processed_item

def match_answer(item):
    
    processed_item = copy.deepcopy(item)
    try:
        if is_valid_time_interval(item.get('answer')) and is_valid_time_interval(item.get('extract_answer')):
            interval1 = ast.literal_eval(item['answer'])
            interval2 = ast.literal_eval(item['extract_answer'])
            
            iou = calculate_time_iou(interval1, interval2)
            if iou > 0.7:
                processed_item['answer_scoring'] = '1'
            else:
                processed_item['answer_scoring'] = '0'
        elif is_valid_space_interval(item.get('answer')) and is_valid_space_interval(item.get('extract_answer')):
            bbox1 = string_to_list(item['answer'])
            bbox2 = string_to_list(item['extract_answer'])
            iou = calculate_space_iou(bbox1, bbox2)
            if iou > 0.5:
                processed_item['answer_scoring'] = '1'
            else:
                processed_item['answer_scoring'] = '0'
        else:
        
            answer = item['answer']
            prompt_new = Answer_Scoring_Prompt_part1 + Answer_Scoring_Prompt_part2.format(
                question=item['question'],
                extract_answer=processed_item['extract_answer'],
                gt_answer=answer
            )
            processed_item['answer_scoring'] = get_output_wo_image(prompt=prompt_new)
    except Exception as e:
        print(f"An error occurred while processing the project: {e}")
        processed_item['answer_scoring'] = 0 
    return processed_item

def recall(item):
    processed_item = copy.deepcopy(item)

    json_object = json.loads(extract_json_between_backticks(processed_item['recall_eval']))
    stats = calculate_recall(json_object)
            
    Video_recall = "" if (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Unmatched']) == 0 else stats['Video Description Steps']['Matched'] / (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Unmatched'])
    
    logic_recall = "" if (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Unmatched']) == 0 else stats['Logical Inference Steps']['Matched'] / (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Unmatched'])
    
    background_recall = "" if (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Unmatched']) == 0 else stats['Background Review Steps']['Matched'] / (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Unmatched'])

    
    processed_item['Video_recall'] = Video_recall
    processed_item['logic_recall'] = logic_recall
    processed_item['background_recall'] = background_recall

    total_matched = (
        stats['Video Description Steps']['Matched'] +
        stats['Logical Inference Steps']['Matched'] +
        stats['Background Review Steps']['Matched']
    )

    total_steps = (
        (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Unmatched']) +
        (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Unmatched']) +
        (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Unmatched'])
    )

    if total_steps == 0:
        overall_recall = ""  
    else:
        overall_recall = total_matched / total_steps

    processed_item['overall_recall'] = overall_recall

    return processed_item

def precision(item):
    processed_item = copy.deepcopy(item)

    json_object = json.loads(extract_json_between_backticks(processed_item['precision_eval']))
    stats = calculate_precision(json_object)
            
    Video_precision = "" if (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Wrong']) == 0 else stats['Video Description Steps']['Matched'] / (stats['Video Description Steps']['Matched'] + stats['Video Description Steps']['Wrong'])
    
    logic_precision = "" if (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Wrong']) == 0 else stats['Logical Inference Steps']['Matched'] / (stats['Logical Inference Steps']['Matched'] + stats['Logical Inference Steps']['Wrong'])
    
    background_precision = "" if (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Wrong']) == 0 else stats['Background Review Steps']['Matched'] / (stats['Background Review Steps']['Matched'] + stats['Background Review Steps']['Wrong'])

    
    processed_item['Video_precision'] = Video_precision
    processed_item['logic_precision'] = logic_precision
    processed_item['background_precision'] = background_precision

    total_matched = (
        stats['Video Description Steps']['Matched'] +
        stats['Logical Inference Steps']['Matched'] +
        stats['Background Review Steps']['Matched']
    )


    total_wrong = (
        stats['Video Description Steps']['Wrong'] +
        stats['Logical Inference Steps']['Wrong'] +
        stats['Background Review Steps']['Wrong']
    )


    if (total_matched + total_wrong) == 0:
        overall_precision = ""  
    else:
        overall_precision = total_matched / (total_matched + total_wrong)

    processed_item['overall_precision'] = overall_precision

    
    total_step_num = 0
    for counts in stats.values():
        total_step_num += sum(counts.values())
        
    redundant_num = 0 
    for counts in stats.values():
        redundant_num += counts['Redundant']
        
    efficiency = (total_step_num - redundant_num) / total_step_num
    processed_item['efficiency'] = efficiency
    return processed_item


def process_item(item):
    try:
    
        item = extract_answer(item)
        item = match_answer(item)
        item = recall(item)
        item = precision(item)
        return item
    except Exception as e:
        print(f"An error occurred while processing the project: {e}")
        return None  

def main():
    parser = argparse.ArgumentParser(description='Process evaluation data with multi-threading')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file path')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file path')
    parser.add_argument('-w', '--workers', type=int, default=50, help='Number of worker threads')
    
    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Processing {input_path} with {args.workers} workers...")
    
    data = read_json(input_path)
    output = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_item, item) for item in data]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(futures),
                         desc="Processing"):
            if (result := future.result()) is not None:
                output.append(result)
    
    save_json(output, output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
