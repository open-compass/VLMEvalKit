import os
import json
import argparse
import contextlib
from tqdm import tqdm
from collections import defaultdict
from reward_score import compute_score
from reward_manager import verifier_manager
import signal
from contextlib import contextmanager
def load_data(input_path):
    all_data = []
    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error: {e}")
    return all_data

def write_jsonl(data_path, dataset, indent=0, mode='w'):
    with open(data_path, mode, encoding='UTF-8') as f:
        if not isinstance(dataset, list):
            dataset = [dataset]
        for data in dataset:
            line = json.dumps(data, ensure_ascii=False, indent=indent if indent != 0 else None)
            f.write(line + '\n')

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def judge(data, output_path):
    for item in tqdm(data, desc="Scoring"):
        # result = compute_score(item['model_output'], item['ground_truth'], item['problem']) # olympiadbench 
        try:
            with timeout(30):
                result = compute_score(item['test_result'], item['answer'], item['question'])
                item['rule_based_acc'] = result['rule_based_acc']
                item['acc'] = result['acc']
                item['extracted_gt'] = result['extracted_gt']
                item['extracted_pred'] = result['extracted_pred']
                write_jsonl(output_path, item, mode='a')
        except TimeoutError:
            print(f"Timeout processing item: {item.get('question', 'Unknown')}")
            continue
    return data
    
def compute_mean_at_n(data, key_acc, log_path=None):
    domain_stats = defaultdict(lambda: defaultdict(list))
    difficulty_stats = defaultdict(lambda: defaultdict(list))
    language_stats = defaultdict(lambda: defaultdict(list))
    overall_acc = []
    try:
        for item in data:
            domain = item['domain']
            difficulty = item['difficulty']
            if item['translate'] is False:
                language = item['language']
            else:
                language = 'zh' if item['language'] == 'en' else 'en'
            acc = 1.0 if item[key_acc] is True else 0.0
            domain_stats[domain][difficulty].append(acc)
            difficulty_stats[difficulty][domain].append(acc)
            language_stats[language][domain].append(acc)
            overall_acc.append(acc)
    except Exception as e:
        print(e)

    def _print_stats():
        print("=== Mean@N Statistics ===\n")
        for domain in sorted(domain_stats.keys()):
            print(f"    Domain: {domain}\n")
            for difficulty in sorted(domain_stats[domain].keys()):
                scores = domain_stats[domain][difficulty]
                mean_acc = sum(scores) / len(scores) if scores else 0
                print(f"        Difficulty: {difficulty:<10} | Count: {len(scores):<4} | Acc: {mean_acc:.4f}\n")
            domain_mean = sum(sum(scores) for scores in domain_stats[domain].values()) / sum(len(scores) for scores in domain_stats[domain].values())
            print(f"        Domain Mean: {domain_mean:.4f}\n")
        
        print("\n")

        for language in sorted(language_stats.keys()):
            print(f"    Language: {language}\n")
            for domain in sorted(language_stats[language].keys()):
                scores = language_stats[language][domain]
                mean_acc = sum(scores) / len(scores) if scores else 0
                print(f"        Domain: {domain:<10} | Count: {len(scores):<4} | Acc: {mean_acc:.4f}\n")
            language_mean = sum(sum(scores) for scores in language_stats[language].values()) / sum(len(scores) for scores in language_stats[language].values())
            print(f"        Language Mean: {language_mean:.4f}\n")
        
        print("\n")

        for difficulty in sorted(difficulty_stats.keys()):
            print(f"    Difficulty: {difficulty}\n")
            for domain in sorted(difficulty_stats[difficulty].keys()):
                scores = difficulty_stats[difficulty][domain]
                mean_acc = sum(scores) / len(scores) if scores else 0
                print(f"        Domain: {domain:<10} | Count: {len(scores):<4} | Acc: {mean_acc:.4f}\n")
            diff_mean = sum(sum(scores) for scores in difficulty_stats[difficulty].values()) / sum(len(scores) for scores in difficulty_stats[difficulty].values())
            print(f"        Difficulty Mean: {diff_mean:.4f}\n")
        
        total_mean = sum(overall_acc) / len(overall_acc) if overall_acc else 0
        print(f"\nOverall Acc: {total_mean:.4f} on {len(overall_acc)} samples")

    if log_path:
        with open(log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            _print_stats()
    _print_stats() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help='Path to the input JSONL file.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output JSONL file.')
    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_path).replace('.jsonl', ''))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(args.input_path))

    data = load_data(args.input_path)
    data = judge(data, output_path)
    compute_mean_at_n(data, 'acc', log_path=os.path.join(output_dir, 'acc.log'))
    compute_mean_at_n(data, 'rule_based_acc', log_path=os.path.join(output_dir, 'rule_based_acc.log'))