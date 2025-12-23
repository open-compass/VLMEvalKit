import re
import numpy as np
from collections import defaultdict
from ...smp import *

# 参考 https://github.com/mistralai/mistral-evals/blob/main/eval/tasks/mm_mt_bench.py

BRACKET_SCORE_RE = re.compile(r"\[\[(\d+\.?\d*)\]\]")

SYSTEM_PROMPT = '''Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the most recent question given the previous conversation as context. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

'''


def get_score(judgement: str) -> float:
    """从 judge 的回答中提取评分"""
    match = re.search(BRACKET_SCORE_RE, judgement)
    if match:
        try:
            rating = float(match.groups()[0])
        except:
            rating = -1.0
    else:
        # 如果无法提取评分，返回 -1
        rating = -1.0
    return rating


def build_judge_prompt(questions: list, ref_answer: str, model_answer: str) -> str:
    """
    构建 judge prompt
    
    Args:
        questions: 用户问题列表
        ref_answer: 参考答案
        model_answer: 模型最后一个回答
    """
    prompt = "<|The Start of Conversation with User|>\n\n"
    
    for q in questions:
        prompt += f"### User:\n{q}\n"

    prompt += f"### Reference answer:\n{ref_answer}\n\n"
    
    prompt += f"\n\n### Assistant's answer:\n{model_answer}\n\n"
    prompt += "<|The End of Conversation with User|>\n\n"
    
    return prompt


def mm_mt_bench_score(model, line, turn_idx):
    """
    对单条 MM-MT-Bench 数据进行评分
    
    多轮对话的每一轮单独评分：
    - questions 和 gt_answers 包含到当前轮为止的所有历史（用参考答案作为历史）
    - line['pred'] 是模型对当前轮问题的回答
    - turn_idx 指定当前是第几轮
    
    例如对于 3 轮对话，会调用 3 次：
    - turn_idx=0: questions=[q1], gt_answers=[a1_ref], pred=model_answer_for_q1
    - turn_idx=1: questions=[q1,q2], gt_answers=[a1_ref,a2_ref], pred=model_answer_for_q2
    - turn_idx=2: questions=[q1,q2,q3], gt_answers=[a1_ref,a2_ref,a3_ref], pred=model_answer_for_q3
    
    Args:
        model: judge 模型
        line: 数据行，包含:
            - question: 所有用户问题列表
            - answer: 所有参考答案列表
            - pred: 模型对当前轮问题的回答（字符串）
            - category: 类别
        turn_idx: 当前轮次索引 (0, 1, 2, ...)
        
    Returns:
        dict: 包含 score, category, turn, judgement, log
    """
    # 解析数据
    questions = eval(line['question']) if isinstance(line['question'], str) else line['question']
    gt_answers = eval(line['answer']) if isinstance(line['answer'], str) else line['answer']
    

    ref_answer = gt_answers[turn_idx]
    prediction = line['pred']  # 当前轮的模型回答
    
    category = line.get('category', 'unknown')
    turn = turn_idx
    
    try:
        # 构建 judge prompt
        # questions 和 gt_answers 包含到当前轮为止的所有对话
        judge_prompt = build_judge_prompt(
            questions, 
            ref_answer,
            prediction
        )
        
        # 构建完整消息
        full_prompt = [
            {"role": "system", "type": "text", "value": SYSTEM_PROMPT},
            {"role": "user", "type": "text", "value": judge_prompt}
        ]
        
        # 调用 judge 模型
        response = model.generate(full_prompt)
        
        # 提取评分
        score = get_score(response)
        
        if score >= 0:
            log = f'Category: {category}, Turn: {turn}, Score: {score}'
        else:
            log = f'Category: {category}, Turn: {turn}, Failed to extract score'
            
    except Exception as e:
        logging.warning(f"Error scoring: {str(e)}")
        score = -1
        response = str(e)
        log = f'Category: {category}, Turn: {turn}, Error: {str(e)}'
    
    return dict(
        score=score,
        category=category,
        turn=turn,
        judgement=response,
        log=log
    )


def calculate_mm_mt_bench_metrics(results: dict) -> dict:
    """
    聚合 MM-MT-Bench 的评分结果
    
    按照 mistral-evals 的 aggregate_metrics 逻辑:
    - micro_average_score: 所有样本的平均分
    - macro_average_score: 各 category 平均分的宏平均
    - {category}_average: 各类别的平均分
    - turn_{n}_average: 各轮次的平均分
    
    Args:
        results: {index: result_dict} 格式的结果字典
                 每个 result_dict 包含 score, category, turn
        
    Returns:
        dict: 包含各种聚合指标
    """
    all_scores = []
    category_scores = defaultdict(list)
    turn_scores = defaultdict(list)
    
    for idx, result in results.items():
        score = result.get('score', -1)
        if score < 0:
            continue
            
        category = result.get('category', 'unknown')
        turn = result.get('turn', 0)
        
        all_scores.append(score)
        category_scores[category].append(score)
        turn_scores[turn].append(score)
    
    # 计算 micro average (所有样本的平均)
    micro_average = float(np.mean(all_scores)) if all_scores else 0.0
    
    # 计算各 category 的平均
    category_averages = {}
    for cat, scores in category_scores.items():
        category_averages[f'{cat}_average'] = float(np.mean(scores))
    
    # 计算各 turn 的平均
    turn_averages = {}
    for turn, scores in turn_scores.items():
        turn_averages[f'turn_{turn}_average'] = float(np.mean(scores))
    
    # 计算 macro average (各 category 平均分的宏平均，不包括 turn)
    # 注意：只对 category 计算 macro average，与 mistral-evals 一致
    macro_average = float(np.mean(list(category_averages.values()))) if category_averages else 0.0
    
    return {
        'micro_average_score': micro_average,
        'macro_average_score': macro_average,
        'valid_count': len(all_scores),
        **category_averages,
        **turn_averages,
    }

