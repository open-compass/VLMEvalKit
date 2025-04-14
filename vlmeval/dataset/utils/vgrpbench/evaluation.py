import ast
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from .score import evaluate_single_puzzle

<<<<<<< HEAD

def get_eval(judge, content):
    """
    Generate evaluation using the judge model.

    Args:
        judge: The evaluation model
        content: Input content for the evaluation

=======
def get_eval(judge, content):
    """
    Generate evaluation using the judge model.
    
    Args:
        judge: The evaluation model
        content: Input content for the evaluation
        
>>>>>>> f7bcc2c8 (add vgrpbench)
    Returns:
        The generated evaluation output
    """
    return judge.generate(content)

<<<<<<< HEAD

def VGRPBench_atomeval(model, prompt, line):
    """
    Perform atomic evaluation for a VGRPBench puzzle.

=======
def VGRPBench_atomeval(model, prompt, line):
    """
    Perform atomic evaluation for a VGRPBench puzzle.
    
>>>>>>> f7bcc2c8 (add vgrpbench)
    Args:
        model: The evaluation model
        prompt: Input prompt for evaluation
        line: Dictionary containing puzzle information
<<<<<<< HEAD

=======
        
>>>>>>> f7bcc2c8 (add vgrpbench)
    Returns:
        dict: Evaluation scores
    """
    print("raw output", prompt)
    output = get_eval(model, prompt)
    print("formatted output", output)
    scores = parse_score(line, output)
    return scores

<<<<<<< HEAD

def parse_score(line, output):
    """
    Parse the score from the model's output for a VGRPBench puzzle.

    Args:
        line: Dictionary-like object containing puzzle information
        output: The model's output string

    Returns:
        dict: Dictionary with perception_correct and answer_correct results
    """

    # Extract category to determine puzzle type
    category = line['category']
    puzzle_type = category.split('_')[0]  # e.g., "thermometers" from "thermometers_4x4"

    # Parse the puzzle state from the states field
    puzzle_data = line['states']
    puzzle_data = ast.literal_eval(puzzle_data)

=======
def parse_score(line, output):
    """
    Parse the score from the model's output for a VGRPBench puzzle.
    
    Args:
        line: Dictionary-like object containing puzzle information
        output: The model's output string
        
    Returns:
        dict: Dictionary with perception_correct and answer_correct results
    """
    
    # Extract category to determine puzzle type
    category = line['category']
    puzzle_type = category.split('_')[0]  # e.g., "thermometers" from "thermometers_4x4"
    
    # Parse the puzzle state from the states field
    puzzle_data = line['states']
    puzzle_data = ast.literal_eval(puzzle_data)
    
>>>>>>> f7bcc2c8 (add vgrpbench)
    # Evaluate the puzzle solution
    evaluation_result = evaluate_single_puzzle(output, puzzle_data, puzzle_type)

    return evaluation_result

<<<<<<< HEAD

def VGRPBench_score(data):
    """
    Calculate scores for VGRPBench puzzles by category.

    Args:
        data: DataFrame containing evaluation results

=======
def VGRPBench_score(data):
    """
    Calculate scores for VGRPBench puzzles by category.
    
    Args:
        data: DataFrame containing evaluation results
        
>>>>>>> f7bcc2c8 (add vgrpbench)
    Returns:
        pandas.DataFrame: Aggregated scores by category
    """
    # Get unique categories without 'overall'
    cates = list(set(data['category']))
    ret = defaultdict(list)

    for c in cates:
        ret['category'].append(c)
        # Filter data for the current category
        sub = data[data['category'] == c]
<<<<<<< HEAD

        # Calculate perception score (as percentage with 2 decimal places)
        perception_score = round(np.mean(sub['perception_correct']) * 100, 2)
        ret['Perception Score'].append(perception_score)

        # Calculate answer score (as percentage with 2 decimal places)
        answer_score = round(np.mean(sub['answer_correct']) * 100, 2)
        ret['Answer Score'].append(answer_score)

    return pd.DataFrame(ret)


def build_prompt(line):
    """
    Build a prompt from the prediction field in the data line.

    Args:
        line: Dictionary containing a 'prediction' field

=======
        
        # Calculate perception score (as percentage with 2 decimal places)
        perception_score = round(np.mean(sub['perception_correct']) * 100, 2)
        ret['Perception Score'].append(perception_score)
        
        # Calculate answer score (as percentage with 2 decimal places)
        answer_score = round(np.mean(sub['answer_correct']) * 100, 2)
        ret['Answer Score'].append(answer_score)
        
    return pd.DataFrame(ret)

def build_prompt(line):
    """
    Build a prompt from the prediction field in the data line.
    
    Args:
        line: Dictionary containing a 'prediction' field
        
>>>>>>> f7bcc2c8 (add vgrpbench)
    Returns:
        str: The prediction text to be used as a prompt
    """
    # Get the prediction entry from the prediction column
    return line['prediction']

<<<<<<< HEAD

def VGRPBench_get_system_prompt(line):
    """
    Get the system prompt for a specific puzzle type in VGRPBench.

    Args:
        line: A data row containing a 'category' field that defines the puzzle type

=======
def VGRPBench_get_system_prompt(line):
    """
    Get the system prompt for a specific puzzle type in VGRPBench.
    
    Args:
        line: A data row containing a 'category' field that defines the puzzle type
        
>>>>>>> f7bcc2c8 (add vgrpbench)
    Returns:
        str: A formatted system prompt loaded from the corresponding filter_prompt.json file
    """
    # Extract puzzle type from category (e.g., "thermometers" from "thermometers_4x4")
    puzzle_type = line['category'].split('_')[0]
<<<<<<< HEAD

    # Construct path to the filter_prompt.json file for this puzzle type
    prompt_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        "formating-prompt",
        puzzle_type,
        "filter_prompt.json"
    )

    # Load and return the prompt from the JSON file
    with open(prompt_file, 'r') as f:
        prompt = json.load(f)

    prompt = str(prompt) + "According to the conversation history with the user feedback do the formatting for the current answer."  # noqa: E501

    return prompt
=======
    
    # Construct path to the filter_prompt.json file for this puzzle type
    prompt_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "configs", 
        "formating-prompt", 
        puzzle_type, 
        "filter_prompt.json"
    )
    
    # Load and return the prompt from the JSON file
    with open(prompt_file, 'r') as f:
        prompt = json.load(f)
    
    prompt = str(prompt) + "According to the conversation history with the user feedback do the formatting for the current answer."

    return prompt
>>>>>>> f7bcc2c8 (add vgrpbench)
