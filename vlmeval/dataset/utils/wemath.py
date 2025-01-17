import pandas as pd
import json
import numpy as np
import os
import argparse

### four_dimensional_metrics.py

# Function to evaluate steps
def evaluate_evaluate_steps(json, steps):
    jokers = [json[[f'joker_{i}', f'knowledge concept_{i}']] for i in range(1, steps + 1)]
    for i in range(steps):
        jokers[i].rename(columns={f'joker_{i + 1}': 'joker', f'knowledge concept_{i + 1}': 'knowledge_concept'}, inplace=True)
    concatenated_steps = pd.concat(jokers, axis=0)
    return concatenated_steps

# Function to load and process JSON data
def load_and_process_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if 'hit' not in df.columns:
        df['processed_answer'] = df['response'].str.split('Answer').str[-1].str.strip().str.replace(r'[>><<:.]', '', regex=True).str.strip()
        df['processed_answer'] = df['processed_answer'].apply(lambda x: x[0] if x and x[0] in 'ABCDEFGH' else None)
        df['joker'] = df['processed_answer'] == df['answer']
    else:
        df['joker'] = df['hit'].astype(bool)
    return df

# Function to process steps data and merge results
def evaluate_process_steps_data(df, steps):
    steps_data = {f'{steps}steps_{i}': df[df['key'] == f'{steps}steps_{i}'] for i in range(1, steps + 1)}
    steps_data[f'{steps}steps_multi'] = df[df['key'] == f'{steps}steps_multi']
    for key, data in steps_data.items():
        data.columns = [col + f'_{key.split("_")[-1]}' for col in data.columns]
    merged_data = steps_data[f'{steps}steps_1']
    for i in range(2, steps + 1):
        merged_data = pd.merge(merged_data, steps_data[f'{steps}steps_{i}'], left_on=f'ID_1', right_on=f'ID_{i}', how='left')
    merged_data = pd.merge(merged_data, steps_data[f'{steps}steps_multi'], left_on=f'ID_1', right_on='ID_multi', how='left')
    return merged_data

# Function to calculate evaluation metrics
def evaluate_calculate_metrics(merged_2steps, merged_3steps):
    metrics = {}
    metrics['steps2_filtered_rows_1_loose'] = merged_2steps[((merged_2steps['joker_1'] == False) & (merged_2steps['joker_2'] == False)) & (merged_2steps['joker_multi'] == True)]
    metrics['steps2_filtered_rows_1_strict'] = merged_2steps[((merged_2steps['joker_1'] == False) | (merged_2steps['joker_2'] == False)) & (merged_2steps['joker_multi'] == True)]
    metrics['steps2_filtered_rows_2'] = merged_2steps[((merged_2steps['joker_1'] == True) & (merged_2steps['joker_2'] == True)) & (merged_2steps['joker_multi'] == False)]
    metrics['steps2_filtered_rows_3'] = merged_2steps[((merged_2steps['joker_1'] == False) | (merged_2steps['joker_2'] == False)) & (merged_2steps['joker_multi'] == False)]
    metrics['steps2_filtered_rows_4_loose'] = merged_2steps[((merged_2steps['joker_1'] == True) | (merged_2steps['joker_2'] == True)) & (merged_2steps['joker_multi'] == True)]
    metrics['steps2_filtered_rows_4_strict'] = merged_2steps[((merged_2steps['joker_1'] == True) & (merged_2steps['joker_2'] == True)) & (merged_2steps['joker_multi'] == True)]
    metrics['steps3_filtered_rows_1_loose'] = merged_3steps[((merged_3steps['joker_1'] == False) & (merged_3steps['joker_2'] == False) & (merged_3steps['joker_3'] == False)) & (merged_3steps['joker_multi'] == True)]
    metrics['steps3_filtered_rows_1_strict'] = merged_3steps[((merged_3steps['joker_1'] == False) | (merged_3steps['joker_2'] == False) | (merged_3steps['joker_3'] == False)) & (merged_3steps['joker_multi'] == True)]
    metrics['steps3_filtered_rows_2'] = merged_3steps[((merged_3steps['joker_1'] == True) & (merged_3steps['joker_2'] == True) & (merged_3steps['joker_3'] == True)) & (merged_3steps['joker_multi'] == False)]
    metrics['steps3_filtered_rows_3'] = merged_3steps[((merged_3steps['joker_1'] == False) | (merged_3steps['joker_2'] == False) | (merged_3steps['joker_3'] == False)) & (merged_3steps['joker_multi'] == False)]
    metrics['steps3_filtered_rows_4_loose'] = merged_3steps[((merged_3steps['joker_1'] == True) | (merged_3steps['joker_2'] == True) | (merged_3steps['joker_3'] == True)) & (merged_3steps['joker_multi'] == True)]
    metrics['steps3_filtered_rows_4_strict'] = merged_3steps[((merged_3steps['joker_1'] == True) & (merged_3steps['joker_2'] == True) & (merged_3steps['joker_3'] == True)) & (merged_3steps['joker_multi'] == True)]
    # metrics.to_csv("/Users/mac/Desktop/测试结果/error_anal/csv/gpt4o-0626.csv", index = False)
    return metrics

# Function to compute evaluation rates and final scores
def evaluate_compute_final_scores(metrics, total_count):
    total_counts = {
        'InadequateGeneralization': len(metrics['steps2_filtered_rows_2']) + len(metrics['steps3_filtered_rows_2']),
        'InsufficientKnowledge': len(metrics['steps2_filtered_rows_3']) + len(metrics['steps3_filtered_rows_3']),
        'CompleteMastery_loose': len(metrics['steps2_filtered_rows_4_loose']) + len(metrics['steps3_filtered_rows_4_loose']),
        'CompleteMastery_strict': len(metrics['steps2_filtered_rows_4_strict']) + len(metrics['steps3_filtered_rows_4_strict']),
        'RoteMemorization_loose': len(metrics['steps2_filtered_rows_1_loose']) + len(metrics['steps3_filtered_rows_1_loose']),
        'RoteMemorization_strict': len(metrics['steps2_filtered_rows_1_strict']) + len(metrics['steps3_filtered_rows_1_strict'])
    }
    rates = {
        'InadequateGeneralization_rate': "{:.2%}".format(total_counts['InadequateGeneralization'] / total_count),
        'InsufficientKnowledge_rate': "{:.2%}".format(total_counts['InsufficientKnowledge'] / total_count),
        'CompleteMastery_loose_rate': "{:.2%}".format(total_counts['CompleteMastery_loose'] / total_count),
        'CompleteMastery_strict_rate': "{:.2%}".format(total_counts['CompleteMastery_strict'] / total_count),
        'RoteMemorization_loose_rate': "{:.2%}".format(total_counts['RoteMemorization_loose'] / (total_counts['CompleteMastery_loose'] + total_counts['RoteMemorization_loose'])),
        'RoteMemorization_strict_rate': "{:.2%}".format(total_counts['RoteMemorization_strict'] / (total_counts['CompleteMastery_strict'] + total_counts['RoteMemorization_strict']))
    }
    return total_counts, rates

# Function to update main results DataFrame
def evaluate_update_main_results_df(main_results_df, model, total_counts, rates):

    final_score_loose = "{:.2%}".format((525 - 0.5 * total_counts['InadequateGeneralization'] - total_counts['RoteMemorization_loose'] - total_counts['InsufficientKnowledge']) / 525)
    final_score_strict = "{:.2%}".format((525 - 0.5 * total_counts['InadequateGeneralization'] - total_counts['RoteMemorization_strict'] - total_counts['InsufficientKnowledge']) / 525)

    new_row = {
        'Model': model,
        'Score (Strict)': final_score_strict,
        'InsufficientKnowledge (Strict)': f"{rates['InsufficientKnowledge_rate']} ({total_counts['InsufficientKnowledge']})",
        'InadequateGeneralization (Strict)': f"{rates['InadequateGeneralization_rate']} ({total_counts['InadequateGeneralization']})",
        'CompleteMastery (Strict)': f"{rates['CompleteMastery_strict_rate']} ({total_counts['CompleteMastery_strict']})",
        'RoteMemorization (Strict)': f"{rates['RoteMemorization_strict_rate']} ({total_counts['RoteMemorization_strict']})",

        'Score (Loose)': final_score_loose,
        'InsufficientKnowledge (Loose)': f"{rates['InsufficientKnowledge_rate']} ({total_counts['InsufficientKnowledge']})",
        'InadequateGeneralization (Loose)': f"{rates['InadequateGeneralization_rate']} ({total_counts['InadequateGeneralization']})",
        'CompleteMastery (Loose)': f"{rates['CompleteMastery_loose_rate']} ({total_counts['CompleteMastery_loose']})",
        'RoteMemorization (Loose)': f"{rates['RoteMemorization_loose_rate']} ({total_counts['RoteMemorization_loose']})"
    }
    main_results_df = main_results_df._append(new_row, ignore_index=True)
    return main_results_df

# Main function to evaluate models
def wemath_evaluate_models(model_name, output_json, main_results_csv_path = None):

    main_results_df = pd.DataFrame(columns=['Model', 'Score (Strict)', 'InsufficientKnowledge (Strict)', 'InadequateGeneralization (Strict)', 'CompleteMastery (Strict)', 'RoteMemorization (Strict)', 'Score (Loose)', 'InsufficientKnowledge (Loose)', 'InadequateGeneralization (Loose)', 'CompleteMastery (Loose)', 'RoteMemorization (Loose)'])

    print(f"Evaluating model: {model_name}, JSON path: {output_json}")
    data = load_and_process_data(output_json)
    data_2steps = data[data['key'].str.contains('2steps')]
    data_3steps = data[data['key'].str.contains('3steps')]
    merged_2steps = evaluate_process_steps_data(data_2steps, 2)
    merged_3steps = evaluate_process_steps_data(data_3steps, 3)

    metrics = evaluate_calculate_metrics(merged_2steps, merged_3steps)
    total_counts, rates = evaluate_compute_final_scores(metrics, total_count=525)

    main_results_df = evaluate_update_main_results_df(main_results_df, model_name, total_counts, rates)

    print(main_results_df.to_string(index = False))
    if main_results_csv_path is not None:
        main_results_df.to_csv(main_results_csv_path, index=False)
        print("Evaluation completed and results saved to CSV.")

### Accuracy.py
# Function to load knowledge structure nodes
def load_knowledge_structure_nodes(filepath):
    with open(filepath, "r") as file:
        nodes = json.load(file)
    nodes = pd.DataFrame(nodes)
    nodes['final_key'] = nodes['full node'].str.split('_').str[-1]
    nodes['root_2'] = nodes['full node'].str.split('_').str[1]
    return nodes

# Function to evaluate steps
def accuracy_evaluate_steps(json, steps, nodes):
    jokers = [json[[f'joker_{i}', f'knowledge concept_{i}']] for i in range(1, steps + 1)]
    for i in range(steps):
        jokers[i] = pd.merge(jokers[i], nodes[['final_key', 'full node', 'root_2']],
                             left_on=f'knowledge concept_{i + 1}', right_on='final_key', how='left')
        jokers[i].rename(columns={f'joker_{i + 1}': 'joker', f'knowledge concept_{i + 1}': 'knowledge_concept'}, inplace=True)
    concatenated_steps = pd.concat(jokers, axis=0)
    return concatenated_steps

# # Function to load and process JSON data
# def accuracy_load_and_process_data(filepath):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     df = pd.DataFrame(data)
#     df['processed_answer'] = df['response'].str.split('Answer').str[-1].str.strip().str.replace(r'[>><<:.]', '', regex=True).str.strip()
#     df['processed_answer'] = df['processed_answer'].apply(lambda x: x[0] if x and x[0] in 'ABCDEFGH' else None)
#     df['joker'] = df['processed_answer'] == df['answer']
#     return df

# Function to process steps data and merge results
def accuracy_process_steps_data(df, steps):
    steps_data = {f'{steps}steps_{i}': df[df['key'] == f'{steps}steps_{i}'] for i in range(1, steps + 1)}
    steps_data[f'{steps}steps_multi'] = df[df['key'] == f'{steps}steps_multi']
    for key, data in steps_data.items():
        data.columns = [col + f'_{key.split("_")[-1]}' for col in data.columns]
    merged_data = steps_data[f'{steps}steps_1']
    for i in range(2, steps + 1):
        merged_data = pd.merge(merged_data, steps_data[f'{steps}steps_{i}'], left_on=f'ID_1', right_on=f'ID_{i}', how='left')
    merged_data = pd.merge(merged_data, steps_data[f'{steps}steps_multi'], left_on=f'ID_1', right_on='ID_multi', how='left')
    return merged_data


# Function to update main results DataFrame
def accuracy_update_main_results_df(nodes, main_results_df, model_name, concatenated_data, merged_2steps, merged_3steps):
    One_step_acc = "{:.2%}".format(concatenated_data['joker'].mean())
    Two_step_acc = "{:.2%}".format(merged_2steps['joker_multi'].mean())
    Three_step_acc = "{:.2%}".format(merged_3steps['joker_multi'].mean())

    new_row = {
        'Model': model_name,
        'One-step(S1)': One_step_acc,
        'Two-step(S2)': Two_step_acc,
        'Three-step(S3)': Three_step_acc
}
     # Calculate rates according to Nodes
    nodes['final_rode'] = nodes['full node'].str.split('_').str[-1]
    csv_final_score = concatenated_data.groupby('final_key')['joker'].mean()
    csv_final_score = pd.merge(nodes, csv_final_score, left_on='final_rode', right_on='final_key', how='left')

    new_row.update(csv_final_score.groupby('root2')['joker'].mean().apply(lambda x: "{:.2%}".format(x)).to_dict())
    main_results_df = main_results_df._append(new_row, ignore_index=True)

    return main_results_df

# Main function to evaluate models
def wemath_accuracy(model_name, output_json, knowledge_structure_nodes_path, main_results_csv_path = None):

    nodes = load_knowledge_structure_nodes(knowledge_structure_nodes_path)

    main_results_df = pd.DataFrame(columns=['Model', 'One-step(S1)', 'Two-step(S2)', 'Three-step(S3)','Understanding and Conversion of Units', 'Angles and Length',
 'Calculation of Plane Figures', 'Understanding of Plane Figures',
       'Calculation of Solid Figures', 'Understanding of Solid Figures',
       'Basic Transformations of Figures','Cutting and Combining of Figures',
 'Direction','Position', 'Route Map','Correspondence of Coordinates and Positions'])

    print(f"Evaluating model: {model_name}, JSON path: {output_json}")
    data = load_and_process_data(output_json)
    data_2steps = data[data['key'].str.contains('2steps')]
    data_3steps = data[data['key'].str.contains('3steps')]
    merged_2steps = accuracy_process_steps_data(data_2steps, 2)
    merged_3steps = accuracy_process_steps_data(data_3steps, 3)

    concatenated_data = pd.concat([accuracy_evaluate_steps(merged_2steps, 2, nodes), accuracy_evaluate_steps(merged_3steps, 3, nodes)], axis=0)
    main_results_df = accuracy_update_main_results_df(nodes, main_results_df, model_name, concatenated_data, merged_2steps, merged_3steps)

    print(main_results_df.to_string(index = False))
    if main_results_csv_path is not None:
        main_results_df.to_csv(main_results_csv_path, index=False)
        print("Evaluation completed and results saved to CSV.")