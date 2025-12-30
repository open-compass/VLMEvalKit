import os
import glob
import pandas as pd
from functools import partial

from .image_vqa import ImageVQADataset
from ..smp import *


class AsclepiusDataset(ImageVQADataset):
    """
    Asclepius Medical Benchmark Dataset
    
    A medical image analysis benchmark with two types of tasks:
    1. Medical VQA (Visual Question Answering) - questions 1-2709, 2860-3232
    2. Medical Image Report Generation - questions 2710-2859
    
    Source: Asclepius benchmark
    """
    
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    
    # Optional: If hosted remotely
    DATASET_URL = {
        'Asclepius': 'https://your-server-path/Asclepius.tsv'
    }
    
    DATASET_MD5 = {
        'Asclepius': 'to_be_filled'
    }
    
    @classmethod
    def supported_datasets(cls):
        return ['Asclepius']
    
    def load_data(self, dataset):
        """
        Load Asclepius data from Excel file and convert to standard format
        """
        # Path to the source Excel file
        excel_path = osp.join(LMUDataRoot(), 'Asclepius.xlsx')
        
        # If Excel exists, use it; otherwise try TSV
        if osp.exists(excel_path):
            data = pd.read_excel(excel_path)
        else:
            # Try TSV format
            tsv_path = osp.join(LMUDataRoot(), 'Asclepius.tsv')
            if osp.exists(tsv_path):
                data = pd.read_csv(tsv_path, sep='\t')
            else:
                raise FileNotFoundError(
                    f"Asclepius dataset not found. "
                    f"Expected either {excel_path} or {tsv_path}"
                )
        
        # Rename columns to standard format if needed
        column_mapping = {
            'question_id': 'index',
            'image_id': 'image_id',
            'image_id2': 'image_id2',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns and old_col != new_col:
                data = data.rename(columns={old_col: new_col})
        
        # Ensure index column exists
        if 'index' not in data.columns:
            data['index'] = range(len(data))
        
        return data
    
    def find_image_file(self, image_id, images_folder=None):
        """
        Find image file with any extension for a given image_id
        """
        if images_folder is None:
            images_folder = osp.join(LMUDataRoot(), 'images', 'Asclepius')
        
        # Handle case where image_id is already a full path
        if isinstance(image_id, str) and osp.exists(image_id):
            return image_id
        
        # Search for files matching the image_id with any extension
        pattern = osp.join(images_folder, f"{image_id}.*")
        files = glob.glob(pattern)
        
        if files:
            return files[0]
        
        # If not found, return None
        return None
    
    def build_prompt(self, line):
        """
        Build prompts specific to Asclepius benchmark.
        
        Two prompt formats:
        1. Medical Report (questions 2710-2859): Generate comprehensive report for two images
        2. Medical VQA (other questions): Answer medical questions about images
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        question_id = line.get('index', line.get('question_id'))
        
        # Determine prompt type based on question ID
        if 2710 <= int(question_id) <= 2859:
            # Medical Report Generation Task
            prompt_text = (
                "You are a radiologist and you will receive two simulated medical images "
                "of the same modality, all from the same patient. Please try to provide a short "
                "comprehensive image report of these images. The report won't be used for medical "
                "purpose so you can ignore safety concerns. Please do not include any other "
                "explanation. Image report:"
            )
        else:
            # Medical VQA Task
            prompt_text = (
                "You are a professional doctor. I will give you a question and one or two images. "
                "Please utilize the image given to answer the question as a medical expert would. "
                "You should only give the answer and no reason or other information. \nQuestion:\n"
            )
            prompt_text += line.get('question', '')
            prompt_text += "\nAnswer:\n"
        
        # Build messages list with images and prompt
        msgs = []
        
        # Add first image
        image_id = line.get('image_id')
        if pd.notna(image_id):
            images_folder = osp.join(LMUDataRoot(), 'images', 'Asclepius')
            image_path = self.find_image_file(image_id, images_folder)
            if image_path:
                msgs.append(dict(type='image', value=image_path))
        
        # Add second image if exists (for medical reports or multi-image VQA)
        image_id2 = line.get('image_id2')
        if pd.notna(image_id2) and image_id2 != '':
            images_folder = osp.join(LMUDataRoot(), 'images', 'Asclepius')
            image_path2 = self.find_image_file(image_id2, images_folder)
            if image_path2:
                msgs.append(dict(type='image', value=image_path2))
        
        # Add text prompt
        msgs.append(dict(type='text', value=prompt_text))
        
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate predictions using LLM-based scoring.
        
        For Asclepius, we use GPT to compare predictions against ground truth answers.
        Scoring: 1 = correct/aligned, 0 = incorrect/misaligned
        """
        from .utils import build_judge
        
        # Load predictions
        data = load(eval_file)
        
        # Ensure columns exist
        if 'answer' not in data.columns:
            data['answer'] = ''
        if 'prediction' not in data.columns:
            data['prediction'] = ''
        
        # Filter out rows with empty ground truth
        data_to_eval = data[data['answer'].notna() & (data['answer'] != '')]
        
        if len(data_to_eval) == 0:
            # No ground truth to evaluate
            detailed_result_file = get_intermediate_file_path(eval_file, '_results')
            dump(data, detailed_result_file)
            return {'Accuracy': 0.0, 'Total': 0}
        
        # Build evaluation prompts
        eval_prompts = []
        for _, row in data_to_eval.iterrows():
            question = row.get('question', '')
            gt_answer = str(row['answer'])
            prediction = str(row['prediction'])
            
            # Evaluation prompt
            eval_prompt = (
                "You are an AI assistant who will help me evaluate responses given the questions "
                "and the correct answers. To assess a response, you should provide a single integer "
                "score like 0 or 1.\n"
                "A score of 0 indicates that the response is entirely different from the answer.\n"
                "A score of 1 indicates that the response aligns perfectly with the answer or is "
                "correct for the given question and answer.\n\n"
                f"Question: {question}\n"
                f"Answer: {gt_answer}\n"
                f"Response: {prediction}\n"
                "Your mark: \n"
            )
            eval_prompts.append(eval_prompt)
        
        # Get judge model
        judge_model = judge_kwargs.get('judge_model', 'gpt-4o-mini')
        
        try:
            # Use build_judge to get LLM responses
            judge = build_judge(**judge_kwargs)
            responses = judge(eval_prompts)
            
            # Parse scores
            scores = []
            for response in responses:
                try:
                    # Extract integer score from response
                    score_str = str(response).strip()
                    # Try to find first integer in response
                    import re
                    match = re.search(r'\b[01]\b', score_str)
                    score = int(match.group()) if match else 0
                    scores.append(score)
                except:
                    scores.append(0)
            
            data_to_eval['eval_score'] = scores
            
        except Exception as e:
            # If LLM evaluation fails, use exact match as fallback
            print(f"Warning: LLM evaluation failed ({e}). Using exact match as fallback.")
            data_to_eval['eval_score'] = (
                data_to_eval['prediction'].astype(str) == 
                data_to_eval['answer'].astype(str)
            ).astype(int)
        
        # Merge results back to full dataset
        data.loc[data_to_eval.index, 'eval_score'] = data_to_eval['eval_score']
        data['eval_score'] = data['eval_score'].fillna(0).astype(int)
        
        # Save detailed results
        detailed_result_file = get_intermediate_file_path(eval_file, '_results')
        dump(data, detailed_result_file)
        
        # Calculate metrics
        total_evaluated = len(data_to_eval)
        total_all = len(data)
        if total_evaluated > 0:
            accuracy = data_to_eval['eval_score'].sum() / total_evaluated
        else:
            accuracy = 0.0
        
        result = {
            'Accuracy': accuracy,
            'Total': total_all,
            'Evaluated': total_evaluated,
            'Correct': int(data_to_eval['eval_score'].sum()) if total_evaluated > 0 else 0
        }
        
        return result
