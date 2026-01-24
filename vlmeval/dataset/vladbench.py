import os
import json
from typing import Dict, List, Tuple, Any, Union
import pandas as pd
import warnings
import ast
import math
from openai import OpenAI
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import misc, file
from vlmeval.smp.file import get_intermediate_file_path
from vlmeval.dataset.utils.vladbench import *
from tqdm import tqdm
import pdb
from pathlib import Path
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download

all_tasks = {
    "Traffic_Knowledge_Understanding": {
        "Road_Traffic_Signals": [
            "Traffic_Light",
            "Pavement_Marking",
            "Traffic_Sign"
        ],
        "Road_Passage_Provisions": [
            "Right_Of_Way"
        ]
    },
    "General_Element_Recognition": {
        "Foreground": [
            "VRU_Recognition",
            "Vehicle_Recognition",
            "Vehicle_Status",
            "Lane_Recognition",
            "Obstruction_Recognition"
        ],
        "Background": [
            "Light",
            "Weather"
        ]
    },
    "Traffic_Graph_Generation": {
        "Signal_Element_Relation": [
            "Sign_Sign_Relation",
            "Sign_Lane_Relation",
            "Light_Lane_Relation"
        ],
        "Lane_Element_Relation": [
            "Lane_Speed_Relation",
            "Lane_Change_Relation"
        ]
    },
    "Target_Attribute_Comprehension": {
        "Intention_Judgment": [
            "VRU_Cutin",
            "Vehicle_Cutin",
            "VRU_Cross",
            "Long_Short_Parking"
        ],
        "Behavior_Understanding": [
            "Vehicle_Bahavior",
            "VRU_Bahavior"
        ]
    },
    "Ego_Decision_Planning": {
        "Ego_Action_Reasoning": [
            "Key_Obsturction_Detection",
            "Spatial_Temporal_Reasoning",
            "Risk_Prediction",
            "Drive_Efficiency"
        ],
        "Meta_Action_Decision": [
            "Longitudinal",
            "Lateral"
        ],
        "Ego_trajectory_Planning":[
            "Trajectory"
        ]
    }
}

weights = {
    'Vehicle_Recognition':[0.3,0.5,0.2],
    'VRU_Recognition':[0.3,0.5,0.2],
    'Obstruction_Recognition':[0.3,0.5,0.2],
    'Sign_Sign_Relation':[0.3,0.5,0.2],
    'Sign_Lane_Relation':[0.3,0.5,0.2],
    'Light_Lane_Relation':[0.3,0.5,0.2],
    'Lane_Speed_Relation':[0.3,0.5,0.2],
    'Lane_Change_Relation':[0.3,0.5,0.2],
    'VRU_Cutin':[0.7,0.1,0.2],
    'Vehicle_Cutin':[0.7,0.1,0.2],
    'VRU_Cross':[0.7,0.1,0.2],
    'Key_Obsturction_Detection':[0.8,0,0.2],
    'Risk_Prediction':[0.7,0.1,0.2],
    'Spatial_Temporal_Reasoning':[0.4,0.4,0.2]
}

def get_vladbench_image_dir(image_dir=None):
    """
    Resolve VLADBench image directory.
    Priority:
    1. User provided local directory
    2. Auto-download from HuggingFace
    """
    if image_dir is not None:
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"VLADBench image dir not found: {image_dir}")
        return image_dir

    return download_vladbench_from_hf()

def download_vladbench_from_hf():
    local_root = os.path.expanduser("~/.cache/vlmeval/vladbench/")

    if os.path.exists(local_root):
        return local_root

    print("VLADBench not found locally, downloading from HuggingFace...")

    snapshot_download(
        repo_id="depth2world/VLADBench",
        repo_type="dataset",
        local_dir=local_root,
        local_dir_use_symlinks=False,
    )
    return local_root

class VLADBench(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "VLADBench": "https://huggingface.co/datasets/depth2world/VLADBench/blob/main/VLADBench.tsv",
                }
    DATASET_MD5 = {"VLADBench": "53c01aa8f9afe2d084728fc8eb21caea",
                   }
    
    IMAGE_DIR = None  # e.g., your_localdir/VLADBench/
    """
    Resolve VLADBench image directory.
    Priority:
    1. User provided local directory
    2. Auto-download from HuggingFace
    """
    IMAGE_DIR = get_vladbench_image_dir(IMAGE_DIR)

    def build_prompt(self, line: Union[int, pd.Series]) -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = misc.toliststr(line["image"])
        question = line['question']
        # form messages
        msgs = []
        if isinstance(tgt_path, list):
            # for p in tgt_path:
                # print(os.path.join(self.IMAGE_DIR, p))
            msgs.extend([dict(type='image', value=os.path.join(self.IMAGE_DIR, p)) for p in tgt_path])

        else:
            # print(os.path.join(self.IMAGE_DIR, tgt_path))
            msgs = [dict(type='image', value=os.path.join(self.IMAGE_DIR, tgt_path))]
        msgs.append(dict(type='text', value=question))

        return msgs

    def get_scores(self, result_file: str) -> pd.DataFrame:
        data = file.load(result_file)
        model_name = os.path.basename(result_file).split('_')[0]
        
        all_results = []
        total_results = [] 
        for fir_ind, fir_task in enumerate(all_tasks):
            sec_tasks = all_tasks[fir_task]
            for sec_ind, sec_task in enumerate(sec_tasks):
                if sec_task=='Ego_trajectory_Planning':continue
                third_tasks = sec_tasks[sec_task]
                third_rows = 0
                for third_ind, third_task in enumerate(third_tasks):
                    # filter samples of third task
                    filter_data = data[data['category3']==third_task]
                    # prepare data structure for evaluation: list(dict(list))
                    third_task_data = [] 
                    same_vision_qas = {'reference': [], 'prediction':[], 'questions':[], 'dimension':[]}
                    dindex = 0
                    for index, row in filter_data.iterrows():
                        if dindex != row['dindex']:
                            third_task_data.append(same_vision_qas)
                            same_vision_qas = {'reference': [], 'prediction':[], 'questions':[], 'dimension':[]}
                        same_vision_qas['reference'].append(row['answer'])
                        same_vision_qas['prediction'].append(row['prediction'])
                        same_vision_qas['questions'].append(row['question'])
                        same_vision_qas['dimension'].append(misc.toliststr(row['dimension']))
                        dindex = row['dindex']
                    third_task_data.append(same_vision_qas)

                    # compute score
                    third_rows +=1
                    model_scores = [third_task]
                    ques_total_num, right_num, obey_instruction, others = func_mapping[third_task](third_task_data, model_name)

                    # weighted sum score
                    if third_task in weights:
                        weight = weights[third_task]
                    else:
                        weight = [0, 0.8, 0.2]
                    temp_score = 100*others*weight[0] + 100*right_num*weight[1] + 100*obey_instruction*weight[2]
                    
                    model_scores.append(temp_score)
                    model_scores.insert(1,ques_total_num)
                    all_results.append(model_scores)
                    total_results.append(model_scores)
                    
                all_results = weighted_row_sum(all_results,third_rows)
                
        total_ = weighted_total(total_results)
        all_results.append(total_)
        df = pd.DataFrame(all_results, columns=['Task','num', model_name])
        return df

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """
        Evaluate model predictions on the ChartQAPro dataset.

        Args:
            eval_file: Path to the file containing model predictions
            **judge_kwargs: Additional arguments for the judge model

        Returns:
            DataFrame with evaluation scores by category
        """
        print(eval_file)
        score = self.get_scores(eval_file)
        score_file = get_intermediate_file_path(eval_file, f'_score', 'csv')
        file.dump(score, score_file)
        return score