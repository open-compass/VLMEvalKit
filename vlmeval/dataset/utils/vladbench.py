import os
import json
import re
import numpy as np
from vlmeval.smp import misc

def weighted_row_sum(data, third_rows, weight_col=1, start_col=2):
    
    data = np.array(data)
    m,n = data.shape
    rows = slice(m-third_rows, m)
    cols = slice(start_col, None)  
    weighted_sum = np.sum(data[rows, cols].astype(float) * data[rows, weight_col].astype(float)[:, np.newaxis], axis=0) / np.sum(data[rows, weight_col].astype(float))
    weighted_sum = ['Mean',np.sum(data[rows, weight_col].astype(float))] + weighted_sum.tolist()
    temp = data.tolist()
    temp.append(weighted_sum)
    return temp



def weighted_total(data, weight_col=1, start_col=2):
    data = np.array(data)
    m,n = data.shape
    rows = slice(0, m)
    cols = slice(start_col, None)  
    weighted_sum = np.sum(data[rows, cols].astype(float) * data[rows, weight_col].astype(float)[:, np.newaxis], axis=0) / np.sum(data[rows, weight_col].astype(float))
    weighted_sum = ['Total',np.sum(data[rows, weight_col].astype(float))] + weighted_sum.tolist()
    return weighted_sum


def box_iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    xB = min(boxA[2], boxB[2])
    yA = max(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    return iou
def clean_string(s):
    while s and (s[0] in ":[]()' ."):
        s = s[1:]
    while s and (s[-1] in ":[]()' ."):
        s = s[:-1]
    return s

def convert_if_number(answer):
    if isinstance(answer, (int, float)):
        return str(answer)
    return answer

def remove_symbols(input_string):
    input_string = str(input_string)
    if 'correct answer is:' in input_string:
        input_string = input_string.split('correct answer is:')[-1]
    cleaned_string = re.sub(r'[\*\n\""]', '', input_string)
    return cleaned_string

def extract_options(text):
    
    pattern = re.compile(r"\[([^\]]+)\]")  
    matches = pattern.findall(text)

    if matches:
        option_string = matches[-1]  
        if "'" not in option_string:       
            option_list = option_string.split(", ")
        else:
            option_list = [item.strip().strip("'") for item in option_string.split("', '")]  
        return option_list
    return []


def compare_and_count(array_a, array_b):
    count = 0
    for a, b in zip(array_a, array_b):
        if a == 1 and b == 1: count+=1
        if a > b:count+=1
    return count
     
def isfile(path):
    return os.path.isfile(path)


def load_json_data(path):
    with open(path,'r',encoding='utf-8') as json_f:
        task_data = json.load(json_f)
        return task_data
    
def save_json_data(path,data):
    with open(path,'w',encoding='utf-8') as json_f:
        json.dump(data,json_f,ensure_ascii=False,indent=4)
        
def Geneal_criterion_QA(third_task_data,MODEL=None):
    ques_total_num = 0
    right_num = 0
    obey_insytruction = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        for q_ind, pred in enumerate(prediction):
            # print(sample['image_path'])
            ques_nopath = sample['questions'][q_ind].lower()
            tips = extract_options(ques_nopath)
            # print(tips)
            if len(tips)==0: pass
            # if len(tips)!=0: print('No tips',sample['image_path'])
                # print('No tips',sample['image_path'])
                # print(ques_nopath)
            pred = remove_symbols(pred)   
            ques_total_num += 1
            clean_pred = clean_string(pred).lower()  
            options_nums = clean_pred.split("', '")
            reference_q_ind = convert_if_number(reference[q_ind]).lower()
            if len(options_nums)==1: 
                if clean_pred in ques_nopath: 
                    obey_insytruction+=1
                if clean_pred==reference_q_ind:
                    right_num+=1
                elif reference_q_ind in clean_pred:
                    ### filter
                    if reference_q_ind in tips:
                        tips.remove(reference_q_ind)
                        if not any(tip in clean_pred for tip in tips):
                            right_num+=1
    return ques_total_num,right_num/ques_total_num,obey_insytruction/ques_total_num,0


def Grounding_criterion_QA(third_task_data,MODEL=None):
    print('MODEL', MODEL)
    if MODEL ==None:
        print('MODEL Input Lacked')
        return -1 
    resize_model_lists = ["qwen", "internvl", "gemini","DriveMM",'ivl']
    ques_total_num = 0
    right_num = 0
    loc_union = []
    obey_insytruction = 0
    PATTERN = re.compile(r'\[\s*([^\],]*\d+[^\],]*)\s*,\s*([^\],]*\d+[^\],]*)\s*,\s*([^\],]*\d+[^\],]*)\s*,\s*([^\],]*\d+[^\],]*)\s*\]')
    box_num = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        for q_ind, pred in enumerate(prediction):
            # print(sample['image_path'])
            ques_total_num += 1
            ques_nopath = sample['questions'][q_ind].lower()
            if 'located in the image?' in ques_nopath:
                matches = PATTERN.findall(pred)
                cleaned_matches = [[float(re.sub(r'[^0-9.]', '', part)) for part in match] for match in matches]
                if len(matches)==1:
                    box_num+=1
                    obey_insytruction+=1
                    predict_bbox = cleaned_matches[0]
                else:
                    predict_bbox = [0.0, 0.0, 0.0, 0.0]
                
                if sum(predict_bbox) <4:
                    predict_bbox = [x * 1000 for x in predict_bbox]
                if any(mn.lower() in MODEL.lower() for mn in resize_model_lists):
                    bbox_gt = list(map(int, misc.toliststr(sample['reference'][q_ind])))
                    width, height = sample['dimension'][q_ind]
                    width, height = float(width), float(height)
                    bbox_gt = [int(1000*bbox_gt[0]/width), int(1000*bbox_gt[1]/height), int(1000*bbox_gt[2]/width), int(1000*bbox_gt[3]/height)]
                elif MODEL =="gemini":
                    bbox_gt = [bbox_gt[1], bbox_gt[0], bbox_gt[3], bbox_gt[2]]
                else:
                    bbox_gt = sample['reference'][q_ind]
                    
                iou = box_iou(predict_bbox, bbox_gt)
                if iou > 0.5: right_num+=1
                loc_union.append(iou)
            else:
                tips = extract_options(ques_nopath)
                # if len(tips)==0: 
                #     print('No tips',sample['image_path'])
                #     print(sample['questions'][q_ind])
                pred = remove_symbols(pred)  
                clean_pred = clean_string(pred).lower()  
                options_nums = clean_pred.split("', '")
                reference_q_ind = convert_if_number(reference[q_ind]).lower()
                if len(options_nums)==1: 
                    if clean_pred in ques_nopath: 
                        obey_insytruction+=1
                    if clean_pred==reference_q_ind:
                        right_num+=1
                        
                    elif reference_q_ind in clean_pred:
                        ### filter
                        if reference_q_ind in tips:
                            tips.remove(reference_q_ind)
                            if not any(tip in clean_pred for tip in tips):
                                right_num+=1
                                
    mean_iou = sum(loc_union)/len(loc_union)
    return ques_total_num, right_num/ques_total_num, obey_insytruction/ques_total_num, mean_iou


        
def Relation_criterion_QA(third_task_data,MODEL=None):
    ques_total_num = 0
    total_score = 0
    obey_insytruction = 0
    totol_improve_score = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        scores_list = []
        for q_ind, pred in enumerate(prediction):
            ques_total_num+=1
            if 'corresponds to' in pred:
                # pattern = r'(?<!\d)(-?\d+|[0-9]+/[0-9]+)(?!\d)'
                pattern = r'corresponds to No.([+-]?\d+|[+-]?\d+/\d+)'
                match = re.search(pattern, pred)
                if match:
                    pred_num = match.group(1).split('/')
                    # print(pred_num)
                else:
                    pred_num = []
            elif 'corresponding to' in pred:
                pattern = r"corresponding to.*is\s+(-?\d+(?:/\d+)*)"
                match = re.search(pattern, pred)
                if match:
                    pred_num = match.group(1).split("/")
                else:
                    pred_num = []
            else:
                pattern = r"(-?\d+(?:/\d+)*)"
                match = re.findall(pattern, pred)
                if match:
                    obey_insytruction+=1
                    pred_num = match[-1].split("/")
                    # print(pred_num)
                else:
                    pred_num = []
            
            ref_num =  reference[q_ind].split('/')
            if any(p_num not in ref_num for p_num in pred_num):
                scores_list.append(0)
                continue
            else:
                temp = 0
                # for p_num in pred_num:
                #     if p_num in ref_num:
                #         total_score += 1
                #         break
                for p_num in pred_num:
                    if p_num in ref_num:
                        temp += 1/len(ref_num)
                        total_score += 1/len(ref_num)
                scores_list.append(temp)
        scores_list = np.array(scores_list)
        scores = compare_and_count(scores_list[len(scores_list)//2:], scores_list[:len(scores_list)//2])
        totol_improve_score += scores
    return ques_total_num,total_score/ques_total_num,obey_insytruction/ques_total_num,totol_improve_score*2/ques_total_num


    
def RoadChange_criterion_QA(third_task_data,MODEL=None):
    ques_total_num = 0
    right_num = 0
    obey_insytruction = 0
    totol_improve_score = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        scores_list = []
        for q_ind, pred in enumerate(prediction):
            ques_nopath = sample['questions'][q_ind].lower()
            tips = extract_options(ques_nopath)
            pred = remove_symbols(pred)   
            ques_total_num += 1
            clean_pred = clean_string(pred).lower()  
            options_nums = clean_pred.split("', '")
            reference_q_ind = convert_if_number(reference[q_ind]).lower()
            if len(options_nums)==1: 
                if clean_pred in ques_nopath: 
                    obey_insytruction+=1
                if clean_pred==reference_q_ind:
                    right_num+=1
                    scores_list.append(1)
                elif reference_q_ind in clean_pred:
                    ### filter
                    if reference_q_ind in tips:
                        tips.remove(reference_q_ind)
                        if not any(tip in clean_pred for tip in tips):
                            right_num+=1
                            scores_list.append(1)
                        else:
                            scores_list.append(0)
                    else:
                        scores_list.append(0)
                else:
                    scores_list.append(0)
        scores_list = np.array(scores_list)
        scores = compare_and_count(scores_list[len(scores_list)//2:], scores_list[:len(scores_list)//2])
        totol_improve_score += scores
                      
    return ques_total_num,right_num/ques_total_num,obey_insytruction/ques_total_num,totol_improve_score*2/ques_total_num

def RoadSpeed_criterion_QA(third_task_data,MODEL=None):
    ques_total_num = 0
    right_num = 0
    obey_insytruction = 0
    totol_improve_score = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        scores_list = []
        for q_ind, pred in enumerate(prediction):
            ques_total_num+=1
            pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'  
            matches = re.findall(pattern, pred)
            
            matches_gt = re.findall(pattern, reference[q_ind])
            # print(reference[q_ind])
            ref_gt = [matches_gt[0][0],matches_gt[0][1]]
            # print(ref_gt)
            temp = 0
            if len(matches)==1:
                pred_limit = [matches[0][0],matches[0][1]]
                obey_insytruction+=1
                for a, b in zip(ref_gt,pred_limit):
                    if a==b:
                        temp+=0.5
            right_num+=temp
            scores_list.append(temp)
            
        scores_list = np.array(scores_list)
        scores = compare_and_count(scores_list[len(scores_list)//2:], scores_list[:len(scores_list)//2])
        totol_improve_score += scores
        
    return ques_total_num,right_num/ques_total_num,obey_insytruction/ques_total_num,totol_improve_score*2/ques_total_num
    # return ques_total_num,right_num,obey_insytruction,totol_improve_score/2


      
def Judge_criterion_QA(third_task_data,MODEL=None):
    des_ques_total_num = 0
    judge_ques_total_num = 0
    des_right_num = 0
    judge_right_num = 0
    obey_insytruction = 0
    for d_ind, sample in enumerate(third_task_data):
        reference = sample['reference']
        prediction = sample['prediction']
        for q_ind, pred in enumerate(prediction):
            ques_nopath = sample['questions'][q_ind].lower()
            tips = extract_options(ques_nopath)
            # print(tips)
            if len(tips)==0: pass
            pred = remove_symbols(pred)   
            clean_pred = clean_string(pred).lower()  
            options_nums = clean_pred.split("', '")
            # reference_q_ind = convert_if_number(reference[q_ind]).lower()
            reference_q_ind = clean_string(convert_if_number(reference[q_ind])).lower()
            if 'yes' == reference_q_ind or 'no'  == reference_q_ind:
                judge_ques_total_num+=1
            else:
                des_ques_total_num += 1
            if len(options_nums)==1: 
                # if clean_pred in ques_nopath: 
                if ''.join(clean_pred.split(';')) in ques_nopath: 
                    obey_insytruction+=1
                if clean_pred==reference_q_ind:
                    if 'yes' == reference_q_ind or 'no'  == reference_q_ind:
                        judge_right_num+=1
                    else:
                        des_right_num+=1
                elif reference_q_ind in clean_pred:
                    ### filter
                    if reference_q_ind in tips:
                        tips.remove(reference_q_ind)
                        if not any(tip in clean_pred for tip in tips):
                            if 'yes' == reference_q_ind or 'no'  == reference_q_ind:
                                judge_right_num+=1
                            else:
                                des_right_num+=1
                else:
                    pass
    if des_ques_total_num==0:
        return (judge_ques_total_num+des_ques_total_num),des_right_num,obey_insytruction/(judge_ques_total_num+des_ques_total_num),judge_right_num/judge_ques_total_num
    else:
        return (judge_ques_total_num+des_ques_total_num),des_right_num/des_ques_total_num,obey_insytruction/(judge_ques_total_num+des_ques_total_num),judge_right_num/judge_ques_total_num
        

func_mapping = {
    'Pavement_Marking': Geneal_criterion_QA,
    'Traffic_Sign': Geneal_criterion_QA,
    'Traffic_Light': Geneal_criterion_QA,
    'Right_Of_Way': Geneal_criterion_QA,
    'Light': Geneal_criterion_QA,
    'Weather': Geneal_criterion_QA,
    'Lane_Recognition': Geneal_criterion_QA,
    'Vehicle_Status': Geneal_criterion_QA,
    'Vehicle_Recognition': Grounding_criterion_QA,
    'VRU_Recognition': Grounding_criterion_QA,
    'Obstruction_Recognition': Grounding_criterion_QA,
    'Light_Lane_Relation': Relation_criterion_QA,
    'Sign_Sign_Relation': Relation_criterion_QA,
    'Sign_Lane_Relation': Relation_criterion_QA,
    'Lane_Change_Relation': RoadChange_criterion_QA,
    'Lane_Speed_Relation': RoadSpeed_criterion_QA,
    'VRU_Cutin': Judge_criterion_QA,
    'Vehicle_Cutin': Judge_criterion_QA,
    'VRU_Cross': Judge_criterion_QA,
    'Long_Short_Parking':Geneal_criterion_QA,
    'Vehicle_Bahavior': Geneal_criterion_QA,
    'VRU_Bahavior': Geneal_criterion_QA,
    'Key_Obsturction_Detection': Judge_criterion_QA,
    'Spatial_Temporal_Reasoning': Judge_criterion_QA,
    'Risk_Prediction': Judge_criterion_QA,
    'Drive_Efficiency': Geneal_criterion_QA,
    'Longitudinal': Geneal_criterion_QA,
    'Lateral': Geneal_criterion_QA
}
