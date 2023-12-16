from vlmeval.smp import *

def generate_list(column_list):
    list_res = []
    for item in column_list:
        if item not in list_res:
            list_res.append(item)
    return list_res

def can_infer_open(prediction, answer):
    prediction = prediction.lower()
    answer = answer.lower()
    if answer in prediction:
        return True
    return False

def generate_result(cate_list_all, cate_name_map, tot, match , hit):
    df_list = []
    for cate in cate_list_all:
        res = defaultdict(list)
        for k in cate:
            res[cate_name_map[cate]].append(k)
            res['tot'].append(tot[k])
            res['match'].append(match[k])
            res['hit'].append(hit[k])
            res['match_rate'].append(match[k] / tot[k] * 100)
            if match[k] == 0:
                res['acc'].append(0)
            else:
                res['acc'].append(hit[k] / match[k] * 100)
        df_list.append(pd.DataFrame(res))
    return df_list
    
def MMMU_eval(result_file):
    data = load(result_file)
    difficulty_list = generate_list(data['topic_difficulty'])
    subfield_list = generate_list(data['subfield'])
    field_list = generate_list(data['field'])
    type_list = generate_list(data['img_type'])
    cate_list_all = [field_list, subfield_list, difficulty_list , type_list]
    from vlmeval.eval.multiple_choice import build_choices, can_infer
    tot = defaultdict(lambda: 0)
    match = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        difficulty = item['topic_difficulty']
        subfield = item['subfield']
        field = item['field']
        img_type = item['img_type']
        cate_list = [difficulty, subfield, field, img_type, cate_list]
        
        tot['Overall'] += 1
        for cate in cate_list:
            tot[cate] += 1
        
        if item['question_type'] == 'multiple-choice':
            choices = build_choices(item)
            matched = can_infer(item['prediction'], choices)
            if matched:
                match['Overall'] += 1
                for cate in cate_list:
                    match[cate] += 1
                if matched == item['answer']:
                    hit['Overall'] += 1
                    for cate in cate_list:
                        hit[cate] += 1
        else:
            can_infer_open = False
            if item['answer'].isinstance(list):
                for answer in item['answer']:
                    if can_infer_open(item['prediction',answer]):
                        can_infer_open = True
                        break
            elif item['answer'].isinstance(str):
                can_infer_open = can_infer_open(item['prediction', item['answer']])
            else:
                print('Unknown answer type!')
            if can_infer_open == True:
                for cate in cate_list:
                    match[cate] += 1
                for cate in cate_list:
                    hit[cate] += 1
        
        cate_name = ['field','subfield','difficulty','img_type']
        cate_name_map = zip(cate_list_all,cate_name)
        df_list = generate_result(cate_list_all, cate_name_map, tot, match, hit)
    dump_path = result_file.replace('.xlsx','_acc.xlsx')
    writer = pd.ExcelWriter(dump_path)
    for i in range(cate_name):
        df_list[i].to_excel(writer, sheet_name = cate_name[i])
    return df_list[0]
